"""attributor_node: SQ batch + sensor streams -> attributed SQ batch.

The attributor consumes the geometry-only SQ batches from
``sq_fit_node`` and joins them with the rover's:

* MicaSense Altum-PT multispectral cube (5 reflectance bands +
  thermal) -- yields ``NDVI``, ``NDRE``, ``GNDVI``, ``EVI``,
  ``SURFACE_TEMP_C`` per SQ.
* OceanOptics UV-VIS-NIR spectrometer -- yields a compressed
  :class:`SpectrumPacket` per SQ that the bore-sight ray hits.
* LiDAR intensity (carried as a 4-column ``Float32MultiArray`` of
  ``[x, y, z, intensity]`` in the rover-local frame) -- yields
  ``LIDAR_INTENSITY_MEAN``, ``LIDAR_INTENSITY_STD``, ``POINT_DENSITY``.

This is a thin ROS wrapper around the kernelcal attributor classes
(:class:`LidarIntensityAttributor`, :class:`MicaSenseAttributor`,
:class:`OceanOpticsAttributor`).  Each tick we pull the most recent
sensor message from a small ring buffer, build a fresh
:class:`SQSpatialIndex` from the incoming SQ batch, run the
attributors against the latest sensor frames, and republish the
batch with the per-SQ properties / spectrum filled in.

Because the spectrum packet is binary (96 bytes), we carry it on the
JSON wire as ``spectrum_b64``; the publisher node base64-decodes it
back into a :class:`SpectrumPacket` before handing it to
:func:`pack_superquadric`.
"""

from __future__ import annotations

import base64
import collections
import time
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from .wire import (
    SQ_BATCH_ATTRIBUTED_TOPIC,
    SQ_BATCH_TOPIC,
    SQBatchEnvelope,
)


def _import_kernelcal():
    """Lazy import of kernelcal attribution + geometry."""
    from kernelcal.distinction_game.geometry import (  # noqa: PLC0415
        LidarIntensityAttributor,
        MicaSenseAttributor,
        OceanOpticsAttributor,
        PropertyId,
        SQSpatialIndex,
        Superquadric,
        get_spec,
    )

    return {
        "LidarIntensityAttributor": LidarIntensityAttributor,
        "MicaSenseAttributor": MicaSenseAttributor,
        "OceanOpticsAttributor": OceanOpticsAttributor,
        "PropertyId": PropertyId,
        "SQSpatialIndex": SQSpatialIndex,
        "Superquadric": Superquadric,
        "get_spec": get_spec,
    }


# ---------------------------------------------------------------------------
# Image unpacking helper
# ---------------------------------------------------------------------------


def _image_msg_to_array(msg: Image) -> np.ndarray:
    """Decode a ``sensor_msgs/Image`` into an ``HxWxC`` float array.

    Supports the formats the rover stack actually produces
    (``32FC1``/``32FC3``/``32FCN`` multispectral cubes, ``8UC1``/``8UC3``).
    Avoids the ``cv_bridge`` dependency so the image-handling code can
    be unit-tested without an OpenCV install.
    """
    enc = (msg.encoding or "").lower()
    h, w = int(msg.height), int(msg.width)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc.startswith("32fc"):
        n_ch = int(enc.replace("32fc", "") or "1")
        arr = raw.view(np.float32).reshape(h, w, n_ch)
        return arr.astype(np.float32, copy=True)
    if enc.startswith("8uc"):
        n_ch = int(enc.replace("8uc", "") or "1")
        arr = raw.reshape(h, w, n_ch).astype(np.float32) / 255.0
        return arr
    raise ValueError(f"unsupported image encoding {enc!r}")


# ---------------------------------------------------------------------------
# Sensor caches.  Each one is a tiny ring of (recv_monotonic_s, payload).
# ---------------------------------------------------------------------------


class _Latest:
    """Thread-safe-ish 'newest only' cache with monotonic-time staleness."""

    def __init__(self, max_age_s: float) -> None:
        self.max_age_s = max_age_s
        self._buf: Deque[Tuple[float, Any]] = collections.deque(maxlen=2)

    def push(self, payload: Any) -> None:
        self._buf.append((time.monotonic(), payload))

    def get(self) -> Optional[Any]:
        if not self._buf:
            return None
        ts, payload = self._buf[-1]
        if (time.monotonic() - ts) > self.max_age_s:
            return None
        return payload


# ---------------------------------------------------------------------------
# Attributor node
# ---------------------------------------------------------------------------


class AttributorNode(Node):
    """Attribute SQ batches with property + spectrum trailers."""

    # Default MicaSense band aliases we accept on the cube channel axis.
    # The cube is published as a ``32FC{N}`` Image; its layout YAML in
    # config/earth_rover.yaml names the channel axis order.
    DEFAULT_MICASENSE_BANDS = ["blue", "green", "red", "red_edge", "nir", "thermal_C"]

    def __init__(self) -> None:
        super().__init__("attributor_node")
        self.declare_parameter("input_topic", SQ_BATCH_TOPIC)
        self.declare_parameter("output_topic", SQ_BATCH_ATTRIBUTED_TOPIC)
        self.declare_parameter(
            "micasense_topic", "/earth_rover/micasense/cube"
        )
        self.declare_parameter(
            "oceanoptics_topic", "/earth_rover/oceanoptics/spectrum"
        )
        self.declare_parameter(
            "lidar_xyzi_topic", "/earth_rover/lidar/xyzi"
        )
        self.declare_parameter("enable_micasense", True)
        self.declare_parameter("enable_oceanoptics", True)
        self.declare_parameter("enable_lidar_intensity", True)
        self.declare_parameter("micasense_max_age_s", 1.0)
        self.declare_parameter("oceanoptics_max_age_s", 2.0)
        self.declare_parameter("lidar_max_age_s", 0.5)
        self.declare_parameter(
            "micasense_bands", list(self.DEFAULT_MICASENSE_BANDS)
        )

        # MicaSense intrinsics (3x3 K, row-major) and extrinsics relative
        # to the rover base frame.  The attributor needs these to project
        # SQ silhouettes into the image plane.
        self.declare_parameter(
            "micasense_K",
            [1000.0, 0.0, 320.0, 0.0, 1000.0, 240.0, 0.0, 0.0, 1.0],
        )

        # OceanOptics bore-sight in rover-local frame.
        self.declare_parameter("ocean_bore_origin", [0.0, 0.0, 0.5])
        self.declare_parameter("ocean_bore_direction", [1.0, 0.0, 0.0])
        self.declare_parameter("ocean_lambda_lo_nm", 350.0)
        self.declare_parameter("ocean_lambda_hi_nm", 1100.0)
        self.declare_parameter("ocean_n_channels", 1024)

        self._kc = _import_kernelcal()

        self._micasense = _Latest(
            max_age_s=float(self.get_parameter("micasense_max_age_s").value)
        )
        self._ocean = _Latest(
            max_age_s=float(self.get_parameter("oceanoptics_max_age_s").value)
        )
        self._lidar_xyzi = _Latest(
            max_age_s=float(self.get_parameter("lidar_max_age_s").value)
        )

        self._sub_in = self.create_subscription(
            String,
            str(self.get_parameter("input_topic").value),
            self._on_sq_batch,
            1,
        )
        self._pub_out = self.create_publisher(
            String,
            str(self.get_parameter("output_topic").value),
            1,
        )

        if bool(self.get_parameter("enable_micasense").value):
            self.create_subscription(
                Image,
                str(self.get_parameter("micasense_topic").value),
                self._on_micasense,
                1,
            )
        if bool(self.get_parameter("enable_oceanoptics").value):
            self.create_subscription(
                Float32MultiArray,
                str(self.get_parameter("oceanoptics_topic").value),
                self._on_oceanoptics,
                1,
            )
        if bool(self.get_parameter("enable_lidar_intensity").value):
            self.create_subscription(
                Float32MultiArray,
                str(self.get_parameter("lidar_xyzi_topic").value),
                self._on_lidar_xyzi,
                1,
            )

        self.get_logger().info("AttributorNode ready")

    # ------------------------------------------------------------------
    # Sensor inboxes
    # ------------------------------------------------------------------

    def _on_micasense(self, msg: Image) -> None:
        try:
            self._micasense.push(_image_msg_to_array(msg))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"micasense decode failed: {exc!r}")

    def _on_oceanoptics(self, msg: Float32MultiArray) -> None:
        try:
            self._ocean.push(np.asarray(msg.data, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"oceanoptics decode failed: {exc!r}")

    def _on_lidar_xyzi(self, msg: Float32MultiArray) -> None:
        try:
            arr = np.asarray(msg.data, dtype=np.float32)
            if arr.size % 4 != 0:
                self.get_logger().warn(
                    f"lidar_xyzi: data length {arr.size} not divisible by 4"
                )
                return
            self._lidar_xyzi.push(arr.reshape(-1, 4))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"lidar_xyzi decode failed: {exc!r}")

    # ------------------------------------------------------------------
    # Helpers: SQ reconstruction and band slicing
    # ------------------------------------------------------------------

    def _reconstruct_sq(self, rec: Mapping[str, Any]) -> Any:
        kc = self._kc
        return kc["Superquadric"](
            scale=np.array(rec["scale"], dtype=float),
            epsilon=np.array(rec["epsilon"], dtype=float),
            R=np.array(rec["R"], dtype=float),
            t=np.array(rec["t"], dtype=float),
            id=str(rec["id"]),
            parent_id=rec.get("parent_id"),
        )

    def _bands_from_cube(
        self, cube: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Slice the multispec cube into named (H, W) band arrays."""
        if cube.ndim != 3:
            return {}
        names = list(self.get_parameter("micasense_bands").value)
        out: Dict[str, np.ndarray] = {}
        for i, name in enumerate(names[: cube.shape[2]]):
            if not name:
                continue
            out[str(name)] = cube[..., i].astype(np.float32, copy=False)
        return out

    # ------------------------------------------------------------------
    # Main path
    # ------------------------------------------------------------------

    def _on_sq_batch(self, msg: String) -> None:
        try:
            envelope = SQBatchEnvelope.from_json(msg.data)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"failed to decode SQ batch envelope: {exc}")
            return

        if not envelope.superquadrics:
            # Forward empty batches verbatim so downstream timing stays consistent.
            out = String()
            out.data = envelope.to_json()
            self._pub_out.publish(out)
            return

        kc = self._kc
        sqs = [self._reconstruct_sq(rec) for rec in envelope.superquadrics]
        index = kc["SQSpatialIndex"](sqs=list(sqs))

        # ---- LiDAR intensity ------------------------------------------------
        lidar_attr = None
        if bool(self.get_parameter("enable_lidar_intensity").value):
            xyzi = self._lidar_xyzi.get()
            if xyzi is not None and xyzi.size > 0:
                lidar_attr = kc["LidarIntensityAttributor"](index=index)
                # Cap returns per SQ to keep Welford variance honest under
                # heavily oversampled foreground objects.
                lidar_attr.attribute(xyzi, max_returns_per_sq=4096)

        # ---- MicaSense reflectance + thermal --------------------------------
        mica_attr = None
        if bool(self.get_parameter("enable_micasense").value):
            cube = self._micasense.get()
            if cube is not None and cube.ndim == 3 and cube.shape[2] >= 2:
                bands = self._bands_from_cube(cube)
                if bands:
                    K = np.asarray(
                        list(self.get_parameter("micasense_K").value),
                        dtype=float,
                    ).reshape(3, 3)
                    mica_attr = kc["MicaSenseAttributor"](
                        index=index, K=K, image_shape=cube.shape[:2]
                    )
                    # Camera at rover origin pointing forward; replace with
                    # a real pose lookup once the URDF / TF tree is wired.
                    R_cw = np.eye(3)
                    t_cw = np.zeros(3)
                    mica_attr.attribute(bands, camera_pose=(R_cw, t_cw))

        # ---- OceanOptics spectrum -------------------------------------------
        ocean_attr = None
        if bool(self.get_parameter("enable_oceanoptics").value):
            spectrum = self._ocean.get()
            if spectrum is not None and spectrum.size > 0:
                lo = float(self.get_parameter("ocean_lambda_lo_nm").value)
                hi = float(self.get_parameter("ocean_lambda_hi_nm").value)
                n_ch = int(self.get_parameter("ocean_n_channels").value)
                wavelengths = np.linspace(lo, hi, spectrum.size)
                origin = np.asarray(
                    list(self.get_parameter("ocean_bore_origin").value),
                    dtype=float,
                )
                direction = np.asarray(
                    list(self.get_parameter("ocean_bore_direction").value),
                    dtype=float,
                )
                ocean_attr = kc["OceanOpticsAttributor"](
                    index=index,
                    n_channels=n_ch,
                    lambda_lo_nm=lo,
                    lambda_hi_nm=hi,
                )
                ocean_attr.attribute(
                    spectrum=spectrum,
                    wavelengths_nm=wavelengths,
                    bore_sight_origin=origin,
                    bore_sight_direction=direction,
                )

        # ---- Compose attributed envelope ------------------------------------
        attributed_records: List[Dict[str, Any]] = []
        for rec, sq in zip(envelope.superquadrics, sqs):
            new = dict(rec)
            store = self._merged_store_for(sq.id, lidar_attr, mica_attr, ocean_attr)
            if store is not None:
                props = store.finalize_for_packing()
                if props:
                    # Use the spec name (lowercase, matches the
                    # kernelcal property registry) so the publisher
                    # and the deepgis-xr server can round-trip the
                    # dict back through ``get_spec``.
                    get_spec = self._kc["get_spec"]
                    new["properties"] = {
                        str(get_spec(p).name): float(v) for p, v in props.items()
                    }
                if store.spectrum is not None and store.spectrum.n_samples > 0:
                    packet = store.spectrum.to_packet()
                    new["spectrum_b64"] = base64.b64encode(
                        packet.to_bytes()
                    ).decode("ascii")
            attributed_records.append(new)

        out_env = SQBatchEnvelope(
            stream_id=envelope.stream_id,
            frame=envelope.frame,
            stamp_ns=envelope.stamp_ns,
            superquadrics=attributed_records,
            attributes={
                **envelope.attributes,
                "attributor_micasense": mica_attr is not None,
                "attributor_oceanoptics": ocean_attr is not None,
                "attributor_lidar_intensity": lidar_attr is not None,
            },
        )
        out = String()
        out.data = out_env.to_json()
        self._pub_out.publish(out)

    @staticmethod
    def _merged_store_for(
        sq_id: str,
        lidar_attr,
        mica_attr,
        ocean_attr,
    ):
        """Take the first non-empty store from any of the three attributors.

        The kernelcal attributors keep their own per-attributor
        ``stores: Dict[sq_id, SuperquadricPropertyStore]`` and update
        each Welford incrementally.  For now we cherry-pick a single
        store per SQ; merging across attributors lands together with
        the server-side SuperquadricPropertyStore.merge() in PR-7.
        """
        candidates = [
            getattr(a, "stores", {}).get(sq_id)
            for a in (ocean_attr, mica_attr, lidar_attr)
            if a is not None
        ]
        for store in candidates:
            if store is not None:
                return store
        return None


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = AttributorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
