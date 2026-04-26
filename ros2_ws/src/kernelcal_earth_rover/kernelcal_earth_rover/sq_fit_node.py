"""sq_fit_node: ``sensor_msgs/PointCloud2`` -> superquadric batch.

The fitter is intentionally lightweight: it expects a *pre-clustered*
point cloud (e.g. Euclidean or DBSCAN clusters from a prior node, or
SAM-style instance masks projected from the RGB stream) where each
input point carries a ``cluster_id`` field.  For each cluster it picks
a canonical SQ primitive shape from a small library:

  * ``cylinder`` for tall, narrow clusters (tree trunks, lamp posts);
  * ``ellipsoid`` for blob-like crowns or bushes;
  * ``cuboid`` for axis-aligned boxes (buildings, walls, vehicles).

This is enough to validate the *pipeline* (cluster -> SQ -> attributor
-> wire). Replacing the heuristic with a proper implicit-form fit
(Gauss-Newton on the inside-outside function, e.g.
``kernelcal.distinction_game.geometry.fit_superquadric_implicit``
once it lands) is a contained swap.

Configuration is read from the standard rclpy parameter server, which
launches populate from ``config/earth_rover.yaml``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from .wire import SQ_BATCH_TOPIC, SQBatchEnvelope, SQRecord


def _import_kernelcal():
    """Lazy import to keep ROS startup fast and to make the node
    importable in CI environments without kernelcal."""
    from kernelcal.distinction_game.geometry import (  # noqa: PLC0415
        FrameSpec,
        superquadric_box,
        superquadric_cylinder,
        superquadric_ellipsoid,
    )

    return {
        "FrameSpec": FrameSpec,
        "superquadric_box": superquadric_box,
        "superquadric_cylinder": superquadric_cylinder,
        "superquadric_ellipsoid": superquadric_ellipsoid,
    }


# ---------------------------------------------------------------------------
# PointCloud2 unpacking (no sensor_msgs_py / pcl required)
# ---------------------------------------------------------------------------


_DTYPE_FROM_PC2 = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def _pointcloud2_to_array(msg: PointCloud2) -> np.ndarray:
    """Unpack a PointCloud2 into a structured numpy array.

    We intentionally do not depend on ``sensor_msgs_py.point_cloud2``
    because that pulls a chain of message-generation bits we don't
    need.  Producers are expected to publish point clouds with at
    least ``x``, ``y``, ``z`` floats; ``intensity`` and ``cluster_id``
    are optional.
    """
    fields = []
    for f in msg.fields:
        np_dtype = _DTYPE_FROM_PC2.get(f.datatype)
        if np_dtype is None:
            raise ValueError(f"unsupported PointCloud2 datatype: {f.datatype}")
        fields.append((f.name, np_dtype, f.offset))
    fields.sort(key=lambda fld: fld[2])

    structured: List[Tuple[str, Any]] = []
    cursor = 0
    for name, np_dtype, offset in fields:
        if offset > cursor:
            structured.append((f"_pad_{cursor}", np.uint8, offset - cursor))
            cursor = offset
        structured.append((name, np_dtype))
        cursor += np.dtype(np_dtype).itemsize
    if msg.point_step > cursor:
        structured.append((f"_pad_tail", np.uint8, msg.point_step - cursor))

    dtype = np.dtype(structured)
    n_points = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    arr = raw[: n_points * msg.point_step].view(dtype)
    return arr


# ---------------------------------------------------------------------------
# Cluster -> SQ primitive heuristic
# ---------------------------------------------------------------------------


def _fit_cluster_to_sq(
    points_xyz: np.ndarray,
    cluster_id: int,
    *,
    kc: Dict[str, Any],
) -> Tuple[Any, str]:
    """Pick a primitive shape for a single cluster and fit it.

    Returns ``(superquadric, shape_label)``; ``shape_label`` is the
    name we used so the attributor can apply shape-specific priors.
    """
    if points_xyz.shape[0] < 3:
        raise ValueError("cluster too small to fit")

    centroid = points_xyz.mean(axis=0)
    centered = points_xyz - centroid
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (3, 3):
        # Single-axis clusters: pad so eigh works.
        cov = np.eye(3) * 1e-6
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort eigenvalues descending so eigvecs[:, 0] is the major axis.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    extents = np.array([
        max(float(np.sqrt(max(eigvals[i], 0.0))) * 2.5, 1e-3)
        for i in range(3)
    ])  # ~2.5 sigma envelope per principal axis

    aspect_major = float(extents[0]) / max(float(extents[2]), 1e-6)
    aspect_z = abs(float(eigvecs[2, 0]))  # major axis projection on world z

    # Cylinder: vertically dominant + tall + narrow.
    if aspect_major > 4.0 and aspect_z > 0.7:
        height = float(extents[0])
        radius = float(0.5 * (extents[1] + extents[2]) / 2.0)
        radius = max(radius, 1e-3)
        sq = kc["superquadric_cylinder"](
            base=tuple(centroid - np.array([0.0, 0.0, height / 2.0])),
            axis=(0.0, 0.0, 1.0),
            radius=radius,
            height=height,
        )
        sq.id = f"sq_cyl_{cluster_id}"
        return sq, "cylinder"

    # Cuboid: middle and minor axes are similar magnitude (boxy).
    if (
        extents[2] / max(extents[0], 1e-6) > 0.35
        and extents[1] / max(extents[0], 1e-6) > 0.5
    ):
        sq = kc["superquadric_box"](
            center=tuple(centroid),
            size=tuple(extents.tolist()),
        )
        sq.id = f"sq_box_{cluster_id}"
        return sq, "box"

    # Default: ellipsoid (blob).
    sq = kc["superquadric_ellipsoid"](
        center=tuple(centroid),
        axes=tuple((extents / 2.0).tolist()),
    )
    sq.id = f"sq_ell_{cluster_id}"
    return sq, "ellipsoid"


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------


class SQFitNode(Node):
    """Subscribe to a clustered ``PointCloud2`` and emit SQ batches."""

    def __init__(self) -> None:
        super().__init__("sq_fit_node")
        self.declare_parameter("input_topic", "/earth_rover/lidar/clustered_points")
        self.declare_parameter("output_topic", SQ_BATCH_TOPIC)
        self.declare_parameter("cluster_id_field", "cluster_id")
        self.declare_parameter("min_cluster_points", 12)
        self.declare_parameter("max_clusters_per_batch", 64)
        self.declare_parameter("stream_id", "earth_rover_01")
        self.declare_parameter("frame_kind", "enu_local")
        self.declare_parameter("frame_origin_lla", [0.0, 0.0, 0.0])
        self.declare_parameter("frame_name", "earth_rover/local")

        in_topic = self.get_parameter("input_topic").value
        out_topic = self.get_parameter("output_topic").value
        self._kc = _import_kernelcal()

        self._sub = self.create_subscription(
            PointCloud2, in_topic, self._on_cloud, 1
        )
        self._pub = self.create_publisher(String, out_topic, 1)
        self.get_logger().info(
            f"SQFitNode listening on {in_topic} -> publishing on {out_topic}"
        )

    def _frame_dict(self) -> Dict[str, Any]:
        kind = str(self.get_parameter("frame_kind").value)
        origin = list(self.get_parameter("frame_origin_lla").value)
        name = str(self.get_parameter("frame_name").value)
        if kind == "enu_local":
            return {
                "kind": "enu_local",
                "params": {"origin_lla": [float(c) for c in origin]},
                "name": name,
            }
        if kind == "ecef":
            return {"kind": "ecef", "params": {}, "name": name}
        # Fall back to wgs84_lla; producers shouldn't really do this.
        return {"kind": "wgs84_lla", "params": {}, "name": name}

    def _on_cloud(self, msg: PointCloud2) -> None:
        try:
            t0 = time.monotonic()
            arr = _pointcloud2_to_array(msg)
            if "x" not in arr.dtype.names:
                self.get_logger().warn("PointCloud2 missing x/y/z fields")
                return

            cluster_field = str(self.get_parameter("cluster_id_field").value)
            if cluster_field not in arr.dtype.names:
                self.get_logger().warn(
                    f"PointCloud2 missing cluster id field {cluster_field!r}; "
                    f"falling back to a single-cluster fit"
                )
                cluster_ids = np.zeros(arr.shape[0], dtype=np.int32)
            else:
                cluster_ids = arr[cluster_field].astype(np.int32, copy=False)

            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(
                np.float32, copy=False
            )

            min_pts = int(self.get_parameter("min_cluster_points").value)
            max_clusters = int(self.get_parameter("max_clusters_per_batch").value)

            unique = [int(c) for c in np.unique(cluster_ids) if c >= 0]
            unique = unique[:max_clusters]

            sq_records: List[Dict[str, Any]] = []
            for cid in unique:
                mask = cluster_ids == cid
                if int(mask.sum()) < min_pts:
                    continue
                cluster_xyz = xyz[mask]
                try:
                    sq, shape = _fit_cluster_to_sq(
                        cluster_xyz.astype(np.float64), cid, kc=self._kc
                    )
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(
                        f"cluster {cid}: fit failed ({exc!r}); skipping"
                    )
                    continue

                # Mean radial residual: |t - p|/|t| is too noisy; use
                # ||p - centroid|| / max_extent as a unitless figure.
                mean_residual = float(
                    np.linalg.norm(
                        cluster_xyz - cluster_xyz.mean(axis=0), axis=1
                    ).mean()
                )

                rec = SQRecord(
                    id=str(sq.id),
                    class_idx=_class_idx_for_shape(shape),
                    scale=[float(x) for x in sq.scale.tolist()],
                    epsilon=[float(x) for x in sq.epsilon.tolist()],
                    R=[[float(x) for x in row] for row in sq.R.tolist()],
                    t=[float(x) for x in sq.t.tolist()],
                    cluster_size=int(mask.sum()),
                    fit_residual=mean_residual,
                )
                sq_records.append(rec.__dict__)

            stamp_ns = (
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
            )
            envelope = SQBatchEnvelope(
                stream_id=str(self.get_parameter("stream_id").value),
                frame=self._frame_dict(),
                stamp_ns=stamp_ns,
                superquadrics=sq_records,
                attributes={
                    "n_input_points": int(arr.shape[0]),
                    "fit_ms": int((time.monotonic() - t0) * 1000.0),
                },
            )
            out = String()
            out.data = envelope.to_json()
            self._pub.publish(out)
            self.get_logger().debug(
                f"published {len(sq_records)} SQs from {arr.shape[0]} points"
            )
        except Exception as exc:  # noqa: BLE001 - keep the node alive on bad messages
            self.get_logger().error(f"sq_fit_node failed: {exc!r}")


def _class_idx_for_shape(shape: str) -> int:
    """Stable numeric ids consumed by the attributor / sceneservice.

    Keep these in sync with the attribution config YAML.  The values
    are arbitrary but sticky -- changing them requires re-deploying
    every consumer.
    """
    return {
        "cylinder": 10,
        "ellipsoid": 11,
        "box": 20,
    }.get(shape, 0)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = SQFitNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
