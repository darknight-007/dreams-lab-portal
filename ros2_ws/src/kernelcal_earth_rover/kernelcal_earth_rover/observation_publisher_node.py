"""observation_publisher_node: attributed SQ batch -> POST /api/v1/observe.

Subscribes to :data:`SQ_BATCH_ATTRIBUTED_TOPIC`, packs each SQ via the
kernelcal binary codec (32 byte primitive + optional parent / property
/ spectrum trailers), and POSTs the concatenated payload to
deepgis-xr's ``/api/v1/observe`` endpoint.

The publisher uses the ``application/octet-stream`` transport with a
JSON envelope in the ``X-Observation-Envelope`` header so the link
doesn't pay the ~33% base64 inflation tax.

Failure / backoff
-----------------
* Transient HTTP errors -- we increment a counter, log at warn, and
  drop the batch.  We do *not* try to buffer on the rover (the link
  is presumed lossy and the next batch carries the next state).
* 4xx responses -- log at error and drop; these are configuration
  bugs, not network issues, and re-sending won't help.
* 5xx responses / connection errors -- log at warn, drop.

A periodic wall-clock-aligned ``tick`` task publishes a heartbeat
``observation_publisher_status`` Float32MultiArray with
``[n_batches_sent, n_batches_failed, last_payload_bytes]`` so a
ground-station dashboard can watch link health without having to
ssh into the rover.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import rclpy
import requests
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from .wire import SQ_BATCH_ATTRIBUTED_TOPIC, SQBatchEnvelope


def _import_kernelcal():
    """Lazy import to keep ROS startup fast."""
    from kernelcal.distinction_game.geometry import (  # noqa: PLC0415
        FrameSpec,
        SpectrumPacket,
        Superquadric,
        pack_superquadric,
        packed_size,
    )

    return {
        "FrameSpec": FrameSpec,
        "SpectrumPacket": SpectrumPacket,
        "Superquadric": Superquadric,
        "pack_superquadric": pack_superquadric,
        "packed_size": packed_size,
    }


class ObservationPublisherNode(Node):
    """Stream attributed SQ batches to deepgis-xr ``POST /api/v1/observe``."""

    def __init__(self) -> None:
        super().__init__("observation_publisher_node")
        self.declare_parameter("input_topic", SQ_BATCH_ATTRIBUTED_TOPIC)
        self.declare_parameter("status_topic", "/earth_rover/observe/status")
        self.declare_parameter(
            "endpoint_url",
            "http://localhost:8000/api/v1/observe/",
        )
        self.declare_parameter("source_id", "earth_rover_01")
        self.declare_parameter("session_id", "")
        self.declare_parameter("auth_token", "")  # optional Bearer
        self.declare_parameter("http_timeout_s", 5.0)
        self.declare_parameter("max_payload_bytes", 1_500_000)
        self.declare_parameter("status_period_s", 5.0)

        self._kc = _import_kernelcal()
        self._lock = threading.Lock()
        self._n_batches_sent = 0
        self._n_batches_failed = 0
        self._last_payload_bytes = 0

        self._sub = self.create_subscription(
            String,
            str(self.get_parameter("input_topic").value),
            self._on_attributed_batch,
            1,
        )
        self._status_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("status_topic").value),
            1,
        )
        self.create_timer(
            float(self.get_parameter("status_period_s").value),
            self._publish_status,
        )

        self.get_logger().info(
            f"ObservationPublisherNode -> "
            f"{self.get_parameter('endpoint_url').value!r} "
            f"as {self.get_parameter('source_id').value!r}"
        )

    # ------------------------------------------------------------------
    # Main path
    # ------------------------------------------------------------------

    def _on_attributed_batch(self, msg: String) -> None:
        try:
            envelope = SQBatchEnvelope.from_json(msg.data)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"failed to decode attributed envelope: {exc}")
            self._bump(failed=True)
            return

        if not envelope.superquadrics:
            return  # nothing to ship

        try:
            payload = self._pack(envelope)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"pack_superquadric failed: {exc}")
            self._bump(failed=True)
            return

        max_bytes = int(self.get_parameter("max_payload_bytes").value)
        if len(payload) > max_bytes:
            self.get_logger().error(
                f"packed payload {len(payload)} bytes > limit {max_bytes}"
            )
            self._bump(failed=True)
            return

        ok = self._post(envelope, payload)
        self._bump(failed=not ok, payload_bytes=len(payload))

    def _pack(self, envelope: SQBatchEnvelope) -> bytes:
        kc = self._kc
        SpectrumPacket = kc["SpectrumPacket"]
        Superquadric = kc["Superquadric"]
        pack = kc["pack_superquadric"]

        out = bytearray()
        for rec in envelope.superquadrics:
            sq = Superquadric(
                scale=np.array(rec["scale"], dtype=float),
                epsilon=np.array(rec["epsilon"], dtype=float),
                R=np.array(rec["R"], dtype=float),
                t=np.array(rec["t"], dtype=float),
                id=str(rec["id"]),
                parent_id=rec.get("parent_id"),
            )
            spectrum = None
            spectrum_b64 = rec.get("spectrum_b64")
            if spectrum_b64:
                try:
                    spectrum = SpectrumPacket.from_bytes(
                        base64.b64decode(spectrum_b64)
                    )
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(
                        f"sq {rec.get('id')!r}: bad spectrum_b64 ({exc!r}); "
                        f"shipping without spectrum"
                    )

            parent_id = rec.get("parent_id")
            parent_hash = (
                hash(str(parent_id)) & 0x7FFFFFFFFFFFFFFF
                if parent_id
                else None
            )
            properties = rec.get("properties") or None
            out += pack(
                sq,
                class_idx=int(rec.get("class_idx") or 0),
                parent_hash=parent_hash,
                properties=properties,
                spectrum=spectrum,
            )
        return bytes(out)

    def _post(self, envelope: SQBatchEnvelope, payload: bytes) -> bool:
        url = str(self.get_parameter("endpoint_url").value)
        timeout = float(self.get_parameter("http_timeout_s").value)
        session_id = (
            str(self.get_parameter("session_id").value).strip()
            or envelope.stream_id
        )
        env_header: Dict[str, Any] = {
            "source_id": str(self.get_parameter("source_id").value),
            "session_id": session_id,
            "frame": envelope.frame,
            "n_superquadrics": len(envelope.superquadrics),
            "payload_size": len(payload),
            "sensor_timestamp": _stamp_ns_to_iso(envelope.stamp_ns),
            "attributes": dict(envelope.attributes),
        }
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Observation-Envelope": json.dumps(env_header),
        }
        token = str(self.get_parameter("auth_token").value).strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            resp = requests.post(
                url, data=payload, headers=headers, timeout=timeout
            )
        except requests.RequestException as exc:
            self.get_logger().warn(f"observe POST failed: {exc!r}")
            return False

        if resp.status_code >= 400:
            self.get_logger().warn(
                f"observe POST returned {resp.status_code}: "
                f"{resp.text[:512]}"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Bookkeeping + heartbeat
    # ------------------------------------------------------------------

    def _bump(self, *, failed: bool, payload_bytes: int = 0) -> None:
        with self._lock:
            if failed:
                self._n_batches_failed += 1
            else:
                self._n_batches_sent += 1
            if payload_bytes:
                self._last_payload_bytes = int(payload_bytes)

    def _publish_status(self) -> None:
        with self._lock:
            sent = self._n_batches_sent
            failed = self._n_batches_failed
            last_bytes = self._last_payload_bytes
        msg = Float32MultiArray()
        msg.data = [float(sent), float(failed), float(last_bytes)]
        self._status_pub.publish(msg)


def _stamp_ns_to_iso(stamp_ns: int) -> str:
    """Convert a monotonic ROS stamp (ns) to an ISO-8601 UTC string.

    The deepgis-xr endpoint expects ISO-8601 here; we use UTC for
    determinism and let the server compare against received_at.
    """
    if not stamp_ns:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    secs = stamp_ns / 1e9
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(secs))


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = ObservationPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
