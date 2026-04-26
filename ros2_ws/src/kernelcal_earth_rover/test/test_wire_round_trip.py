"""Offline unit tests for the earth_rover edge pipeline.

These tests run *without* rclpy (no ROS daemon required), so they
verify the parts of the pipeline that are reusable from CI / from a
laptop developer environment:

* The :mod:`kernelcal_earth_rover.wire` envelope round-trips.
* The publisher's pack-payload helper produces bytes the deepgis-xr
  observe endpoint can decode.
* The full earth_rover -> deepgis envelope shape can be recovered
  end-to-end through the kernelcal codec.

Tests skip gracefully if kernelcal isn't installed (e.g. on a CI
worker that only sees the rover code).
"""

from __future__ import annotations

import base64
import os
import sys

import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PKG_ROOT)


kernelcal = pytest.importorskip("kernelcal.distinction_game.geometry")


from kernelcal_earth_rover.wire import (  # noqa: E402  (after sys.path tweak)
    SQ_BATCH_ATTRIBUTED_TOPIC,
    SQ_BATCH_TOPIC,
    SCHEMA_VERSION,
    SQBatchEnvelope,
    SQRecord,
)


# ---------------------------------------------------------------------------
# Wire envelope
# ---------------------------------------------------------------------------


def test_topic_constants_are_distinct() -> None:
    assert SQ_BATCH_TOPIC != SQ_BATCH_ATTRIBUTED_TOPIC
    assert SQ_BATCH_TOPIC.startswith("/")
    assert SQ_BATCH_ATTRIBUTED_TOPIC.startswith("/")


def test_envelope_round_trip_minimal() -> None:
    env = SQBatchEnvelope(
        stream_id="earth_rover_01",
        frame={"kind": "ecef", "params": {}, "name": "test"},
        stamp_ns=1_700_000_000_000_000_000,
    )
    s = env.to_json()
    restored = SQBatchEnvelope.from_json(s)
    assert restored.stream_id == env.stream_id
    assert restored.frame == env.frame
    assert restored.stamp_ns == env.stamp_ns
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.superquadrics == []


def test_envelope_round_trip_with_records() -> None:
    rec = SQRecord(
        id="sq_a",
        class_idx=11,
        scale=[1.0, 1.0, 1.0],
        epsilon=[1.0, 1.0],
        R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t=[3.0, 4.0, 5.0],
        properties={"ndvi": 0.71},
        cluster_size=120,
        fit_residual=0.04,
    )
    env = SQBatchEnvelope(
        stream_id="earth_rover_01",
        frame={
            "kind": "enu_local",
            "params": {"origin_lla": [33.42, -111.94, 350.0]},
        },
        stamp_ns=42,
        superquadrics=[rec.__dict__],
    )
    restored = SQBatchEnvelope.from_json(env.to_json())
    assert len(restored.superquadrics) == 1
    rec2 = restored.superquadrics[0]
    assert rec2["id"] == "sq_a"
    assert rec2["class_idx"] == 11
    assert rec2["properties"] == {"ndvi": 0.71}


def test_envelope_rejects_old_schema() -> None:
    bad = '{"stream_id":"x","frame":{"kind":"ecef","params":{}},' \
          '"stamp_ns":0,"schema_version":99}'
    with pytest.raises(ValueError):
        SQBatchEnvelope.from_json(bad)


# ---------------------------------------------------------------------------
# Payload packing (mirrors observation_publisher_node._pack)
# ---------------------------------------------------------------------------


def _build_local_sqs():
    """Tree (cylinder + crown) and a building, all in ENU local."""
    from kernelcal.distinction_game.geometry import (  # noqa: PLC0415
        superquadric_box,
        superquadric_cylinder,
        superquadric_ellipsoid,
    )

    trunk = superquadric_cylinder(
        base=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        radius=0.20,
        height=4.0,
    )
    crown = superquadric_ellipsoid(center=(0.0, 0.0, 4.0), axes=(2.0, 2.0, 1.5))
    crown.parent_id = trunk.id
    building = superquadric_box(center=(15.0, 12.0, 4.0), size=(8.0, 6.0, 8.0))
    return [trunk, crown, building]


def _pack_envelope_with_publisher(envelope: SQBatchEnvelope) -> bytes:
    """Inline the publisher's _pack() so we don't need rclpy at test time."""
    import numpy as np
    from kernelcal.distinction_game.geometry import (  # noqa: PLC0415
        SpectrumPacket,
        Superquadric,
        pack_superquadric,
    )

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
            spectrum = SpectrumPacket.from_bytes(base64.b64decode(spectrum_b64))
        parent_id = rec.get("parent_id")
        parent_hash = (
            hash(str(parent_id)) & 0x7FFFFFFFFFFFFFFF if parent_id else None
        )
        out += pack_superquadric(
            sq,
            class_idx=int(rec.get("class_idx") or 0),
            parent_hash=parent_hash,
            properties=rec.get("properties") or None,
            spectrum=spectrum,
        )
    return bytes(out)


def test_publisher_payload_round_trips_through_codec() -> None:
    from kernelcal.distinction_game.geometry import (  # noqa: PLC0415
        unpack_superquadric,
    )

    sqs = _build_local_sqs()
    records = [
        SQRecord(
            id=str(sq.id),
            class_idx=ci,
            scale=[float(x) for x in sq.scale.tolist()],
            epsilon=[float(x) for x in sq.epsilon.tolist()],
            R=[[float(x) for x in row] for row in sq.R.tolist()],
            t=[float(x) for x in sq.t.tolist()],
            parent_id=sq.parent_id,
            properties={"ndvi": 0.71} if ci == 11 else None,
            cluster_size=200,
        ).__dict__
        for sq, ci in zip(sqs, (10, 11, 20))
    ]
    envelope = SQBatchEnvelope(
        stream_id="earth_rover_01",
        frame={
            "kind": "enu_local",
            "params": {"origin_lla": [33.4258, -111.94, 350.0]},
        },
        stamp_ns=0,
        superquadrics=records,
    )
    payload = _pack_envelope_with_publisher(envelope)
    assert len(payload) > 0

    cursor = 0
    n = 0
    has_parent_seen = False
    while cursor < len(payload):
        sq, meta = unpack_superquadric(payload[cursor:])
        cursor += int(meta["bytes_consumed"])
        n += 1
        if meta.get("parent_hash") is not None:
            has_parent_seen = True
    assert n == len(records)
    # crown has parent_id -> parent_hash trailer.
    assert has_parent_seen
