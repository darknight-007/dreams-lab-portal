"""Shared wire formats for the earth_rover edge pipeline.

The three rover nodes (``sq_fit_node`` -> ``attributor_node`` ->
``observation_publisher_node``) talk to each other over plain
``std_msgs/String`` topics carrying JSON envelopes.  We use JSON for
intra-rover hops because the data volume on the rover SoM is
negligible compared to the constrained uplink, and because it keeps
debugging trivial (``ros2 topic echo`` works directly).

Only the *uplink* hop (POST /api/v1/observe) uses the kernelcal binary
codec, where every byte matters.

Two envelope shapes
-------------------

* :data:`SQ_BATCH_TOPIC` (``/earth_rover/sq_batch``) -- emitted by the
  fitter, has only geometry (no properties yet).
* :data:`SQ_BATCH_ATTRIBUTED_TOPIC` (``/earth_rover/sq_batch_attributed``)
  -- emitted by the attributor, same shape plus per-SQ property /
  spectrum payloads.

Both envelopes share these top-level fields:

* ``stream_id`` -- producer mission id.
* ``frame`` -- :class:`kernelcal.distinction_game.geometry.FrameSpec`
  rendered via :meth:`FrameSpec.to_dict` (the local ENU station that
  the rover's SLAM is anchored to).
* ``stamp_ns`` -- ROS time at the head of the batch (int).
* ``superquadrics`` -- list of dicts, see :class:`SQRecord`.

Splitting the envelopes into a small typed helper module here keeps
the node code uncluttered and gives us one place to evolve the
on-the-wire schema if we ever want to rev it (``schema_version``
field is reserved at the top level for that).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


SQ_BATCH_TOPIC: str = "/earth_rover/sq_batch"
SQ_BATCH_ATTRIBUTED_TOPIC: str = "/earth_rover/sq_batch_attributed"

#: Bumped any time the envelope schema (not the kernelcal binary codec)
#: changes incompatibly.  Consumers should refuse mismatched envelopes.
SCHEMA_VERSION: int = 1


@dataclass
class SQRecord:
    """JSON-friendly single-SQ record carried between rover nodes.

    Field naming matches :class:`kernelcal.distinction_game.geometry.Superquadric`
    one-for-one so the attributor and publisher can reconstruct
    ``Superquadric`` instances directly with ``Superquadric(**rec)``-ish
    constructors.
    """

    id: str
    class_idx: int
    scale: List[float]      # length-3
    epsilon: List[float]    # length-2
    R: List[List[float]]    # 3x3
    t: List[float]          # length-3 in the producer frame
    parent_id: Optional[str] = None
    properties: Optional[Dict[str, float]] = None  # "NDVI" -> 0.71, ...
    spectrum: Optional[Dict[str, Any]] = None       # SpectrumPacket.to_dict()
    cluster_size: int = 0   # number of LiDAR points behind the fit
    fit_residual: float = 0.0  # mean radial residual (m)


@dataclass
class SQBatchEnvelope:
    """The full JSON envelope published on the rover-internal topics."""

    stream_id: str
    frame: Dict[str, Any]   # FrameSpec.to_dict()
    stamp_ns: int
    superquadrics: List[Dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    attributes: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "SQBatchEnvelope":
        data = json.loads(raw)
        if not isinstance(data, Mapping):
            raise ValueError("envelope must be a JSON object")
        if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(
                f"envelope schema_version mismatch: "
                f"got {data.get('schema_version')!r}, "
                f"expected {SCHEMA_VERSION}"
            )
        return cls(
            stream_id=str(data["stream_id"]),
            frame=dict(data["frame"]),
            stamp_ns=int(data["stamp_ns"]),
            superquadrics=list(data.get("superquadrics") or []),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            attributes=dict(data.get("attributes") or {}),
        )
