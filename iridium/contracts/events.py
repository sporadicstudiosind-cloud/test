"""Event envelope validation and exact payload storage.

Architecture §5.1 and Invariant 3: exact physical data survives compression.
The compressed latent path is for reasoning; the exact array path is for
computation, export, and every numeric claim in a report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"

_VALIDATORS: dict[str, Draft202012Validator] = {}


class EventError(ValueError):
    pass


def validator(name: str) -> Draft202012Validator:
    if name not in _VALIDATORS:
        with open(SCHEMA_DIR / f"{name}.json", "r", encoding="utf-8") as handle:
            _VALIDATORS[name] = Draft202012Validator(json.load(handle))
    return _VALIDATORS[name]


def validate(envelope: dict[str, Any], schema: str = "event.v1") -> None:
    """Raise ``EventError`` listing every schema violation, not just the first."""
    errors = sorted(validator(schema).iter_errors(envelope), key=lambda e: list(e.path))
    if errors:
        detail = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors
        )
        raise EventError(f"{schema} validation failed: {detail}")


def digest(array: np.ndarray) -> str:
    """Content digest over dtype, shape and bytes, in C order."""
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype.str).encode())
    hasher.update(str(contiguous.shape).encode())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


class ExactStore:
    """Content-addressed storage for exact arrays.

    Payloads are written once and referenced by digest. Round-tripping an array
    through the store must be bit-exact: this is the property that keeps a
    lossy learned representation from becoming the only record of a result.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, array: np.ndarray) -> tuple[str, str]:
        """Store an array; return ``(uri, sha256)``."""
        contiguous = np.ascontiguousarray(array)
        sha = digest(contiguous)
        path = self.root / f"{sha}.npy"
        if not path.exists():
            np.save(path, contiguous, allow_pickle=False)
        return (f"exact://{sha}", sha)

    def get(self, uri: str) -> np.ndarray:
        if not uri.startswith("exact://"):
            raise EventError(f"not an exact-store uri: {uri}")
        sha = uri[len("exact://"):]
        path = self.root / f"{sha}.npy"
        if not path.exists():
            raise EventError(f"payload not found: {uri}")
        array = np.load(path, allow_pickle=False)
        if digest(array) != sha:
            raise EventError(f"payload digest mismatch for {uri}")
        return array


@dataclass
class Event:
    envelope: dict[str, Any]
    store: ExactStore

    @classmethod
    def create(
        cls,
        store: ExactStore,
        *,
        event_id: str,
        workspace_id: str,
        stream_id: str,
        source_role: str,
        modality: str,
        sequence_number: int,
        event_time_ns: int,
        arrival_time_ns: int,
        clock_id: str,
        array: np.ndarray,
        axes: list[str],
        units: dict[str, Any] | None = None,
        channel_schema: list[dict[str, Any]] | None = None,
        scenario_revision: str | None = None,
        coordinate_frame_id: str | None = None,
        producer: str = "test_harness",
    ) -> "Event":
        uri, sha = store.put(array)
        envelope: dict[str, Any] = {
            "schema_version": "iridium.event.v1",
            "event_id": event_id,
            "workspace_id": workspace_id,
            "stream_id": stream_id,
            "scenario_revision": scenario_revision,
            "source_role": source_role,
            "modality": modality,
            "sequence_number": sequence_number,
            "event_time_ns": event_time_ns,
            "arrival_time_ns": arrival_time_ns,
            "clock_id": clock_id,
            "coordinate_frame_id": coordinate_frame_id,
            "payload": {
                "uri": uri,
                "sha256": sha,
                "dtype": str(np.asarray(array).dtype),
                "shape": list(np.asarray(array).shape),
                "axes": axes,
            },
            "units": units,
            "provenance": {"producer": producer, "parent_events": []},
        }
        if channel_schema is not None:
            envelope["channel_schema"] = channel_schema
        validate(envelope)
        return cls(envelope, store)

    def array(self) -> np.ndarray:
        payload = self.envelope["payload"]
        array = self.store.get(payload["uri"])
        if list(array.shape) != list(payload["shape"]):
            raise EventError("payload shape does not match envelope")
        if str(array.dtype) != payload["dtype"]:
            raise EventError("payload dtype does not match envelope")
        return array

    @property
    def is_instruction(self) -> bool:
        """Only an authorized instruction may direct action.

        Architecture §12.4: text inside observed or generated content is data.
        This property is the single place that distinction is read from.
        """
        return self.envelope["source_role"] == "instruction"

    @property
    def latency_ns(self) -> int:
        return self.envelope["arrival_time_ns"] - self.envelope["event_time_ns"]


def assert_causal(events: list[Event], commit_arrival_ns: int) -> None:
    """Reject any event that had not arrived when an output was committed.

    Architecture Invariant 5. Note this uses *arrival* time, not event time: an
    event whose timestamp belongs in the past but which arrived late cannot
    have influenced an earlier committed output.
    """
    late = [
        e.envelope["event_id"]
        for e in events
        if e.envelope["arrival_time_ns"] > commit_arrival_ns
    ]
    if late:
        raise EventError(
            f"events not available at commit time were included: {late}"
        )
