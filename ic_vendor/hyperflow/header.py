# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
#
# `HyperFlowMetadata.from_dict` is ported from HyperFlow's `lora.py` (https://github.com/Video-Rebirth/hyperflow,
# Copyright 2026 The HyperFlow authors, Apache License 2.0). See NOTICE.
"""The self-describing safetensors header of a HyperFlow weights file."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .schedule import validate_sigmas

REQUIRED_METADATA = ("hyperflow", "hyperflow_version", "hyperflow_gate", "lora_alpha", "base_model")


@dataclass(frozen=True)
class HyperFlowMetadata:
    version: str
    gate: float
    lora_alpha: float
    base_model: str
    base_model_revision: str | None = None
    lora_rank: int | None = None
    sigmas: tuple[float, ...] | None = None
    video_shift: float | None = None
    audio_shift: float | None = None
    tasks: tuple[str, ...] = ()
    raw: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, metadata: dict[str, str] | None) -> "HyperFlowMetadata":
        metadata = dict(metadata or {})
        missing = [key for key in REQUIRED_METADATA if key not in metadata]
        if missing or metadata.get("hyperflow", "").lower() != "true":
            raise ValueError(
                "Not a HyperFlow weights file: the safetensors header must carry "
                f'{list(REQUIRED_METADATA)} with hyperflow == "true"; missing {missing or ["hyperflow=true"]}. '
                "Pick the HyperFlow file (e.g. minimax_h3_hyperflow_8step_v1.0.safetensors), not another H3 LoRA."
            )
        sigmas = _json_tuple(metadata.get("hyperflow_sigmas"), float)
        if sigmas is not None:
            sigmas = tuple(validate_sigmas(sigmas))
        return cls(
            version=metadata["hyperflow_version"],
            gate=float(metadata["hyperflow_gate"]),
            lora_alpha=float(metadata["lora_alpha"]),
            base_model=metadata["base_model"],
            base_model_revision=metadata.get("base_model_revision"),
            lora_rank=int(metadata["lora_rank"]) if "lora_rank" in metadata else None,
            sigmas=sigmas,
            video_shift=float(metadata["hyperflow_video_shift"]) if "hyperflow_video_shift" in metadata else None,
            audio_shift=float(metadata["hyperflow_audio_shift"]) if "hyperflow_audio_shift" in metadata else None,
            tasks=_json_tuple(metadata.get("tasks"), str) or (),
            raw=metadata,
        )


def _json_tuple(value, cast):
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list in the safetensors header, got {value!r}.")
    return tuple(cast(item) for item in parsed)


def read_metadata(path: str) -> HyperFlowMetadata:
    """Read and validate the header without loading any tensor."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return HyperFlowMetadata.from_dict(handle.metadata())
