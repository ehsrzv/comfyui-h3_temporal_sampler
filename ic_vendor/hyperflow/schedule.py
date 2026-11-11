# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
#
# `DEFAULT_SIGMAS_8STEP`, `validate_sigmas` and `shift_sigmas` are ported from HyperFlow's `schedule.py`
# (https://github.com/Video-Rebirth/hyperflow, Copyright 2026 The HyperFlow authors, Apache License 2.0), which in
# turn derives them from diffusers' `MiniMaxH3Scheduler.set_timesteps`. See NOTICE.
"""The HyperFlow sigma grid. Pure Python: no torch, no ComfyUI."""

from __future__ import annotations

from collections.abc import Sequence

#: Raw (unshifted) grid the adapter was trained on: 9 points, 8 model evaluations. Only a fallback: the loader uses
#: the grid stored in the weights file header (`hyperflow_sigmas`).
DEFAULT_SIGMAS_8STEP: tuple[float, ...] = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0)

#: The shifts the adapter was trained with (video, audio). ComfyUI's MiniMax H3 defaults are the same.
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def validate_sigmas(sigmas: Sequence[float]) -> list[float]:
    """Check a raw rectified-flow grid: strictly decreasing, first point <= 1.0, last point exactly 0.0."""
    grid = [float(s) for s in sigmas]
    if len(grid) < 2:
        raise ValueError(f"A sigma grid needs at least two points, got {len(grid)}.")
    if any(b >= a for a, b in zip(grid, grid[1:])):
        raise ValueError(f"The sigma grid must be strictly decreasing, got {grid}.")
    if grid[0] > 1.0 or grid[-1] != 0.0:
        raise ValueError(f"The sigma grid must start at or below 1.0 and end at exactly 0.0, got {grid}.")
    return grid


def shift_sigmas(sigmas: Sequence[float], shift: float) -> list[float]:
    """The exponential shift ``s * sigma / (1 + (s - 1) * sigma)`` that MiniMax H3's schedulers apply."""
    if shift <= 0:
        raise ValueError(f"`shift` must be positive, got {shift}.")
    return [shift * s / (1.0 + (shift - 1.0) * s) for s in validate_sigmas(sigmas)]


def is_subgrid(sample_sigmas: Sequence[float], grid: Sequence[float], tol: float = 1e-4) -> bool:
    """Whether ``sample_sigmas`` is the grid itself or a contiguous run of it (e.g. after SplitSigmas)."""
    sample = [float(s) for s in sample_sigmas]
    grid = [float(s) for s in grid]
    if len(sample) < 2 or len(sample) > len(grid):
        return False
    for start in range(len(grid) - len(sample) + 1):
        if all(abs(a - b) <= tol * max(1.0, abs(b)) for a, b in zip(sample, grid[start:start + len(sample)])):
            return True
    return False
