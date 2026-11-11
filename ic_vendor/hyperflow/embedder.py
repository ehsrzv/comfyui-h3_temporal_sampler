# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
#
# The two-time blend `emb_t(t) + gate * (emb_r(r) - emb_t(t))` follows HyperFlow's `TwoTimeEmbedder`
# (https://github.com/Video-Rebirth/hyperflow, Apache License 2.0), itself after AnyFlow (Gu et al., 2026).
"""Two-time (t, r) conditioning on ComfyUI's native MiniMax H3.

ComfyUI's ``MiniMaxH3Model._forward`` embeds the forward's distinct timesteps with one call,
``self.time_embedder(t_vals)``, and every row of the packed sequence then indexes that table by
``t_row[t] * 3 + modality_tag``. HyperFlow conditions each row on the interval it integrates, so every distinct ``t``
needs its endpoint ``r``:

* generated video rows (and text, which follows video): ``r = 1 - sigma_next``;
* generated audio rows: ``r = 1 - shift_{video -> audio}(sigma_next)``;
* pinned conditioning rows (fl2va keyframes, ref2va references, preserved inpaint rows): ``r = t``;
* partially masked rows, ``t = 1 - m * sigma``: ``r = 1 - m * sigma_next``.

The per-step values come from :class:`StepContext`, set by the diffusion-model wrapper around each forward.
"""

from __future__ import annotations

import logging
import threading

import torch

logger = logging.getLogger("HyperFlow")

#: float32 timesteps are compared at this tolerance. Distinct H3 timesteps are never this close on the HyperFlow grid.
T_MATCH_TOL = 2e-6


class StepContext:
    """What one forward needs to turn its distinct timesteps into endpoints. All sigmas are on the video schedule."""

    __slots__ = ("t_video", "t_audio", "r_video", "r_audio", "pin", "sigma", "sigma_next")

    def __init__(self, t_video, t_audio, r_video, r_audio, pin, sigma, sigma_next):
        self.t_video = float(t_video)
        self.t_audio = float(t_audio)
        self.r_video = float(r_video)
        self.r_audio = float(r_audio)
        self.pin = float(pin)
        self.sigma = float(sigma)
        self.sigma_next = float(sigma_next)


class HyperFlowController:
    """Owns the endpoint embedder and the per-forward context; one per loaded HyperFlow model."""

    def __init__(self, endpoint_module: torch.nn.Module, gate: float, metadata):
        self.endpoint_module = endpoint_module.float().eval().requires_grad_(False)
        self.gate = float(gate)
        self.metadata = metadata
        self._local = threading.local()
        self._device_copies: dict = {}
        self._warned_mask = False

    # -- context -------------------------------------------------------------------------------------------------
    @property
    def context(self) -> StepContext | None:
        return getattr(self._local, "context", None)

    @context.setter
    def context(self, value: StepContext | None) -> None:
        self._local.context = value

    # -- endpoint embedder ---------------------------------------------------------------------------------------
    def _endpoint_on(self, device: torch.device) -> torch.nn.Module:
        key = str(device)
        module = self._device_copies.get(key)
        if module is None:
            import copy

            module = copy.deepcopy(self.endpoint_module).to(device)
            self._device_copies = {key: module}  # keep one device copy (the time embedder is ~60 MB in fp32)
        return module

    # -- t -> r ----------------------------------------------------------------------------------------------------
    def endpoints(self, t_vals: torch.Tensor, ctx: StepContext) -> torch.Tensor:
        ratio = ctx.sigma_next / ctx.sigma if ctx.sigma > 0 else 0.0
        out = []
        for t in t_vals.detach().float().cpu().tolist():
            if abs(t - ctx.t_video) <= T_MATCH_TOL:
                out.append(ctx.r_video)
            elif abs(t - ctx.t_audio) <= T_MATCH_TOL:
                out.append(ctx.r_audio)
            elif t >= ctx.pin - T_MATCH_TOL:
                out.append(t)  # pinned conditioning: an anchor has nowhere to go
            else:
                if not self._warned_mask:
                    self._warned_mask = True
                    logger.warning(
                        "HyperFlow: partially masked rows (denoise mask between 0 and 1) get r = 1 - m * sigma_next "
                        "on the video schedule. HyperFlow was not trained on masked rows; check the output."
                    )
                out.append(1.0 - (1.0 - t) * ratio)
        return torch.tensor(out, dtype=torch.float32, device=t_vals.device)

    def make_forward(self, base_forward):
        """The replacement for ``time_embedder.forward``: ``emb_t + gate * (emb_r - emb_t)``."""

        def forward(t_vals):
            emb_t = base_forward(t_vals)
            ctx = self.context
            if ctx is None:
                raise RuntimeError(
                    "HyperFlow needs the endpoint of this forward, but none was set. The HyperFlow model must be "
                    "sampled through ComfyUI's samplers (its diffusion-model wrapper sets the endpoint per step)."
                )
            r = self.endpoints(t_vals, ctx)
            emb_r = self._endpoint_on(t_vals.device)(r).to(emb_t.dtype)
            return emb_t + self.gate * (emb_r - emb_t)

        return forward
