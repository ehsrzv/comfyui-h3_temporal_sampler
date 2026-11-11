# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""Putting HyperFlow on a ComfyUI MiniMax H3 ModelPatcher.

Three things, all on a clone, all reversible by ComfyUI's own unpatching:

1. the 314 LoRA targets that exist natively (blocks, token refiner, base time embedder) become ordinary weight
   patches through ``comfy.lora.load_lora`` -- q/k/v as row-slice patches of the fused ``qkv_proj``;
2. an endpoint time embedder (the checkpoint's original time embedder plus the endpoint LoRA, merged, fp32) and an
   object patch on ``time_embedder.forward`` that blends the two embeddings;
3. a diffusion-model wrapper that, per forward, works out each distinct timestep's endpoint from the sampler's sigma
   schedule -- the part ComfyUI's sampler loop never passes on its own.
"""

from __future__ import annotations

import copy
import logging
import types

import torch

from . import keys as hf_keys
from .embedder import HyperFlowController, StepContext
from .header import read_metadata
from .schedule import AUDIO_SHIFT, DEFAULT_SIGMAS_8STEP, VIDEO_SHIFT, is_subgrid, shift_sigmas

logger = logging.getLogger("HyperFlow")

WRAPPER_KEY = "hyperflow"
OPTIONS_KEY = "hyperflow"
#: At sigma = 1 the video and audio timesteps are both 0 and ComfyUI gives them one embedding row, but their
#: endpoints differ. Evaluating that step at sigma * (1 - STEP0_NUDGE) separates the rows (t_video ~ 1e-4,
#: t_audio ~ 4e-4); the sinusoidal input moves by < 1e-3 rad, far below the step's own resolution.
STEP0_NUDGE = 1e-4
_SIGMA_MATCH_TOL = 1e-3


def check_model(model) -> torch.nn.Module:
    """Return the H3 diffusion model, or raise with what to load instead."""
    import comfy.model_base

    base = getattr(model, "model", None)
    if not isinstance(base, comfy.model_base.MiniMaxH3):
        raise ValueError(f"HyperFlow is a MiniMax H3 adapter; this model is {type(base).__name__}.")
    dm = base.diffusion_model
    if getattr(dm, "use_adaln_curves", False) or not hasattr(dm, "time_embedder"):
        raise ValueError(
            "This MiniMax H3 checkpoint is the curve-form variant (adaln_t_table, no time_embedder), e.g. the "
            "*_pruned_* files. HyperFlow patches the time embedder and adds a second one for the step endpoint, so it "
            "needs a checkpoint that still has time_embedder.* weights. For IC load "
            "minimax_h3_ref2va_int8_convrot.safetensors or minimax_h3_ref2va_bf16.safetensors (NON-PRUNED)."
        )
    import comfy.patcher_extension

    if model.wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {}).get(WRAPPER_KEY):
        raise ValueError("This model already carries HyperFlow; load it once, on the plain base model.")
    return dm


def build_endpoint_module(model, dm, endpoint_lora, alpha, rank, strength):
    """Original (unpatched) time embedder weights + the endpoint LoRA, merged into a standalone fp32 module."""
    module = copy.deepcopy(dm.time_embedder).to("cpu", torch.float32)
    prefix = "diffusion_model.time_embedder."
    original = model.get_key_patches(prefix)  # backup-aware: the checkpoint's weights, never a patched copy
    scale = strength * alpha / rank
    for name, param in module.named_parameters():
        weight = original[prefix + name][0][0]
        weight = weight.detach().to("cpu", torch.float32, copy=True)
        leaf = name.split(".")[0]
        if name.endswith(".weight") and leaf in endpoint_lora:
            a, b = endpoint_lora[leaf]
            delta = b.to(torch.float32) @ a.to(torch.float32)
            if delta.shape != weight.shape:
                raise ValueError(f"endpoint {leaf}: LoRA delta {tuple(delta.shape)} vs weight {tuple(weight.shape)}.")
            weight = weight + scale * delta
        param.data = weight
    # The model's operations may be ComfyUI cast ops; the copy runs on its own device at fp32 either way.
    for sub in module.modules():
        if hasattr(sub, "comfy_cast_weights"):
            sub.comfy_cast_weights = False
    return module


def apply_hyperflow(model, path: str, strength: float = 1.0):
    """Return a clone of ``model`` with HyperFlow applied. ``model`` is a ComfyUI ModelPatcher."""
    import comfy.lora
    import comfy.patcher_extension
    import comfy.utils

    dm = check_model(model)
    metadata = read_metadata(path)
    sd = comfy.utils.load_torch_file(path, safe_load=True)

    heads, head_dim = dm.blocks[0].attn.heads, dm.blocks[0].attn.head_dim
    converted = hf_keys.convert(sd, metadata.lora_alpha, qkv_rows=heads * head_dim)
    if metadata.lora_rank is not None and metadata.lora_rank != converted.rank:
        raise ValueError(f"Header says lora_rank={metadata.lora_rank} but the tensors have rank {converted.rank}.")
    _check_shapes(model, converted)

    m = model.clone()
    patches = comfy.lora.load_lora(converted.lora_sd, converted.key_map, log_missing=False)
    if len(patches) != len(converted.key_map):
        raise ValueError(f"ComfyUI built {len(patches)} LoRA patches for {len(converted.key_map)} HyperFlow targets.")
    added = m.add_patches(patches, strength)
    if len(added) != len(patches):
        missing = sorted(str(k) for k in set(patches) - set(added))
        raise ValueError(f"{len(missing)} HyperFlow patch target(s) are not in this model, e.g. {missing[:4]}.")

    endpoint = build_endpoint_module(m, dm, converted.endpoint, metadata.lora_alpha, converted.rank, strength)
    controller = HyperFlowController(endpoint, metadata.gate, metadata)
    base_forward = types.MethodType(type(dm.time_embedder).forward, dm.time_embedder)
    m.add_object_patch("diffusion_model.time_embedder.forward", controller.make_forward(base_forward))

    m.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, make_wrapper(controller, dm)
    )
    m.model_options[OPTIONS_KEY] = {
        "sigmas": list(metadata.sigmas or DEFAULT_SIGMAS_8STEP),
        "video_shift": metadata.video_shift or VIDEO_SHIFT,
        "audio_shift": metadata.audio_shift or AUDIO_SHIFT,
        "version": metadata.version,
        "gate": metadata.gate,
    }
    logger.info(
        "HyperFlow %s: %d weight patches + two-time embedder (gate %.4g, rank %d, strength %.3g).",
        metadata.version, len(added), metadata.gate, converted.rank, strength,
    )
    return m


def _check_shapes(model, converted) -> None:
    sd = model.model_state_dict("diffusion_model.")
    for module, target in converted.targets.items():
        if target.endpoint is not None:
            continue
        spec = converted.key_map[module]
        key, rows = (spec, None) if isinstance(spec, str) else (spec[0], spec[1][2])
        if key not in sd:
            raise KeyError(f"HyperFlow target {key} ({module}) does not exist on this model.")
        weight = sd[key]
        out_rows, in_cols = int(weight.shape[0]), int(weight.shape[1])
        a = converted.lora_sd[f"{module}.lora_A.weight"]
        b = converted.lora_sd[f"{module}.lora_B.weight"]
        expect_rows = rows if rows is not None else out_rows
        if target.qkv_index is not None and out_rows != 3 * rows:
            raise ValueError(f"{key}: {out_rows} rows is not 3 x {rows}; not a fused q|k|v projection.")
        if a.shape[1] != in_cols or b.shape[0] != expect_rows:
            raise ValueError(
                f"Shape mismatch for {module} -> {key}: weight {tuple(weight.shape)}, "
                f"lora_A {tuple(a.shape)}, lora_B {tuple(b.shape)}."
            )


def make_wrapper(controller: HyperFlowController, dm):
    import comfy.ldm.minimax.model as h3

    state = {"checked": None, "warned_cfg": False}

    def wrapper(executor, x, timestep, context, transformer_options, *args, **kwargs):
        sample_sigmas = transformer_options.get("sample_sigmas")
        if sample_sigmas is None:
            raise RuntimeError("HyperFlow needs the sampler's sigma schedule (transformer_options['sample_sigmas']).")
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", dm.sigma_shift_video))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", dm.sigma_shift_audio))

        schedule = [float(s) for s in sample_sigmas.flatten().tolist()]
        signature = (tuple(schedule), shift_v)
        if state["checked"] != signature:
            grid = shift_sigmas(controller.metadata.sigmas or DEFAULT_SIGMAS_8STEP, shift_v)
            if not is_subgrid(schedule, grid):
                raise ValueError(
                    "HyperFlow runs a fixed 8-step grid stored in its weights file, but this sampler was given "
                    f"{len(schedule) - 1} steps on another schedule. Feed the sampler's sigmas from the "
                    "'HyperFlow Sigmas' node (euler, cfg 1.0); a different schedule would silently condition every "
                    "step on the wrong interval."
                )
            state["checked"] = signature

        sigma = float((timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6))
        index = min(range(len(schedule)), key=lambda i: abs(schedule[i] - sigma))
        if abs(schedule[index] - sigma) > _SIGMA_MATCH_TOL or index + 1 >= len(schedule):
            raise ValueError(
                f"HyperFlow got a model call at sigma {sigma:.5f}, which is not a step of its grid. Use the 'euler' "
                "sampler: multi-stage samplers (heun, dpm_2, ...) evaluate between grid points."
            )
        sigma_next = schedule[index + 1]

        # IC keeps every audio row clean (mask 0), so at sigma 1 there is
        # no generated audio timestep colliding with video's t=0. Preserve
        # the official IC first step exactly instead of nudging its time.
        audio_mask = kwargs.get("audio_denoise_mask")
        audio_pinned = audio_mask is not None and not bool(torch.any(audio_mask))
        if sigma >= 1.0 - 1e-6 and not audio_pinned:
            timestep = timestep * (1.0 - STEP0_NUDGE)
        # exactly as MiniMaxH3Model._forward derives them, so the float32 values match its unique_t
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - h3.time_shift_sigma(sigma_v, shift_v, shift_a))
        r_v = 1.0 - sigma_next
        r_a = 1.0 - float(h3.time_shift_sigma(torch.tensor(sigma_next, dtype=torch.float32), shift_v, shift_a))

        payload = kwargs.get("minimax_payload") or {}
        pin = min(
            float(payload.get("visual_cond_noise_aug", h3.VISUAL_COND_TIMESTEP)),
            float(payload.get("audio_cond_noise_aug", h3.AUDIO_COND_TIMESTEP)),
            h3.VISUAL_COND_TIMESTEP,
        )
        if 1 in (transformer_options.get("cond_or_uncond") or []) and not state["warned_cfg"]:
            state["warned_cfg"] = True
            logger.warning("HyperFlow was distilled for cfg 1.0; this run evaluates an unconditional branch.")

        previous = controller.context
        controller.context = StepContext(t_v, t_a, r_v, r_a, pin, float(sigma_v), sigma_next)
        try:
            return executor(x, timestep, context, transformer_options, *args, **kwargs)
        finally:
            controller.context = previous

    return wrapper
