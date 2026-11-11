"""Official IC + HyperFlow + LMS Sharp, applied through ModelPatcher clones.

Vendored from kkget/ComfyUI-Open-H3-Regenerate-IC (Apache-2.0); changes vs
upstream: import path (..vendor -> ..ic_vendor), missing-curve hint
(scripts/download_assets.py), and the sigma grid removed — apply_recipe
returns the model only, no sigmas, no sampler choice."""
from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

import torch
from safetensors import safe_open

from ..ic_vendor.hyperflow.keys import convert
from ..ic_vendor.hyperflow.patch import apply_hyperflow, check_model, _check_shapes

CURVE_SHA256 = "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"
ASSETS = Path(__file__).resolve().parents[1] / "assets"


def load_factors(path, dtype):
    # Cast one tensor at a time: the official IC file is FP32 (~8.3 GB).
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k).to(dtype) for k in f.keys()}


def add_checked(model, state, key_map, strength):
    import comfy.lora
    patches = comfy.lora.load_lora(state, key_map, log_missing=False)
    if len(patches) != len(key_map):
        raise ValueError(f"Only {len(patches)}/{len(key_map)} LoRA targets were recognized")
    added = model.add_patches(patches, strength)
    if len(added) != len(patches):
        raise ValueError(f"LoRA has missing model targets: {set(patches) - set(added)}")
    return len(added)


def apply_ic(model, path, strength=1.0, dtype=torch.bfloat16):
    dm = model.get_model_object("diffusion_model")
    sd = load_factors(path, dtype)
    rank = next(v.shape[0] for k, v in sd.items() if k.endswith(".lora_A.weight"))
    converted = convert(sd, rank, qkv_rows=dm.blocks[0].attn.heads * dm.blocks[0].attn.head_dim,
                        require_endpoint=False)
    expected = 6 * (len(dm.blocks) + len(dm.token_refiner.blocks))
    if len(converted.targets) != expected or converted.endpoint or any(
            t.native.startswith("time_embedder") for t in converted.targets.values()):
        raise ValueError(f"Select the original open_h3_regenerate_2k_ic_lora.safetensors: expected {expected} IC modules")
    _check_shapes(model, converted)
    out = model.clone()
    add_checked(out, converted.lora_sd, converted.key_map, strength)
    return out


def load_curve(path=None):
    path = Path(path or ASSETS / "lms_curve.bin")
    if not path.is_file():
        raise FileNotFoundError("Missing Sharp curve. Run scripts/download_assets.py")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CURVE_SHA256:
        raise ValueError("The LMS Sharp reference curve has the wrong SHA256")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(1025, 8).clone()


class CurveContext:
    def __init__(self, table):
        self.table = table.float().cpu()
        self.local = threading.local()

    def record(self, forward):
        def run(t):
            self.local.t = t.detach().float()
            return forward(t)
        return run

    def coords(self, x):
        t = getattr(self.local, "t", None)
        if t is None or len(t) != len(x):
            raise RuntimeError("Sharp curve called without the matching H3 model timestep")
        table = self.table.to(x.device)
        pos = t.to(x.device).clamp(0, 1) * (len(table) - 1)
        lo = pos.floor().long().clamp(max=len(table) - 2)
        return torch.lerp(table[lo], table[lo + 1], (pos - lo).unsqueeze(-1))


def curve_forward(linear, weight, context, strength):
    # Patch only forward; parameter paths and ComfyUI's cast/offload hooks survive.
    base_forward = type(linear).forward
    cache = {}

    def run(x):
        out = base_forward(linear, x)
        w = cache.get(x.device)
        if w is None:
            cache.clear()
            cache[x.device] = w = weight.to(x.device)
        return out + (context.coords(x) @ w.t() * strength).to(out.dtype)
    return run


def apply_sharp(model, path, strength=1.0, dtype=torch.bfloat16, table=None):
    if strength == 0:
        return model
    dm = model.get_model_object("diffusion_model")
    context = CurveContext(load_curve() if table is None else table)
    sd = load_factors(path, dtype)
    roots = sorted(k.removesuffix(".lora_A.weight") for k in sd if k.endswith(".lora_A.weight"))
    if set(sd) != {r + s for r in roots for s in (".lora_A.weight", ".lora_B.weight")}:
        raise ValueError("Unexpected tensor keys in the official LMS Sharp LoRA")
    out, linears, mapping, curves = model.clone(), {}, {}, set()
    weights = model.model_state_dict("diffusion_model.")
    for root in roots:
        a, b = sd[root + ".lora_A.weight"], sd[root + ".lora_B.weight"]
        key = root + ".weight"
        if key not in weights or a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"Invalid Sharp target {root}")
        match = re.fullmatch(r"diffusion_model.blocks.(\d+).adaln_proj.linear", root)
        if match and a.shape[1] == 8:
            if b.shape[0] != weights[key].shape[0]:
                raise ValueError(f"Sharp curve shape mismatch: {root}")
            curves.add(int(match[1]))
            linear = out.get_model_object(root)
            out.add_object_patch(root + ".forward", curve_forward(linear, b.float() @ a.float(), context, strength))
        else:
            if (b.shape[0], a.shape[1]) != tuple(weights[key].shape):
                raise ValueError(f"Sharp LoRA shape mismatch: {root}")
            linears[root + ".lora_A.weight"], linears[root + ".lora_B.weight"] = a, b
            mapping[root] = key
    if curves != set(range(len(dm.blocks))):
        raise ValueError(f"Expected {len(dm.blocks)} Sharp curve branches, got {len(curves)}")
    out.model_options["h3_ic_sharp_curves"] = len(curves)
    add_checked(out, linears, mapping, strength)
    forward = out.get_model_object("diffusion_model.time_embedder.forward")
    out.add_object_patch("diffusion_model.time_embedder.forward", context.record(forward))
    return out


def apply_recipe(model, ic, hyperflow, sharp, sharp_strength=1.0, dtype=torch.bfloat16):
    check_model(model)
    out = apply_hyperflow(model, str(hyperflow))
    out = apply_sharp(out, sharp, sharp_strength, dtype)
    out = apply_ic(out, ic, 1.0, dtype)
    meta = out.model_options["hyperflow"]
    if meta["video_shift"] != 12.0 or meta["audio_shift"] != 3.0:
        raise ValueError("The official IC recipe requires HyperFlow shifts 12 (video) and 3 (audio)")
    return out
