"""H3 IC Loader — the official Open-H3-Regenerate-2K-IC LoRA recipe, stripped down.

Applies the three original official IC LoRAs (IC + HyperFlow 8-step + LMS Sharp)
to a native non-pruned Ref2VA MODEL, including HyperFlow two-time conditioning
and the 50 Sharp curve branches. Returns MODEL only.

Deliberately NOT included (unlike kkget's OpenH3ICLoader):
- no memory chunking (no token_chunk / head_chunk / key_chunk, no attention
  choice): your own attention and chunk nodes govern the model.
- no SIGMAS output and no sampler choice: feed sigmas from your own source
  (e.g. MiniMaxH3SigmaShift) into H3 Temporal Sampler.
- no sampler, encode or decode node: use H3 Temporal Sampler for sampling.

Recipe code is vendored from kkget/ComfyUI-Open-H3-Regenerate-IC (Apache-2.0),
see ic_native/adapters.py and ic_vendor/hyperflow/. The Sharp curve asset
(assets/lms_curve.bin) is not shipped in git; fetch it with
scripts/download_assets.py.
"""
CATEGORY = "h3 temporal sampler"
IC = "open_h3_regenerate_2k_ic_lora.safetensors"
HF = "minimax_h3_hyperflow_8step_v1.0.safetensors"
SHARP = "minimax_h3_lms_v1.0_r64.safetensors"


def files(category, preferred):
    import folder_paths
    names = folder_paths.get_filename_list(category)
    selected = next((n for n in names if n.replace("\\", "/").split("/")[-1] == preferred), preferred)
    return (names if selected in names else [selected] + names, {"default": selected})


class H3ICLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "ic_lora": files("loras", IC),
            "hyperflow_lora": files("loras", HF),
            "sharp_lora": files("loras", SHARP),
            "sharp_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": .05}),
            "lora_dtype": (["bf16", "fp32"], {"default": "bf16", "tooltip": "IC/Sharp factor storage in RAM; bf16 halves the FP32 IC file's RAM footprint."}),
        }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Apply the three original official IC LoRAs to a native non-pruned Ref2VA MODEL: "
                   "HyperFlow two-time conditioning and the 50 Sharp curve branches. "
                   "No memory chunking, no attention override, no SIGMAS output — "
                   "your own attention/chunk nodes govern the model and sigmas come from your own source.")

    def load(self, model, ic_lora, hyperflow_lora, sharp_lora, sharp_strength, lora_dtype):
        import folder_paths
        import torch
        from .ic_native.adapters import apply_recipe
        paths = [folder_paths.get_full_path_or_raise("loras", name) for name in (ic_lora, hyperflow_lora, sharp_lora)]
        # apply_recipe also computes the official sigma grid, but it is discarded:
        # this loader returns MODEL only; sigmas are supplied by your own source.
        # "up3" is only the recipe label stored in model_options["open_h3_ic"].
        model, _sigmas = apply_recipe(model, *paths, sharp_strength=sharp_strength,
            dtype=torch.bfloat16 if lora_dtype == "bf16" else torch.float32, sampler="up3")
        return (model,)


NODE_CLASS_MAPPINGS = {"H3ICLoader": H3ICLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ICLoader": "H3 IC Loader"}
