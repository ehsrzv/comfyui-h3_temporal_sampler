"""H3 nested-latent save/load.

ComfyUI core's SaveLatent/LoadLatent assume a plain Tensor and crash on
MiniMax H3's audio-video latent, which is a torch NestedTensor
(AttributeError: 'NestedTensor' object has no attribute 'contiguous').

These two nodes round-trip the latent dict with torch.save / torch.load,
preserving the NestedTensor structure exactly, so a stage-2-only test
reproduces the same latent every time.

Install: drop this file into ComfyUI/custom_nodes/ and restart ComfyUI.

Nodes:
- H3SaveNestedLatent / H3LoadNestedLatent: the original separate save/load pair.
- H3LatentCache: one node for both directions. Takes the live latent input;
  with load_mode OFF it saves the input to a NEW auto-incremented file
  (like the old saver) and passes the latent through; with load_mode ON it
  ignores the input and outputs the latent loaded from the file chosen in
  the latent_file dropdown.
"""

import os

import torch

import folder_paths

EXT = ".h3latent.pt"


def _latent_dir():
    return folder_paths.get_output_directory()


class H3LatentCache:
    """Save/load on one node, switched by a boolean.

    load_mode OFF (default): saves the incoming latent to a NEW file
        <output_dir>/<prefix>.h3latent.pt (auto-increments like the old
        saver: _00001, _00002, ...) and passes the latent through unchanged,
        so the workflow continues normally.
    load_mode ON: ignores the incoming latent and outputs the latent
        loaded from the file chosen in the latent_file dropdown.
    NOTE: the dropdown is built when the node is placed; after saving new
    files, re-add (or duplicate) the node to refresh the list.
    """

    @classmethod
    def INPUT_TYPES(cls):
        try:
            files = sorted(f for f in os.listdir(_latent_dir()) if f.endswith(EXT))
        except OSError:
            files = []
        if not files:
            files = ["<no .h3latent.pt saved yet>"]
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Live latent. Saved to a new file in SAVE mode; ignored in LOAD mode."}),
                "load_mode": ("BOOLEAN", {"default": False,
                                          "label_on": "LOAD from file",
                                          "label_off": "SAVE to new file",
                                          "tooltip": "OFF: save the input to a new auto-numbered file and pass it through. ON: ignore the input and load the file chosen below."}),
                "filename_prefix": ("STRING", {"default": "h3_stage2_latent",
                                               "tooltip": "Prefix for newly saved files (auto-numbered: _00001, _00002, ...)."}),
                "latent_file": (files, {"tooltip": "Saved file to load when load_mode is ON."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "h3 temporal tile"

    def run(self, latent, load_mode, filename_prefix, latent_file):
        out_dir = _latent_dir()

        if load_mode:
            if latent_file.startswith("<no "):
                raise ValueError(
                    "[H3LatentCache] no .h3latent.pt files saved yet — "
                    "run once with load_mode OFF to save one first.")
            path = os.path.join(out_dir, os.path.basename(latent_file))
            if not os.path.exists(path):
                raise ValueError(f"[H3LatentCache] file not found: {path}")
            data = torch.load(path, map_location="cpu", weights_only=False)
            print(f"[H3LatentCache] LOADED {path} (input latent ignored)")
            out = dict(data)
            mode = "loaded"
            shown = os.path.basename(path)
        else:
            prefix = ((filename_prefix or "h3_stage2_latent").strip()
                      or "h3_stage2_latent")
            prefix = os.path.basename(prefix)
            path = os.path.join(out_dir, prefix + EXT)
            counter = 1
            while os.path.exists(path):
                path = os.path.join(out_dir, f"{prefix}_{counter:05d}{EXT}")
                counter += 1
            out = dict(latent)
            torch.save(out, path)
            print(f"[H3LatentCache] SAVED {path} (passed through)")
            mode = "saved"
            shown = os.path.basename(path)

        return {"ui": {"h3_latent_cache": [f"{mode}: {shown}"]},
                "result": (out,)}

    @classmethod
    def IS_CHANGED(cls, latent, load_mode, filename_prefix, latent_file):
        # Never cache: save must write fresh on every run, and load must see
        # files written outside this run.
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "H3LatentCache": H3LatentCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LatentCache": "H3 Latent Cache (Save/Load)",
}
