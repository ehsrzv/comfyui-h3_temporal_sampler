# H3 Temporal Sampler

![H3 Temporal Sampler](https://github.com/ehsrzv/comfyui-h3_temporal_tile/blob/main/assets/banner.jpg?raw=true)

ComfyUI custom nodes for long-form MiniMax H3 video generation. Tile a long
H3 audio-video latent into overlapping segments, denoise them consistently,
and cache latents between runs.

> This node is designed both for extending video duration and for increasing
> size up to 2K and beyond, but it increases render cost and is primarily
> designed for working around VRAM limits. In practice, the author has used
> it to upscale up to 4K — more would be possible, but MiniMax gains little
> from upscaling past that point.

## Nodes

All nodes live under the **h3 temporal tile** category.

### H3 Temporal Sampler

Tiled `SamplerCustomAdvanced` for H3 AV latents. Splits a long latent into
overlapping time segments (grid-snapped to multiples of 5 latent frames),
denoises each segment, and joins them seamlessly.

**Inputs**

| Widget | Type | Default | Notes |
|---|---|---|---|
| enable | BOOLEAN | True | OFF = one plain pass over the full latent, like SamplerCustomAdvanced. |
| noise | NOISE | — | Full-length noise field; sliced per segment so overlaps share identical initial noise. |
| guider | GUIDER | — | Conditioning guider, applied identically to every segment. |
| sampler | SAMPLER | — | Sampler algorithm used for every segment. |
| sigmas | SIGMAS | — | Sigma schedule; identical for every segment so the tiles match. |
| latent_image | LATENT | — | Full-length H3 AV latent (nested video + audio). |
| num_segments | INT | 4 | 1–10. How many overlapping time segments to split into. |
| smart_bounds | BOOLEAN | True | ON = place boundaries at motion valleys (low-motion points) instead of even spacing. Falls back to even spacing if infeasible. |
| overlap_frames | INT | 10 | 5–20, snapped to multiples of 5. Overlap between neighbours. Larger = smoother joins, more compute. |
| blend_mode | COMBO | adaptive | Join style for the final assembly: `linear` / `smoothstep` / `adaptive`. |

**Outputs:** `output` (LATENT), `denoised_output` (LATENT)

**How it works**

![How it works](https://github.com/ehsrzv/comfyui-h3_temporal_tile/blob/main/assets/schematic.jpg?raw=true)

Each segment is sampled independently (one model init, shared across
segments via equal-shape padding), then joined with the selected
`blend_mode`. A per-boundary seam-quality report is printed (lower =
cleaner; `CHECK` flags scores ≥ 1.0).

**Notes**

- Segment length is capped at a frame-pixel VRAM budget (validated
  reference: 34 frames at 84×144); with `smart_bounds` off, raise
  `num_segments` if segments get too large for VRAM.
- Overlap is reserved up front, so padded segments never exceed the cap.

**Tuning segments for your hardware**

`num_segments` is your main lever against VRAM exhaustion: more segments mean
shorter segments, and shorter segments use less VRAM. If a render runs out of
memory, raise `num_segments`; if every segment is comfortably small, lower it
for a faster render.

The trade-off is time: each extra segment adds its own denoising pass (plus
overlap work), so total render time grows as the segment count grows. As a rule
of thumb, use the fewest segments that still keep every segment under the VRAM
cap — that is the shortest render your GPU can handle.

### H3 Latent Cache (Save/Load)

Save and load H3 nested latents. ComfyUI's core Save/Load Latent nodes
crash on H3's `NestedTensor` AV latent; this node round-trips the latent
dict with `torch.save` / `torch.load`, preserving the structure exactly.

**Inputs**

| Widget | Type | Default | Notes |
|---|---|---|---|
| latent | LATENT | — | Live latent. Saved to a new file in SAVE mode; ignored in LOAD mode. |
| load_mode | BOOLEAN | False | OFF = save the input to a new auto-numbered file and pass it through. ON = ignore the input, load the file chosen below. |
| filename_prefix | STRING | h3_stage2_latent | Prefix for newly saved files (auto-numbered `_00001`, `_00002`, …). |
| latent_file | COMBO | — | Saved file to load when `load_mode` is ON. |

**Output:** `latent` (LATENT)

Files are written as `.h3latent.pt` into ComfyUI's output
directory. After saving new files, re-add (or duplicate) the node to
refresh the dropdown list.

## Installation

### Option 1: Git clone (recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ehsrzv/comfyui-h3_temporal_tile.git
```

Then restart ComfyUI.

### Option 2: Download zip

1. Download the zip from the
   [releases page](https://github.com/ehsrzv/comfyui-h3_temporal_tile/releases)
   (or the latest zip).
2. Extract it into `ComfyUI/custom_nodes/` so you get:
   `ComfyUI/custom_nodes/comfyui-h3_temporal_tile/`
3. Restart ComfyUI.

### Option 3: ComfyUI Manager

Search for `h3-temporal-tile` in ComfyUI Manager and install.

---

After installation, find the nodes under the **h3 temporal tile** category.
No extra dependencies — only ComfyUI itself and a MiniMax H3 setup.

## Requirements

- ComfyUI (recent)
- MiniMax H3 model + a compatible guider/sampler setup
- The latent passed to H3 Temporal Sampler must be an H3 AV latent
  (nested video + audio samples)

## Acknowledgments

Example workflow based on the workflow templates from
[LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
(Apache-2.0).

## License

MIT
