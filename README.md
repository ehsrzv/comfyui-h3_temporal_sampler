# H3 Temporal Sampler

![H3 Temporal Sampler](https://github.com/ehsrzv/comfyui-h3_temporal_sampler/blob/main/assets/banner.jpg?raw=true)

ComfyUI custom nodes for seamless high-resolution MiniMax H3 video generation. Split
an H3 audio-video latent that exceeds your VRAM into overlapping segments,
denoise them consistently, and cache latents between runs.

> This node is optimized for increasing output size (up to 2K and beyond), but
> it can also be used for longer videos — working around VRAM limits at the cost
> of longer render times. In practice, the author has used it to upscale up to 4K
> — more would be possible, but MiniMax gains little from upscaling past that
> point.

## Nodes

All nodes live under the **h3 temporal sampler** category.

### H3 Temporal Sampler

Segmented `SamplerCustomAdvanced` for H3 AV latents. Splits the latent into
overlapping time segments (grid-snapped to multiples of 5 latent frames) so
each fits in VRAM, denoises each segment, and joins them seamlessly.

**Inputs**

| Widget | Type | Default | Notes |
|---|---|---|---|
| enable | BOOLEAN | True | OFF = one plain pass over the full latent, like SamplerCustomAdvanced. |
| noise | NOISE | — | Full-length noise field; sliced per segment so overlaps share identical initial noise. |
| guider | GUIDER | — | Conditioning guider, applied identically to every segment. |
| sampler | SAMPLER | — | Sampler algorithm used for every segment. |
| sigmas | SIGMAS | — | Sigma schedule; identical for every segment so the segments match. |
| latent_image | LATENT | — | Full-length H3 AV latent (nested video + audio). |
| num_segments | INT | 4 | 1–10. How many overlapping segments to split the latent into — more segments fit larger sizes in VRAM. |
| smart_bounds | BOOLEAN | False | ON = place boundaries at motion valleys (low-motion points) instead of even spacing. Falls back to even spacing if infeasible. |
| overlap_frames | INT | 5 | 5–20, snapped to multiples of 5. Overlap between neighbours. Larger = smoother joins, more compute. |
| blend_mode | COMBO | multiband | Join style for the final assembly: `linear` / `adaptive` / `multiband` (fine detail blended narrowly — best for fine textures). |
| seam_lock | BOOLEAN | True | ON = freeze the leading frames of each segment (after the first) from the merged timeline via the denoise mask, so joins stay invisible. |
| seam_lock_frames | INT | 2 | 1–4. How many leading frames to freeze; the freeze tapers off gradually (masks 0.0, 0.25, 0.5, 0.75). |

**Outputs:** `output` (LATENT), `denoised_output` (LATENT)

**How it works**

![How it works](https://github.com/ehsrzv/comfyui-h3_temporal_sampler/blob/main/assets/schematic.jpg?raw=true)

Each segment is sampled independently (one model init, shared across
segments via equal-shape padding), then joined with the selected
`blend_mode`. With Seam Lock on, the leading frames of each segment are
frozen from the merged timeline via the denoise mask before sampling, so
joins stay invisible. A per-boundary seam-quality report is printed (lower
= cleaner; `CHECK` flags scores ≥ 1.0).

**Notes**

- Each segment is capped by a VRAM budget (validated reference:
  34 latent frames at 84×144); if segments get too large, raise
  `num_segments`.
- Overlap is reserved up front, so padded segments never exceed the cap.

**Tuning segments for your hardware**

`num_segments` is your main lever: more segments mean shorter segments, and
shorter segments use less VRAM — which is also how you push output size higher
than a single pass could handle. If a render runs out of memory, raise
`num_segments`; if every segment is comfortably small, lower it for a faster
render.

The trade-off is time: each extra segment adds its own denoising pass (plus
overlap work), so total render time grows with the segment count. As a rule of
thumb, use the fewest segments that still fit — that is the shortest render
your GPU can handle.

There is no universal number: the right count depends on your GPU's VRAM, the
target size and the clip duration, so tune it yourself until you get a feel
for what your card handles. The current defaults are tuned for the author's
own system.

### Demo

https://github.com/user-attachments/assets/955a1ff6-3759-4d37-a455-b27ec17e182e

Sample render — Stage 1 was short enough for a single pass, no Temporal Sampler needed. Stage 2 crashed on VRAM, so it was split into 4 segments (overlap 5, Seam Lock Frames 2) to get through. Segment cuts at approx. 2.1s, 4.9s and 7.7s of the 11.5s clip. Fine textures stay perfectly stable across all 4 segments, with no visible seams. Seam-quality scores: 0.67 / 0.36 / 0.15 (lower = cleaner).

Texture check across the segment boundaries: facial skin tone and lighting stay continuous, the tie's dot pattern keeps its size, spacing and alignment, and the jacket's houndstooth weave shows no breaks — only natural singing motion.

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
git clone https://github.com/ehsrzv/comfyui-h3_temporal_sampler.git
```

Then restart ComfyUI.

### Option 2: Download zip

1. Download the zip from the
   [releases page](https://github.com/ehsrzv/comfyui-h3_temporal_sampler/releases)
   (or the latest zip).
2. Extract it into `ComfyUI/custom_nodes/` so you get:
   `ComfyUI/custom_nodes/comfyui-h3_temporal_sampler/`
3. Restart ComfyUI.

### Option 3: ComfyUI Manager

Search for `h3-temporal-sampler` in ComfyUI Manager and install.

---

After installation, find the nodes under the **h3 temporal sampler** category.
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

Apache 2.0
