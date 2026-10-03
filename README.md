# Sampler Optimizer

ComfyUI custom nodes for long-form MiniMax H3 video generation. Tile a long
H3 audio-video latent into overlapping segments, denoise them consistently,
and cache latents between runs.

> This node is designed both for extending video duration and for increasing
> size up to 2K and beyond, but it increases render cost and is primarily
> designed for working around VRAM limits.

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
| step_average | BOOLEAN | True | ON = MultiDiffusion-style per-step consensus (see below). OFF = sample each segment independently, then join with `blend_mode`. |
| noise | NOISE | — | Full-length noise field; sliced per segment so overlaps share identical initial noise. |
| guider | GUIDER | — | Conditioning guider, applied identically to every segment. |
| sampler | SAMPLER | — | Sampler algorithm used for every segment. |
| sigmas | SIGMAS | — | Sigma schedule; identical for every segment so the tiles match. |
| latent_image | LATENT | — | Full-length H3 AV latent (nested video + audio). |
| num_segments | INT | 2 | 1–10. How many overlapping time segments to split into. |
| smart_bounds | BOOLEAN | False | ON = place boundaries at motion valleys (low-motion points) instead of even spacing. Falls back to even spacing if infeasible. |
| overlap_frames | INT | 10 | 5–20, snapped to multiples of 5. Overlap between neighbours. Larger = smoother joins, more compute. |
| blend_mode | COMBO | linear | Join style for the final assembly in non-step-average mode: `linear` / `smoothstep` / `adaptive`. Disabled while `step_average` is ON. |

**Outputs:** `output` (LATENT), `denoised_output` (LATENT)

**How it works**

- **step_average ON** (recommended): all segments advance one denoising step
  at a time. After every step, the true overlap regions of neighbouring
  segments are averaged with cosine-ramped weights and written back to both
  sides, so the tiles converge to one consistent video instead of being
  blended afterwards. Final assembly is a gentle linear join. The model is
  held loaded across all steps (single init).
- **step_average OFF**: each segment is sampled independently (one model
  init, shared across segments via equal-shape padding), then joined with
  the selected `blend_mode`. A per-boundary seam-quality report is printed
  (lower = cleaner; `CHECK` flags scores ≥ 1.0).

**Notes**

- Segment length is capped at a frame-pixel VRAM budget (validated
  reference: 34 frames at 84×144); with `smart_bounds` off, raise
  `num_segments` if segments get too large for VRAM.
- Overlap is reserved up front, so padded segments never exceed the cap.
- H3 is a FLOW model: per-step noise inversion uses the CONST sampling
  formula.

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

Files are written as `<prefix>.h3latent.pt` into ComfyUI's output
directory. After saving new files, re-add (or duplicate) the node to
refresh the dropdown list.

## Installation

1. Download the zip and extract it into `ComfyUI/custom_nodes/` so you get:
   `ComfyUI/custom_nodes/comfyui-h3_temporal_tile/`
2. Restart ComfyUI.
3. Find the nodes under the **h3 temporal tile** category.

## Requirements

- ComfyUI (recent)
- MiniMax H3 model + a compatible guider/sampler setup
- The latent passed to H3 Temporal Sampler must be an H3 AV latent
  (nested video + audio samples)

## License

MIT
