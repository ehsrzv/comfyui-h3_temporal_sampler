import torch


GRID = 5  # latent frames per keyframe-token block (~= 17 pixel frames)


def _snap_down(v):
    return (int(v) // GRID) * GRID


def _unbind_av(samples):
    # comfy's NestedTensor.unbind() takes no dim argument.
    try:
        video, audio = samples.unbind()
    except TypeError:
        video, audio = samples.unbind(0)
    return video, audio


def _pack_av(v, a, template=None):
    # Rebuild with the SAME class as the input (comfy.nested_tensor.NestedTensor),
    # whose constructor takes a plain list. torch.nested requires equal ndim and
    # cannot pack the 5D video + 4D audio pair.
    cls = type(template) if template is not None else None
    if cls is None:
        from comfy.nested_tensor import NestedTensor
        cls = NestedTensor
    return cls([v, a])


def _slice_mask(mask, v0, v1, a0, a1, Tv, si=0):
    """Slice a denoise mask down to the current segment.

    Plain torch masks slice directly. H3 latents carry the noise_mask as a
    comfy NestedTensor (no .dim()), so those are unbound to their
    [video, audio] pair first. Returns None when the mask can't be sliced —
    sampling then proceeds unmasked instead of crashing.
    """
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        try:
            if mask.dim() >= 3 and mask.shape[2] == Tv:
                return mask[:, :, v0:v1].contiguous()
        except Exception:
            pass
        return None
    try:
        m_video, m_audio = _unbind_av(mask)
        return type(mask)([m_video[:, :, v0:v1].contiguous(),
                           m_audio[..., a0:a1].contiguous()])
    except Exception:
        if si == 0:
            print("[H3TemporalSampler] noise_mask has unsupported structure "
                  f"({type(mask).__name__}) — sampling without denoise mask")
        return None


def _inner_bounds(Tv, num_segments):
    """Inner segment boundary positions (GRID-snapped), plus segment count."""
    N = max(1, min(int(num_segments), max(1, Tv // GRID)))
    bounds = [0]
    for i in range(1, N):
        b = int(round(i * Tv / N / GRID)) * GRID
        b = max(b, bounds[-1] + GRID)
        b = min(b, Tv - (N - i) * GRID)
        bounds.append(b)
    bounds.append(Tv)
    return bounds, N


def _plan_segments(Tv, Ta, num_segments, overlap, bounds=None):
    """Plan [(v0, v1, a0, a1), ...] covering [0, Tv).

    overlap: int (same for every boundary) or a list of per-boundary
    overlaps. bounds: explicit [0, ..., Tv] boundary list (e.g. from
    _smart_bounds); when None, boundaries are evenly spaced (GRID-snapped).
    """
    if bounds is None:
        bounds, N = _inner_bounds(Tv, num_segments)
    else:
        N = len(bounds) - 1
    if isinstance(overlap, (list, tuple)):
        ovs = [max(GRID, (int(o) // GRID) * GRID) for o in overlap]
        ovs = ((ovs + [ovs[-1]] * max(0, N - 1))[:max(0, N - 1)] if ovs
               else [GRID] * max(0, N - 1))
    else:
        ovs = [max(GRID, (int(overlap) // GRID) * GRID)] * max(0, N - 1)
    ratio = Ta / Tv if Tv else 0
    segs = []
    for i in range(N):
        v0 = 0 if i == 0 else max(0, bounds[i] - ovs[i - 1])
        v1 = Tv if i == N - 1 else bounds[i + 1]
        a0 = int(round(v0 * ratio))
        a1 = int(round(v1 * ratio))
        segs.append((v0, v1, a0, a1))
    return segs


def _blend_weight(ov, mode, device, dtype, a_tail=None, b_head=None, sharpness=6.0):
    """Per-frame blend weights s in [0,1] across an overlap of ov frames.

    linear     : straight ramp (validated default)
    smoothstep : 3t^2 - 2t^3, softer ends, faster middle -> less ghosting
    cosine     : 0.5 - 0.5*cos(pi*t), close cousin of smoothstep
    gaussian   : each side weighted by a Gaussian centered on itself
                 (Mixture-of-Diffusers style), renormalized to exact 0->1
    sigmoid    : logistic S-curve, steepness set by `sharpness`
                 (high sharpness -> close to a hard cut)
    adaptive   : spends the 0->1 transition budget where the sides AGREE and
                 rushes through high-disagreement frames (no lingering in the
                 ghost zone); degrades to linear when the overlap is uniform
    center_hann: SeedVR2-style — pure A in the first third, Hann crossfade
                 only in the middle third, pure B in the last third.
                 Minimal time in the ghost zone, zero-derivative joins
                 (no pop like a hard cut). Linear fallback for ov < 3.
    """
    t = torch.linspace(0, 1, ov, device=device, dtype=dtype)
    if ov < 2:
        return t
    if mode == "smoothstep":
        s = t * t * (3 - 2 * t)
    elif mode == "cosine":
        s = 0.5 - 0.5 * torch.cos(t * torch.pi)
    elif mode == "gaussian":
        sig = 0.45
        w_a = torch.exp(-(t / sig) ** 2)
        w_b = torch.exp(-((t - 1) / sig) ** 2)
        s = w_b / (w_a + w_b)
        s = (s - s[0]) / (s[-1] - s[0])
    elif mode == "sigmoid":
        k = max(float(sharpness), 0.5)
        s = torch.sigmoid(k * (t - 0.5))
        s = (s - s[0]) / (s[-1] - s[0])
    elif mode == "adaptive" and a_tail is not None and b_head is not None:
        # Per-frame disagreement -> per-frame step size: big steps where the
        # sides disagree (rush through), small steps where they agree.
        with torch.no_grad():
            d = (a_tail.float() - b_head.float()).abs().mean(dim=(0, 1, 3, 4))
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            w = d + 0.15
            raw = torch.cumsum(w, dim=0)
            denom = (raw[-1] - raw[0]).item()
        if denom < 1e-8:
            s = t
        else:
            s = ((raw - raw[0]) / denom).to(dtype)
    elif mode == "center_hann":
        if ov >= 3:
            u = torch.clamp((t - 1.0 / 3.0) / (1.0 / 3.0), 0.0, 1.0)
            s = 0.5 - 0.5 * torch.cos(u * torch.pi)
        else:
            s = t
    else:  # "linear" and any fallback
        s = t
    return s


def _blend_multiband(a_tail, b_head, narrow_frac=0.5):
    """Two-band temporal blend of one overlap region.

    Splits each side into low (temporal blur) and high (residual) bands:
    lows blend across the whole overlap (smooth large-scale transition),
    highs blend only in the central `narrow_frac` of the overlap, so fine
    detail never sits long in the ghost zone. Pure A / pure B at the edges.
    a_tail, b_head: [B, C, ov, H, W].
    """
    ov = a_tail.shape[2]
    device, dtype = a_tail.device, a_tail.dtype
    if ov >= 3:
        def _low(x):
            xp = torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
            return torch.nn.functional.avg_pool3d(xp, kernel_size=(3, 1, 1), stride=1)
    else:
        def _low(x):
            return x
    low_a, low_b = _low(a_tail), _low(b_head)
    high_a, high_b = a_tail - low_a, b_head - low_b
    t = torch.linspace(0, 1, ov, device=device, dtype=dtype)
    w_low = t * t * (3 - 2 * t)
    tc = torch.clamp((t - (1 - narrow_frac) / 2) / narrow_frac, 0, 1)
    w_high = tc * tc * (3 - 2 * tc)
    w_low = w_low.view(1, 1, ov, 1, 1)
    w_high = w_high.view(1, 1, ov, 1, 1)
    return ((1 - w_low) * low_a + w_low * low_b
            + (1 - w_high) * high_a + w_high * high_b)


def _motion_to_overlap(score):
    """Map a normalized motion score to a GRID-snapped overlap (v1 heuristics).

    Auto mode never exceeds 10 (2*GRID): bigger overlaps cost more compute
    than they buy (lowered from 15 on 2026-10-03: heavy).
    """
    if score < 0.10:
        return GRID
    return 2 * GRID


def _motion_profile(video):
    """Normalized per-frame motion of the input latent: (Tv-1,) tensor."""
    with torch.no_grad():
        v = video.float()
        mot = (v[:, :, 1:] - v[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))
        scale = v.std() + 1e-8
        return mot / scale


def _auto_num_segments(Tv, video):
    """Pick segment count so each segment is ~34 latent frames at 84x144.

    Memory scales with frames x pixels; the reference is his validated
    stage-2 run (172 frames at 84x144 latent in 5 segments). v1 heuristic,
    printed every run.
    """
    H, W = video.shape[3], video.shape[4]
    budget = 34 * 84 * 144  # frame-pixels per segment
    target = max(10, int(budget / max(1, H * W)))
    N = max(1, min(10, (Tv + target - 1) // target))
    return min(N, max(1, Tv // GRID))


def _smart_bounds(Tv, motion, N, window=10, max_len=None):
    """Place N-1 inner boundaries at motion valleys (GRID-snapped).

    motion: normalized (Tv-1,) profile. The cost of a candidate is the MEAN
    motion in a small window around it (a narrow calm moment counts; nearby
    spikes push the boundary away). Boundaries keep a minimum gap so no
    segment gets degenerate, and no segment may exceed max_len (the
    frame-pixel VRAM budget — without this the DP happily clusters all
    boundaries in one calm region and leaves a giant VRAM-busting segment).
    Returns [0, ..., Tv], or None if infeasible (caller falls back to even
    spacing).
    """
    if N <= 1:
        return [0, Tv]
    if max_len is None:
        max_len = Tv
    max_len = max(GRID, int(max_len))
    min_gap = max(GRID, GRID * ((Tv // N) // 2 // GRID))
    if (N + 1) * min_gap > Tv:
        return None
    if N * max_len < Tv:
        # Even the tightest packing can't cover the video: infeasible.
        return None
    cands = [p for p in range(GRID, Tv, GRID)
             if p >= min_gap and Tv - p >= min_gap]
    if len(cands) < N - 1:
        return None

    def cost(p):
        lo = max(0, p - window)
        hi = min(len(motion), p + window)
        if hi > lo:
            return float(motion[lo:hi].mean().item())
        return float(motion.mean().item())

    costs = [cost(p) for p in cands]
    nb = N - 1
    INF = float("inf")
    dp = [[INF] * len(cands) for _ in range(nb)]
    par = [[-1] * len(cands) for _ in range(nb)]
    for i in range(len(cands)):
        if cands[i] <= max_len:
            dp[0][i] = costs[i]
    for j in range(1, nb):
        for i in range(len(cands)):
            best, bi = INF, -1
            for k in range(i):
                gap = cands[i] - cands[k]
                if gap >= min_gap and gap <= max_len and dp[j - 1][k] < best:
                    best, bi = dp[j - 1][k], k
            if bi >= 0:
                dp[j][i] = best + costs[i]
                par[j][i] = bi
    # The last boundary must leave a final segment within max_len.
    best_i, best_v = -1, INF
    for i in range(len(cands)):
        if Tv - cands[i] <= max_len and dp[nb - 1][i] < best_v:
            best_v, best_i = dp[nb - 1][i], i
    if best_i < 0 or best_v == INF:
        return None
    chosen, j, i = [], nb - 1, best_i
    while j >= 0:
        chosen.append(cands[i])
        i = par[j][i]
        j -= 1
    chosen.reverse()
    return [0] + chosen + [Tv]


def _auto_overlaps(mot, Tv, inner_bounds, num_segments, window=15):
    """Per-boundary overlap from LOCAL motion (mot: normalized (Tv-1,) profile).

    A boundary sitting in a calm region gets a small overlap (less compute);
    one in a fast region gets a bigger overlap (safer join, max 10). Each
    overlap is capped so the total redundant work stays under ~40%.
    Returns (overlaps, scores) for the inner boundaries.
    """
    N = int(num_segments)
    cap = GRID * max(1, int(0.4 * Tv / max(1, N - 1) / GRID)) if N > 1 else GRID
    ovs, scores = [], []
    for p in inner_bounds:
        lo = max(0, p - window)
        hi = min(Tv - 1, p + window)
        if hi > lo:
            local = float(mot[lo:hi].median().item())
        else:
            local = float(mot.median().item())
        ovs.append(min(_motion_to_overlap(local), cap))
        scores.append(local)
    return ovs, scores


def _apply_blend(a_tail, b_head, mode, sharpness=6.0):
    """Join one overlap region with a concrete mode; returns same-shaped tensor."""
    ov = a_tail.shape[2]
    if mode == "midpoint":
        cut = ov // 2
        return torch.cat([a_tail[:, :, :cut], b_head[:, :, cut:]], dim=2)
    if mode == "multiband":
        return _blend_multiband(a_tail, b_head)
    s = _blend_weight(ov, mode, a_tail.device, a_tail.dtype,
                      a_tail, b_head, sharpness=sharpness)
    s = s.view(1, 1, ov, 1, 1)
    return (1 - s) * a_tail + s * b_head


def _score_blend(b, a_tail, b_head):
    """How good is a candidate blend? Lower is better.

    Two terms:
    - temporal: the blend's frame-to-frame motion vs the natural motion of
      the two sides. Uses mean deviation plus 2x the max deviation, so a
      single hard pop (e.g. a bad midpoint cut) scores badly even if the
      average looks fine.
    - fidelity (0.5x): how close each blended frame stays to ONE side. A
      frame sitting halfway between disagreeing sides is a visible double
      image; a frame equal to one side is clean.
    """
    ov = b.shape[2]
    if ov < 2:
        return 0.0
    with torch.no_grad():
        bf, af, hf = b.float(), a_tail.float(), b_head.float()
        db = (bf[:, :, 1:] - bf[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))
        da = (af[:, :, 1:] - af[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))
        dh = (hf[:, :, 1:] - hf[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))
        natural = 0.5 * (da + dh)
        scale = bf.std() + 1e-8
        dev = (db - natural).abs()
        temporal = (dev.mean() + 2.0 * dev.max()) / (natural.mean() + 0.05 * scale)
        side_gap = (af - hf).abs().mean()
        fidelity = (torch.minimum((bf - af).abs(), (bf - hf).abs()).mean()
                    / (side_gap + 0.05 * scale))
        score = float((temporal + 0.5 * fidelity).item())
    return score


_AUTO_CANDIDATES = ["linear", "smoothstep", "midpoint", "adaptive", "multiband"]


def _blend_auto(a_tail, b_head, sharpness=6.0, tag=""):
    """Try each candidate blend, measure, keep the best. Returns (blend, name)."""
    best_name, best_blend, best_score = None, None, float("inf")
    scores = {}
    for name in _AUTO_CANDIDATES:
        b = _apply_blend(a_tail, b_head, name, sharpness)
        s = _score_blend(b, a_tail, b_head)
        scores[name] = s
        if s < best_score:
            best_score, best_name, best_blend = s, name, b
    if tag:
        sstr = ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
        print(f"[H3TemporalSampler] auto-blend {tag}: -> {best_name} ({sstr})")
    return best_blend, best_name


def _merge_next(acc, nv, na, nv0, nv1, na0, na1, crossfade,
                blend_mode="linear", blend_sharpness=6.0, tag="", seam_log=None):
    """Incrementally merge one segment into the accumulator.

    acc is None or (mv, ma, pv1, pa1) where the accumulated result covers
    [0, pv1) video frames / [0, pa1) audio tokens. The new segment covers
    [nv0, nv1) / [na0, na1). Returns (mv, ma, nv1, na1, used_mode) where
    used_mode is the blend actually applied ('auto' resolves per boundary).
    When seam_log (a list) is given, (tag, used_mode, overlap, score) is
    appended for the quality report; lower score = cleaner join.
    """
    if acc is None:
        return (nv, na, nv1, na1, blend_mode)
    mv, ma, pv1, pa1 = acc
    ov = pv1 - nv0  # video overlap, in frames
    oa = pa1 - na0  # audio overlap, in tokens
    used = blend_mode
    if crossfade and ov > 0:
        a_tail, b_head = mv[:, :, pv1 - ov:pv1], nv[:, :, :ov]
        if blend_mode == "auto":
            # Try candidate blends, measure, keep the best for this boundary.
            blend_v, used = _blend_auto(a_tail, b_head,
                                        sharpness=blend_sharpness, tag=tag)
        else:
            blend_v = _apply_blend(a_tail, b_head, blend_mode, blend_sharpness)
        if seam_log is not None:
            try:
                s = _score_blend(blend_v, a_tail, b_head)
            except Exception:
                s = float("nan")
            seam_log.append((tag, used, ov, s))
        mv = torch.cat([mv[:, :, :pv1 - ov], blend_v, nv[:, :, ov:]], dim=2)
        if oa > 0:
            ta = torch.linspace(0, 1, oa, device=ma.device, dtype=ma.dtype)
            t_a = ta.view(1, 1, 1, oa)
            blend_a = (1 - ta) * ma[..., pa1 - oa:pa1] + ta * na[..., :oa]
            ma = torch.cat([ma[..., :pa1 - oa], blend_a, na[..., oa:]], dim=-1)
        else:
            ma = torch.cat([ma, na], dim=-1)
    else:
        # Hard cut: keep the accumulated tail, drop the new segment's head.
        if ov > 0:
            nv = nv[:, :, ov:]
        if oa > 0:
            na = na[..., oa:]
        mv = torch.cat([mv, nv], dim=2)
        ma = torch.cat([ma, na], dim=-1)
    return (mv, ma, nv1, na1, used)


class H3TemporalSampler:
    """Tiled SamplerCustomAdvanced for H3 AV latents.

    Inputs:  noise (NOISE), guider (GUIDER), sampler (SAMPLER), sigmas (SIGMAS),
             latent_image (LATENT) + tiling controls.
    With enable=False the node bypasses tiling and behaves exactly like a
    plain SamplerCustomAdvanced (single pass over the full latent).
    MultiDiffusion-style: all segments advance one denoising step at a time;
    after every step the overlap regions are averaged (cosine-ramped) and
    written back to both sides. Final assembly is a gentle linear join
    (the tiles already agree).
    smart_bounds=True places boundaries at motion valleys instead of even
    spacing (segments capped at the frame-pixel VRAM budget, so no giant
    tiles). A seam-quality score per boundary is printed at the end
    (lower is better; CHECK flags suspicious joins).
    Outputs: output (LATENT), denoised_output (LATENT).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True,
                                             "tooltip": "ON: tiled sampling with overlap blending. OFF: one plain pass over the full latent, like SamplerCustomAdvanced."}),
                "noise": ("NOISE", {"tooltip": "Full-length noise field; sliced per segment so overlaps share identical initial noise."}),
                "guider": ("GUIDER", {"tooltip": "Conditioning guider (prompt + reference). Applied identically to every segment."}),
                "sampler": ("SAMPLER", {"tooltip": "Sampler algorithm used for every segment."}),
                "sigmas": ("SIGMAS", {"tooltip": "Sigma schedule; identical for every segment so the tiles match."}),
                "latent_image": ("LATENT", {"tooltip": "Full-length H3 AV latent to denoise in tiled segments."}),
                "num_segments": ("INT", {"default": 4, "min": 1, "max": 10,
                                         "tooltip": "How many overlapping time segments the latent is split into."}),
                "smart_bounds": ("BOOLEAN", {"default": True,
                                             "tooltip": "ON: place segment boundaries at motion valleys (low-motion points) instead of even spacing. Falls back to even spacing if infeasible."}),
                "overlap_frames": ("INT", {"default": 10, "min": 5, "max": 20, "step": GRID,
                                           "tooltip": "Overlap between neighbours, snapped to multiples of 5 latent frames. Larger = smoother joins, more compute."}),
                "blend_mode": (["linear", "smoothstep", "adaptive"],
                               {"default": "adaptive",
                                "tooltip": "Join style for the final assembly: linear (straight ramp) / smoothstep (softer ends) / adaptive (rushes through disagreeing frames)."}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "h3 temporal tile"

    def sample(self, noise, guider, sampler, sigmas, latent_image,
               num_segments=4, overlap_frames=10,
               blend_mode="adaptive", enable=True,
               smart_bounds=True):
        import comfy.sample
        import comfy.utils
        import comfy.nested_tensor
        import comfy.model_management
        import latent_preview

        latent = dict(latent_image)
        samples = latent["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError(
                "[H3TemporalSampler] expected an H3 AV latent (nested samples), got "
                + str(type(samples)))

        # --- same preamble as SamplerCustomAdvanced ---
        samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, samples,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None))
        latent["samples"] = samples

        if not enable:
            # Bypass: behave exactly like a plain SamplerCustomAdvanced —
            # one guider.sample() call on the full latent, no tiling.
            mask = latent.get("noise_mask", None)
            full_noise = noise.generate_noise(latent)
            x0_output = {}
            callback = latent_preview.prepare_callback(
                guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            out = guider.sample(full_noise, samples, sampler, sigmas,
                                denoise_mask=mask, callback=callback,
                                disable_pbar=disable_pbar, seed=noise.seed)
            out = out.to(comfy.model_management.intermediate_device())
            out_latent = dict(latent)
            out_latent.pop("downscale_ratio_spacial", None)
            out_latent.pop("downscale_ratio_temporal", None)
            out_latent["samples"] = out
            if "x0" in x0_output:
                x0 = x0_output["x0"]
                if getattr(out, "is_nested", False) and not getattr(x0, "is_nested", False):
                    latent_shapes = [x.shape for x in out.unbind()]
                    x0 = comfy.nested_tensor.NestedTensor(
                        comfy.utils.unpack_latents(x0, latent_shapes))
                x0 = guider.model_patcher.model.process_latent_out(x0.cpu())
                out_denoised = dict(latent)
                out_denoised.pop("downscale_ratio_spacial", None)
                out_denoised.pop("downscale_ratio_temporal", None)
                out_denoised["samples"] = x0
            else:
                out_denoised = out_latent
            print("[H3TemporalSampler] tiling disabled -> single plain pass")
            return (out_latent, out_denoised)

        video, audio = _unbind_av(samples)  # [B,24,Tv,H,W], [B,32,2,Ta]
        Tv = video.shape[2]
        Ta = audio.shape[-1]

        # 1. segment count (manual)
        N_req = int(num_segments)
        N = max(1, min(10, N_req))

        # 2. motion profile, only if smart_bounds needs it
        need_motion = bool(smart_bounds)
        mot = _motion_profile(video) if need_motion else None

        # 3. boundary positions: smart (motion valleys) or even
        bounds = None
        if bool(smart_bounds) and N > 1:
            # Cap segment length at the frame-pixel VRAM budget so the DP
            # can't cluster boundaries and leave a VRAM-busting segment.
            # The cap applies to the FINAL overlapping segments, not just
            # the bounds: middle segments grow by 2*ov when overlap is
            # added, so the bounds are planned against (max_seg_len - 2*ov).
            H, W = video.shape[3], video.shape[4]
            seg_budget = 34 * 84 * 144  # his validated stage-2 reference
            max_seg_len = max(2 * GRID,
                              (seg_budget // max(1, H * W) // GRID) * GRID)
            _ov_in = int(overlap_frames)
            _max_ov = max(GRID, (_ov_in // GRID) * GRID)
            _bounds_max = max(2 * GRID, max_seg_len - _max_ov)
            bounds = _smart_bounds(Tv, mot, N, max_len=_bounds_max)
            if bounds is None:
                print("[H3TemporalSampler] smart bounds infeasible "
                      "-> even spacing")
            else:
                print(f"[H3TemporalSampler] smart bounds: {bounds} "
                      f"(max seg {max_seg_len}f, bounds capped at {_bounds_max}f "
                      f"for {_max_ov}f overlap)")

        # 4. overlap: manual (GRID-snapped)
        overlap_frames_in = int(overlap_frames)
        overlap = max(GRID, (overlap_frames_in // GRID) * GRID)
        segments = _plan_segments(Tv, Ta, N, overlap, bounds=bounds)
        if len(segments) != N_req:
            print(f"[H3TemporalSampler] num_segments adjusted to {len(segments)} "
                  f"for {Tv} latent frames")

        # The last segment's tail is never cross-faded, so edge-padding it
        # would imprint replicated frames onto the final output frames.
        # Extend it backward (grid-snapped) until it is the longest segment,
        # so it is never padded. The merge measures overlap from coordinates,
        # so a larger overlap stays seamless (identical noise in the overlap).
        if len(segments) > 1:
            longest = max(v1 - v0 for v0, v1, _, _ in segments)
            v0, v1, a0, a1 = segments[-1]
            need = longest - (v1 - v0)
            if need > 0:
                ext = ((need + GRID - 1) // GRID) * GRID
                v0 = max(0, v0 - ext)
                a0 = int(round(v0 * (Ta / Tv))) if Tv else a0
                segments[-1] = (v0, v1, a0, a1)

        # Full noise field once; sliced per segment so the overlap region sees
        # identical initial noise in both neighbours.
        full_noise = noise.generate_noise(latent)
        n_video, n_audio = _unbind_av(full_noise)

        mask = latent.get("noise_mask", None)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        acc = None       # (mv, ma, pv1, pa1) for the denoised samples
        acc_x0 = None    # (mv, ma, pv1, pa1) for the x0 predictions
        x0_ok = True
        seam_log = []    # (tag, used_mode, overlap, score) per boundary

        # Equal-shape padding: every segment is edge-padded to the longest
        # segment's video/audio length, then trimmed back after sampling.
        # Identical shapes across guider.sample() calls => the model compiler
        # initializes ONCE instead of once per segment.
        Lmax = max(v1 - v0 for v0, v1, _, _ in segments)
        Amax = max(a1 - a0 for _, _, a0, a1 in segments)

        crossfade = True
        blend_sharpness = 6.0
        seg_desc = ", ".join(f"[{v0},{v1})" for v0, v1, _, _ in segments)
        print(f"[H3TemporalSampler] {len(segments)} segments ({seg_desc}), "
              f"padded to {Lmax}f/{Amax}a (one model init), "
              f"overlap {overlap}f, {blend_mode if crossfade else 'hard cut'}")

        def _pad_edge(t, pad, dim):
            if pad <= 0:
                return t
            edge = t.narrow(dim, t.shape[dim] - 1, 1)
            rep = [1] * t.dim()
            rep[dim] = pad
            return torch.cat([t, edge.repeat(*rep).contiguous()], dim=dim)

        for si, (v0, v1, a0, a1) in enumerate(segments):
            lv, la = v1 - v0, a1 - a0
            pv, pa = Lmax - lv, Amax - la
            seg_v = _pad_edge(video[:, :, v0:v1].contiguous(), pv, 2)
            seg_a = _pad_edge(audio[..., a0:a1].contiguous(), pa, -1)
            seg_samples = _pack_av(seg_v, seg_a, template=samples)
            seg_noise = _pack_av(_pad_edge(n_video[:, :, v0:v1].contiguous(), pv, 2),
                                 _pad_edge(n_audio[..., a0:a1].contiguous(), pa, -1),
                                 template=full_noise)
            seg_mask = _slice_mask(mask, v0, v1, a0, a1, Tv, si)
            if seg_mask is not None and (pv or pa):
                if isinstance(seg_mask, torch.Tensor):
                    seg_mask = _pad_edge(seg_mask, pv, 2)
                else:
                    try:
                        mv_m, ma_m = _unbind_av(seg_mask)
                        seg_mask = type(seg_mask)([_pad_edge(mv_m, pv, 2),
                                                   _pad_edge(ma_m, pa, -1)])
                    except Exception:
                        seg_mask = None

            x0_output = {}
            callback = latent_preview.prepare_callback(
                guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

            # Same seed for every segment: identical per-step SDE noise in the
            # overlap -> the two neighbours agree there -> seamless cross-fade.
            out = guider.sample(seg_noise, seg_samples, sampler, sigmas,
                                denoise_mask=seg_mask, callback=callback,
                                disable_pbar=disable_pbar, seed=noise.seed)
            out = out.to(comfy.model_management.intermediate_device())
            o_video, o_audio = _unbind_av(out)
            if pv:
                o_video = o_video[:, :, :lv].contiguous()
            if pa:
                o_audio = o_audio[..., :la].contiguous()

            acc = _merge_next(acc, o_video, o_audio, v0, v1, a0, a1,
                              crossfade, blend_mode, blend_sharpness,
                              tag=f"boundary{si - 1}", seam_log=seam_log)
            used_mode = acc[4]
            acc = acc[:4]

            if "x0" in x0_output:
                x0 = x0_output["x0"]
                if getattr(out, "is_nested", False) and not getattr(x0, "is_nested", False):
                    latent_shapes = [x.shape for x in out.unbind()]
                    x0 = comfy.nested_tensor.NestedTensor(
                        comfy.utils.unpack_latents(x0, latent_shapes))
                x0 = guider.model_patcher.model.process_latent_out(x0.cpu())
                x0_video, x0_audio = _unbind_av(x0)
                if pv:
                    x0_video = x0_video[:, :, :lv].contiguous()
                if pa:
                    x0_audio = x0_audio[..., :la].contiguous()
                acc_x0 = _merge_next(acc_x0, x0_video, x0_audio, v0, v1, a0, a1,
                                   crossfade, used_mode, blend_sharpness)[:4]
            else:
                x0_ok = False

            del seg_samples, seg_noise, out, x0_output

        mv, ma, final_v1, final_a1 = acc
        out_latent = dict(latent)
        out_latent.pop("downscale_ratio_spacial", None)
        out_latent.pop("downscale_ratio_temporal", None)
        out_latent["samples"] = _pack_av(mv, ma, template=samples)

        if x0_ok and acc_x0 is not None:
            xmv, xma, _, _ = acc_x0
            out_denoised = dict(latent)
            out_denoised.pop("downscale_ratio_spacial", None)
            out_denoised.pop("downscale_ratio_temporal", None)
            out_denoised["samples"] = _pack_av(xmv, xma, template=samples)
        else:
            out_denoised = out_latent

        if seam_log:
            parts = []
            for t, mname, ov, s in seam_log:
                flag = "" if s < 1.0 else " CHECK"
                parts.append(f"{t}: {mname} ov={ov} score={s:.2f}{flag}")
            print("[H3TemporalSampler] seam quality (lower is better): "
                  + " | ".join(parts))

        print(f"[H3TemporalSampler] done -> video {tuple(mv.shape)} "
              f"audio {tuple(ma.shape)}")
        return (out_latent, out_denoised)



NODE_CLASS_MAPPINGS = {
    "H3TemporalSampler": H3TemporalSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TemporalSampler": "H3 Temporal Sampler",
}
