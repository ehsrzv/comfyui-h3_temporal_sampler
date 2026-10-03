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


class _HeldModelFallback(Exception):
    """Raised when the held-model fast path cannot be used safely."""


def _held_path_ok(guider):
    """Safety hatch for the held-model step_average path.

    We bypass guider.sample()'s per-call prepare/cleanup lifecycle and drive
    guider.inner_sample() directly with the model held loaded. That is only
    safe when the guider exposes the same inner entry point ComfyUI's own
    Guider.sample() uses (plus the conds/model_patcher attributes it needs).
    The OUTER_SAMPLE wrapper check happens in _held_setup, after the
    model_options are prepared, mirroring Guider.sample()'s ordering.
    """
    return (hasattr(guider, "inner_sample")
            and hasattr(guider, "original_conds")
            and hasattr(guider, "model_patcher"))


def _held_setup(guider, segs):
    """One-time model init for the whole step_average run.

    Replicates Guider.sample()'s preamble (conds refresh, model_options
    clone, hook prep) and outer_sample()'s setup (prepare_sampling,
    pre_run) exactly once. Raises _HeldModelFallback when a custom node
    wraps OUTER_SAMPLE (those wrappers would otherwise be silently
    skipped) so the caller can fall back to the legacy per-call path.
    """
    import comfy.hooks
    import comfy.model_management
    import comfy.model_patcher
    import comfy.patcher_extension
    import comfy.sampler_helpers
    import comfy.utils
    from comfy.samplers import (preprocess_conds_hooks,
                                get_total_hook_groups_in_conds,
                                filter_registered_hooks_on_conds,
                                cast_to_load_options)
    model_patcher = guider.model_patcher
    # --- Guider.sample() preamble (minus the outer_sample executor) ---
    guider.conds = {}
    for k in guider.original_conds:
        guider.conds[k] = list(map(lambda a: a.copy(),
                                   guider.original_conds[k]))
    preprocess_conds_hooks(guider.conds)
    orig_model_options = guider.model_options
    guider.model_options = comfy.model_patcher.create_model_options_clone(
        orig_model_options)
    orig_hook_mode = model_patcher.hook_mode
    if get_total_hook_groups_in_conds(guider.conds) <= 1:
        model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
    comfy.sampler_helpers.prepare_model_patcher(
        model_patcher, guider.conds, guider.model_options)
    filter_registered_hooks_on_conds(guider.conds, guider.model_options)
    # --- safety hatch: OUTER_SAMPLE wrappers must not be skipped ---
    wrappers = comfy.patcher_extension.get_all_wrappers(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        guider.model_options, is_model_options=True)
    if wrappers:
        raise _HeldModelFallback(
            f"{len(wrappers)} OUTER_SAMPLE wrapper(s) registered")
    # --- outer_sample() setup, hoisted out of the per-step loop ---
    # All segments are padded to identical shapes, so one probe is enough
    # (prepare_sampling only uses it for a memory estimate).
    probe, _ = comfy.utils.pack_latents(_unbind_av(segs[0]["pure"]))
    inner_model, run_conds, loaded_models = \
        comfy.sampler_helpers.prepare_sampling(
            model_patcher, probe.shape, guider.conds, guider.model_options)
    guider.inner_model = inner_model
    guider.conds = run_conds
    guider.loaded_models = loaded_models
    device = model_patcher.load_device
    cast_to_load_options(guider.model_options, device=device,
                         dtype=model_patcher.model_dtype())
    model_patcher.pre_run()
    return {"model_patcher": model_patcher, "device": device,
            "orig_model_options": orig_model_options,
            "orig_hook_mode": orig_hook_mode}


def _held_single(guider, ctx, sampler, n_in, latent_nested, sig_pair,
                 denoise_mask, callback, disable_pbar, seed):
    """One denoising step with the model held loaded.

    Replicates Guider.sample()'s per-call body (nested packing, mask prep,
    conds refresh, device moves) but calls guider.inner_sample() directly
    instead of going through outer_sample()'s prepare/cleanup lifecycle.
    Mathematically identical to one guider.sample() call.
    """
    import comfy.model_management
    import comfy.nested_tensor
    import comfy.sampler_helpers
    import comfy.utils
    from comfy.samplers import preprocess_conds_hooks
    model_patcher = ctx["model_patcher"]
    device = ctx["device"]
    if latent_nested.is_nested:
        latent_image, latent_shapes = comfy.utils.pack_latents(
            _unbind_av(latent_nested))
        noise, _ = comfy.utils.pack_latents(_unbind_av(n_in))
    else:
        latent_image, latent_shapes = latent_nested, [latent_nested.shape]
        noise = n_in
    cb = callback
    if len(latent_shapes) > 1 and cb is not None:
        packed_callback = cb

        def cb_wrap(step, x0, x, total_steps):
            x0 = comfy.nested_tensor.NestedTensor(
                comfy.utils.unpack_latents(x0, latent_shapes))
            x = comfy.nested_tensor.NestedTensor(
                comfy.utils.unpack_latents(x, latent_shapes))
            return packed_callback(step, x0, x, total_steps)
        cb = cb_wrap
    dm = denoise_mask
    if dm is not None:
        if dm.is_nested:
            denoise_masks = list(dm.unbind())
            denoise_masks = denoise_masks[:len(latent_shapes)]
        else:
            denoise_masks = [dm]
        for i in range(len(denoise_masks), len(latent_shapes)):
            denoise_masks.append(torch.ones(latent_shapes[i]))
        for i in range(len(denoise_masks)):
            denoise_masks[i] = comfy.sampler_helpers.prepare_mask(
                denoise_masks[i], latent_shapes[i],
                model_patcher.load_device)
        if len(denoise_masks) > 1:
            dm, _ = comfy.utils.pack_latents(denoise_masks)
        else:
            dm = denoise_masks[0]
        dm = dm.float()
    # conds refresh, exactly like Guider.sample() does per call
    # (inner_sample overwrites guider.conds with the processed ones).
    guider.conds = {}
    for k in guider.original_conds:
        guider.conds[k] = list(map(lambda a: a.copy(),
                                   guider.original_conds[k]))
    preprocess_conds_hooks(guider.conds)
    noise = noise.to(device=device, dtype=torch.float32)
    latent_image = latent_image.to(device=device, dtype=torch.float32)
    sigmas = sig_pair.to(device)
    with comfy.model_management.cuda_device_context(device):
        out_packed = guider.inner_sample(
            noise, latent_image, device, sampler, sigmas, dm, cb,
            disable_pbar, seed, latent_shapes=latent_shapes)
    if len(latent_shapes) > 1:
        out = comfy.nested_tensor.NestedTensor(
            comfy.utils.unpack_latents(out_packed, latent_shapes))
    else:
        out = out_packed
    return out


def _held_teardown(guider, ctx):
    """Release the held model. Mirrors outer_sample()'s finally block,
    then Guider.sample()'s finally block, in the original order."""
    import comfy.sampler_helpers
    from comfy.samplers import cast_to_load_options
    model_patcher = ctx["model_patcher"]
    try:
        model_patcher.cleanup()
        comfy.sampler_helpers.cleanup_models(guider.conds,
                                             guider.loaded_models)
    finally:
        cast_to_load_options(guider.model_options,
                             device=model_patcher.offload_device)
        guider.model_options = ctx["orig_model_options"]
        model_patcher.hook_mode = ctx["orig_hook_mode"]
        model_patcher.restore_hook_patches()
        for attr in ("conds", "inner_model", "loaded_models"):
            if hasattr(guider, attr):
                delattr(guider, attr)


class H3TemporalSampler:
    """Tiled SamplerCustomAdvanced for H3 AV latents (step_average only).

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
                "step_average": ("BOOLEAN", {"default": False,
                                             "tooltip": "ON: MultiDiffusion-style step-average — segments advance one denoising step at a time with per-step overlap consensus (cosine). OFF: sample each segment independently, then join with blend_mode."}),
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
               smart_bounds=True, step_average=False):
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

        # step_average: MultiDiffusion-style per-step consensus (independent mode).
        # Otherwise: legacy path — sample each segment independently, then
        # join with blend_mode.
        if step_average:
            return self._sample_step_average(
                noise, guider, sampler, sigmas, latent, samples,
                video, audio, Tv, Ta, segments, Lmax, Amax,
                n_video, n_audio, mask, disable_pbar, full_noise,
                6.0)

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

    def _sample_step_average(self, noise, guider, sampler, sigmas, latent,
                             samples, video, audio, Tv, Ta, segments,
                             Lmax, Amax, n_video, n_audio, mask, disable_pbar,
                             full_noise, blend_sharpness):
        """MultiDiffusion-style tiled sampling: step-major loop.

        All segments advance one denoising step at a time; after every step
        the TRUE overlap regions of neighbouring segments are averaged with
        cosine-ramped weights and written back to both sides, so the tiles
        converge to one consistent video instead of being blended afterwards.
        Final assembly is a gentle cosine join (the tiles already agree) and
        the usual seam-quality report still prints.

        Each single step reuses the exact guider.sample() machinery with a
        2-sigma slice. Two details keep it faithful to a full run:
        - guider.sample() expects (pure_noise, latent_image) and applies
          noise*sigma + latent_image itself, so the per-step noise input is
          (state - latent_image)/f(sigma) with the EPS-family scaling f, and
          the ORIGINAL segment latent stays the latent_image argument. Conds,
          denoise masks and inpainting therefore behave exactly as in a full
          run. (Supports EPS-family and CONST/flow noise scaling; the per-step
          inversion mirrors the model's own noise_scaling.)
        - Per-step SDE noise: ComfyUI seeds its noise generator once per
          sample() call, so a fresh call per step would repeat step-0 noise
          every step. We pass seed+t instead: identical across segments at
          step t (overlap agreement), varying across steps (proper
          stochastic behaviour), fully deterministic. The noise SEQUENCE
          differs from a full single-segment run by design; cross-segment
          agreement is what matters here.
        All segments stay padded to Lmax/Amax. The model is held loaded
        across the whole step loop (held-model path): Guider.sample()'s
        preamble and outer_sample()'s prepare/pre_run run once, then each
        step drives the REAL guider.inner_sample() directly, and cleanup
        runs once at the end -- so "Model Initializing" happens once per
        run instead of once per step. If the guider doesn't expose the
        inner entry point, if OUTER_SAMPLE wrappers are registered, or if
        the one-time setup fails, it falls back to the legacy per-call
        guider.sample() path (slower, same math). Averaging only touches
        true overlap coordinates (from the segment plan); padded tails are
        trimmed at the end.
        """
        import math
        import comfy.model_management
        import comfy.nested_tensor
        import comfy.utils
        import latent_preview

        n_seg = len(segments)

        # Sampling-family check: the per-step noise inversion must mirror the
        # model's noise_scaling / inverse_noise_scaling.
        # - EPS:   S(s,n,L) = s*n + L (sqrt(1+s^2)*n + L at max_denoise),
        #          I(s,x) = x            -> n_in = (state - L)/f
        # - CONST (flow): S(s,n,L) = s*ns*n + (1-s)*L,
        #          I(s,x) = x/(1-s)      -> n_in = (1-s)*(state - L)/(s*ns)
        # The combined class is always named "ModelSampling", so detect by
        # isinstance, not by name (a name check silently misses flow models).
        import comfy.model_sampling as _cms
        try:
            ms = guider.model_patcher.model.model_sampling
            ms_name = type(ms).__name__
            sigma_max = float(ms.sigma_max)
            is_const = isinstance(ms, _cms.CONST)
            noise_scale = float(getattr(ms, "noise_scale", 1.0))
        except Exception:
            ms_name, sigma_max, is_const, noise_scale = (
                "unknown", float(sigmas[0].item()), False, 1.0)
        if is_const:
            print(f"[H3TemporalSampler] step_average: CONST/flow sampling "
                  f"(noise_scale={noise_scale})")

        def _scale_f(s):
            # Mirror KSampler.max_denoise: sqrt(1+s^2) at full denoise.
            if math.isclose(sigma_max, s, rel_tol=1e-05) or s > sigma_max:
                return math.sqrt(1.0 + s * s)
            return s

        def _pad_edge(t, pad, dim):
            if pad <= 0:
                return t
            edge = t.narrow(dim, t.shape[dim] - 1, 1)
            rep = [1] * t.dim()
            rep[dim] = pad
            return torch.cat([t, edge.repeat(*rep).contiguous()], dim=dim)

        # Per-segment packed originals (padded), masks, real lengths.
        segs = []
        for si, (v0, v1, a0, a1) in enumerate(segments):
            lv, la = v1 - v0, a1 - a0
            pv, pa = Lmax - lv, Amax - la
            Lp = _pack_av(_pad_edge(video[:, :, v0:v1].contiguous(), pv, 2),
                          _pad_edge(audio[..., a0:a1].contiguous(), pa, -1),
                          template=samples)
            Np = _pack_av(_pad_edge(n_video[:, :, v0:v1].contiguous(), pv, 2),
                          _pad_edge(n_audio[..., a0:a1].contiguous(), pa, -1),
                          template=full_noise)
            mk = _slice_mask(mask, v0, v1, a0, a1, Tv, si)
            if mk is not None and (pv or pa):
                if isinstance(mk, torch.Tensor):
                    mk = _pad_edge(mk, pv, 2)
                else:
                    try:
                        mv_m, ma_m = _unbind_av(mk)
                        mk = type(mk)([_pad_edge(mv_m, pv, 2),
                                       _pad_edge(ma_m, pa, -1)])
                    except Exception:
                        mk = None
            segs.append({"L": Lp, "pure": Np, "mask": mk,
                         "v0": v0, "v1": v1, "a0": a0, "a1": a1,
                         "lv": lv, "la": la,
                         # Precomputed: unbound latent (for the inversion) and
                         # its device, so the per-step loop doesn't re-unbind.
                         "LvLa": _unbind_av(Lp)})

        # True per-boundary overlaps from coordinates (the last segment may
        # have been extended backward, enlarging its overlap).
        bnd_ov = []
        for i in range(n_seg - 1):
            bnd_ov.append((max(0, segs[i]["v1"] - segs[i + 1]["v0"]),
                           max(0, segs[i]["a1"] - segs[i + 1]["a0"])))

        # Precompute consensus blend weights (they don't change across steps).
        # Stored as (video_w, audio_w) per boundary, or None if no overlap.
        bnd_w = []
        for (ov, oa) in bnd_ov:
            vw = (_blend_weight(ov, "cosine", torch.device("cpu"),
                               torch.float32).view(1, 1, ov, 1, 1)
                  if ov > 0 else None)
            aw = (_blend_weight(oa, "cosine", torch.device("cpu"),
                               torch.float32).view(1, 1, 1, oa)
                  if oa > 0 else None)
            bnd_w.append((vw, aw))

        num_steps = sigmas.shape[-1] - 1
        print(f"[H3TemporalSampler] step_average: {n_seg} segments x "
              f"{num_steps} steps, consensus every step "
              f"(sampling={ms_name})")

        states = [None] * n_seg   # (video, audio) tensors, padded shape
        x0_parts = {}
        x0_ok = True
        base_seed = int(noise.seed)

        # Held-model fast path: prepare/compile the model ONCE for the whole
        # run and drive guider.inner_sample() directly per step, instead of
        # paying a full "Model Initializing" per guider.sample() call.
        # Falls back to the legacy per-call path when the guider doesn't
        # expose the inner entry point, when OUTER_SAMPLE wrappers are
        # registered, or when the one-time setup fails.
        held_ctx = None
        if _held_path_ok(guider):
            try:
                held_ctx = _held_setup(guider, segs)
            except _HeldModelFallback as e:
                print(f"[H3TemporalSampler] {e} -- legacy per-call path")
            except Exception as e:
                print(f"[H3TemporalSampler] held-model setup failed "
                      f"({type(e).__name__}: {e}) -- legacy per-call path")
        if held_ctx is not None:
            print("[H3TemporalSampler] step_average: held-model path "
                  "(single init for all steps x segments)")

            def do_sample(n_in, sg, sig_pair, cb, seed):
                return _held_single(guider, held_ctx, sampler, n_in, sg["L"],
                                    sig_pair, sg["mask"], cb,
                                    disable_pbar, seed)
        else:
            def do_sample(n_in, sg, sig_pair, cb, seed):
                return guider.sample(n_in, sg["L"], sampler, sig_pair,
                                     denoise_mask=sg["mask"], callback=cb,
                                     disable_pbar=disable_pbar, seed=seed)
        try:
            for t in range(num_steps):
                last = (t == num_steps - 1)
                s_t = float(sigmas[t].item())
                sig_pair = sigmas[t:t + 2]
                if is_const:
                    # CONST/flow: n_in = (1-s_t)*(state - L)/(s_t*ns).
                    # (inverse_noise_scaling divides by (1-s), so the state
                    #  must be re-scaled before inversion.)
                    c = ((1.0 - s_t) / (s_t * noise_scale)
                         if s_t > 0 else 0.0)
                else:
                    f = _scale_f(s_t)
                for si, sg in enumerate(segs):
                    if states[si] is None:
                        n_in = sg["pure"]
                    else:
                        Lv, La = sg["LvLa"]
                        sv, sa = states[si]
                        if is_const:
                            n_in = _pack_av(c * (sv - Lv), c * (sa - La),
                                            template=full_noise)
                        else:
                            n_in = _pack_av((sv - Lv) / f, (sa - La) / f,
                                            template=full_noise)
                    x0_output = {}
                    cb = (latent_preview.prepare_callback(
                        guider.model_patcher, 1, x0_output) if last else None)
                    out = do_sample(n_in, sg, sig_pair, cb, base_seed + t)
                    ov_v, ov_a = _unbind_av(out)
                    # Device invariant: the H3 guider returns `out` on the
                    # compute device while the segment latents live on CPU.
                    # Keep the per-step states on the segment-latent device
                    # so the noise inversion, the consensus write-back and
                    # the final assembly never mix devices.
                    _Lv0, _La0 = sg["LvLa"]
                    if ov_v.device != _Lv0.device:
                        ov_v = ov_v.to(_Lv0.device)
                    if ov_a.device != _La0.device:
                        ov_a = ov_a.to(_La0.device)
                    states[si] = (ov_v, ov_a)
                    if last:
                        if "x0" in x0_output:
                            x0_parts[si] = x0_output["x0"]
                        else:
                            x0_ok = False
                    del n_in, out, x0_output, cb

                # Consensus: average the true overlaps, write back to both sides.
                # Head/tail regions of a segment are disjoint, so boundaries can
                # be processed in order. Weights are precomputed (bnd_w).
                for i, ((ov, oa), (vw, aw)) in enumerate(zip(bnd_ov, bnd_w)):
                    if ov > 0:
                        lv_i = segs[i]["lv"]
                        sv_i = states[i][0]
                        sv_j = states[i + 1][0]
                        w = vw.to(device=sv_i.device, dtype=sv_i.dtype)
                        m = ((1 - w) * sv_i[:, :, lv_i - ov:lv_i]
                             + w * sv_j[:, :, :ov])
                        sv_i[:, :, lv_i - ov:lv_i] = m
                        sv_j[:, :, :ov] = m
                    if oa > 0:
                        la_i = segs[i]["la"]
                        sa_i = states[i][1]
                        sa_j = states[i + 1][1]
                        w = aw.to(device=sa_i.device, dtype=sa_i.dtype)
                        m = ((1 - w) * sa_i[..., la_i - oa:la_i]
                             + w * sa_j[..., :oa])
                        sa_i[..., la_i - oa:la_i] = m
                        sa_j[..., :oa] = m
        finally:
            if held_ctx is not None:
                _held_teardown(guider, held_ctx)

        # Final assembly: trim pads, gentle join with the selected blend mode
        # (tiles already agree), seam-quality report as usual.
        acc = None
        acc_x0 = None
        seam_log = []
        for si, sg in enumerate(segs):
            sv, sa = states[si]
            o_video = sv[:, :, :sg["lv"]].contiguous()
            o_audio = sa[..., :sg["la"]].contiguous()
            acc = _merge_next(acc, o_video, o_audio, sg["v0"], sg["v1"],
                              sg["a0"], sg["a1"], True, "linear",
                              blend_sharpness,
                              tag=f"boundary{si - 1}", seam_log=seam_log)
            acc = acc[:4]
            if x0_ok and si in x0_parts:
                x0 = x0_parts[si]
                if not getattr(x0, "is_nested", False):
                    shapes = [sv.shape, sa.shape]
                    x0 = comfy.nested_tensor.NestedTensor(
                        comfy.utils.unpack_latents(x0, shapes))
                x0 = guider.model_patcher.model.process_latent_out(x0.cpu())
                xv, xa = _unbind_av(x0)
                xv = xv[:, :, :sg["lv"]].contiguous()
                xa = xa[..., :sg["la"]].contiguous()
                acc_x0 = _merge_next(acc_x0, xv, xa, sg["v0"], sg["v1"],
                                     sg["a0"], sg["a1"], True, "linear",
                                     blend_sharpness)[:4]
            del sv, sa

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
            for tg, mname, ovv, s in seam_log:
                flag = "" if s < 1.0 else " CHECK"
                parts.append(f"{tg}: {mname} ov={ovv} score={s:.2f}{flag}")
            print("[H3TemporalSampler] seam quality (lower is better): "
                  + " | ".join(parts))

        print(f"[H3TemporalSampler] done (step_average) -> video "
              f"{tuple(mv.shape)} audio {tuple(ma.shape)}")
        return (out_latent, out_denoised)


NODE_CLASS_MAPPINGS = {
    "H3TemporalSampler": H3TemporalSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TemporalSampler": "H3 Temporal Sampler",
}
