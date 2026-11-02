"""Sync light detection: a brightness STEP that persists, either in an operator ROI or found automatically on the
coarse brightness grid (local steps only; global exposure changes are removed by subtracting the frame median)."""
from __future__ import annotations

import numpy as np


def _step_candidates(s: np.ndarray, fps: float, k_s: float = 0.4, persist_s: float = 1.0, z_min: float = 6.0,
                     min_step: float = 12.0, max_cands: int = 5) -> list[dict]:
    """Onsets in a 1-D brightness series: mean(after) - mean(before) over k_s windows, z-scored by the series'
    robust frame-to-frame noise; the post level must persist for persist_s with no large drop."""
    s = np.asarray(s, dtype=np.float64)
    n = len(s); k = max(2, int(round(k_s * fps))); P = max(k, int(round(persist_s * fps)))
    if n < 2 * k + 2:
        return []
    c = np.concatenate([[0.0], np.cumsum(s)])
    before = (c[k:n - k] - c[0:n - 2 * k]) / k          # mean s[t-k:t], t = k..n-k-1
    after = (c[2 * k:n] - c[k:n - k]) / k               # mean s[t:t+k]
    d = after - before
    noise = max(1.4826 * np.median(np.abs(np.diff(s) - np.median(np.diff(s)))), 0.5)
    z = d / (noise * np.sqrt(2.0 / k))
    t_axis = np.arange(k, n - k)
    cands = []
    order = np.argsort(-d)
    for i in order:
        if d[i] < min_step or z[i] < z_min:
            break
        t = int(t_axis[i])
        if any(abs(t - c0["onset_frame_pos"]) < 2 * k for c0 in cands):
            continue
        post = s[t:min(n, t + k + P)]
        pre = s[max(0, t - k - P):t]
        if len(post) < k:
            continue
        persist_ok = (post.min() > before[i] + 0.5 * d[i]) and (len(pre) < k or pre.max() < before[i] + 0.5 * d[i])
        # refine onset: first frame whose level exceeds the midpoint between the two plateaus
        mid = before[i] + 0.5 * d[i]
        seg = s[max(0, t - k):min(n, t + k)]
        above = np.where(seg > mid)[0]
        onset = int(max(0, t - k) + above[0]) if len(above) else t
        cands.append({"onset_frame_pos": onset, "step": float(d[i]), "z": float(z[i]), "pre_level": float(before[i]),
                      "post_level": float(after[i]), "persistent": bool(persist_ok)})
        if len(cands) >= max_cands:
            break
    return cands


def detect_flash_roi(series: np.ndarray, frame_idx: np.ndarray, pts_ms: np.ndarray, fps: float, **kw) -> list[dict]:
    cands = _step_candidates(series, fps, **kw)
    for c in cands:
        p = c.pop("onset_frame_pos")
        c.update(onset_frame=int(frame_idx[p]), onset_t_s=float(pts_ms[p] / 1000.0), source="roi")
    return sorted(cands, key=lambda c: c["onset_t_s"])


def detect_flash_auto(grid: np.ndarray, frame_idx: np.ndarray, pts_ms: np.ndarray, fps: float, z_min: float = 10.0,
                      min_step: float = 25.0, max_cands: int = 5) -> list[dict]:
    """Search every grid cell for a persistent local brightness step. Returns the strongest candidates (by z)."""
    g = grid.astype(np.float64)                     # (F, ny, nx)
    F, ny, nx = g.shape
    med = np.median(g.reshape(F, -1), axis=1)       # global exposure trend
    loc = g - med[:, None, None]
    cands = []
    for iy in range(ny):
        for ix in range(nx):
            for c in _step_candidates(loc[:, iy, ix], fps, z_min=z_min, min_step=min_step, max_cands=2):
                if not c["persistent"]:
                    continue
                p = c.pop("onset_frame_pos")
                c.update(onset_frame=int(frame_idx[p]), onset_t_s=float(pts_ms[p] / 1000.0), cell=[int(ix), int(iy)],
                         source="auto")
                cands.append(c)
    cands.sort(key=lambda c: -c["z"])
    # merge candidates at the same onset (neighbouring cells of one light)
    merged = []
    for c in cands:
        if any(abs(c["onset_frame"] - m["onset_frame"]) <= 2 for m in merged):
            continue
        merged.append(c)
    return merged[:max_cands]


def nearest_flash(cands: list[dict], t_s: float, tol_s: float) -> dict | None:
    best = None
    for c in cands:
        d = abs(c["onset_t_s"] - t_s)
        if d <= tol_s and (best is None or d < abs(best["onset_t_s"] - t_s)):
            best = c
    return best
