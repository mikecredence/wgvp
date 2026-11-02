"""Subject selection in a combined two-view video, with the IMU pelvis cross-correlation as the primary selector.

The two views are already synchronised with each other (one file), so there is ONE unknown lag: the video time of
IMU t=0. Every track fragment (tracker id, split at identity jumps) in every view gets a correlation-vs-lag curve
against the IMU pelvis displacement; the subject is the fragment set, across both views, that shares the lag with
the highest joint support. Fragments too short or too weak to be scored (brief re-entries) are attached by
appearance (torso colour histogram) when they are clearly closer to the subject than to any concurrent person.
Seated or standing people have no gait-band pelvis oscillation and score ~0, however large they are in frame.

Secondary: walkway ROI (operator polygon); fallback: operator pick. All three are reported with their agreement.
"""
from __future__ import annotations

import numpy as np

from .signals import match_signals, combined_lag_curve, parabolic_peak, FS_MATCH
from .tracking import TrackSet, hip_vertical_signal, track_summary, VIS_FULL, VIS_PARTIAL

DEFAULTS = {
    "min_track_s": 2.0,          # fragments shorter than this are attached by appearance only
    "accept_r": 0.30,            # fine correlation needed to accept a fragment at the shared lag ...
    "min_excess": 0.15,          # ... and it must exceed the empirical chance level of its overlap / lag window by this much
    "min_margin": 0.12,          # (best accepted excess) - (best other person's excess) for a confident primary
    "min_r_primary": 0.35,       # best accepted r for a confident primary (with excess >= min_excess + 0.05)
    "seed_min_overlap_s": 5.0,   # only fragments at least this long may propose the shared lag
    "lag_tol_s": 0.12,           # a fragment's own best lag must sit within this of the shared lag (sync precision is ~20 ms)
    "lag_penalty": 2.0,          # excess is reduced by lag_penalty x |own best lag - shared lag| (s) when ranking / accepting
    "app_attach": 0.35,          # L1 distance (0..2) below which a fragment is attached by appearance ...
    "app_ratio": 0.7,            # ... and only if closer than app_ratio x its distance to the nearest other person
    "roi_min_cross": 0.3,
    "gap_min_s": 0.5,
}


# ----------------------------------------------------------------------------------------------------------------
def score_fragments(ts: TrackSet, imu: dict, params: dict) -> dict:
    ref = imu["disp_up"].astype(np.float64)
    rows = {}
    for tid in ts.track_ids():
        tr = ts.track(int(tid))
        summ = track_summary(ts, int(tid))
        rows[int(tid)] = {**summ, "match": {"ok": False, "reason": "too short"}, "tr": tr}
        if summ["duration_s"] < params["min_track_s"] or len(tr["pos"]) < 10:
            continue
        t, y = hip_vertical_signal(tr)
        if np.isfinite(y).sum() < 0.4 * len(y) or np.isfinite(y).sum() < 20:
            rows[int(tid)]["match"] = {"ok": False, "reason": "hips rarely visible"}; continue
        rows[int(tid)]["match"] = match_signals(t, y, ref)
    return rows


# Empirical chance level of the fine correlation for UNRELATED fragment/reference pairs (95th percentile), measured on
# 680 pairs from crowded public footage (tools/selection_test_real_footage.py + the calibration in the report):
# overlap 1.5-3 s: 0.82 (max over all lags) / 0.52 (max inside the +-0.15 s window); 3-5 s: 0.73 / 0.52; 5-8 s: 0.65 / 0.33.
# Longer overlaps follow the band-limited theory (about 5.5 independent samples per second, lag function correlated
# over ~10 samples). Values are interpolated in log(overlap) and blended in log(number of independent lags).
_CH_T = np.array([2.0, 4.0, 6.3, 10.0, 20.0, 45.0, 90.0])
_CH_ALL = np.array([0.82, 0.73, 0.65, 0.55, 0.42, 0.30, 0.22])    # ~1000 independent lag trials
_CH_WIN = np.array([0.52, 0.50, 0.33, 0.22, 0.14, 0.10, 0.07])    # ~3 independent lag trials (+-0.15 s window)


def chance_r(overlap_s: float, n_lags: int) -> float:
    """95th-percentile correlation an unrelated band-limited pair reaches by chance, for this overlap and this many
    lags searched. Correlations are judged by their EXCESS over this level (a 4-s fragment reaches 0.73 by chance
    over a full-range search; a 45-s one about 0.30)."""
    T = float(np.clip(overlap_s, _CH_T[0], _CH_T[-1]))
    lt = np.log(T)
    r_all = float(np.interp(lt, np.log(_CH_T), _CH_ALL)); r_win = float(np.interp(lt, np.log(_CH_T), _CH_WIN))
    n_ind = max(n_lags / 10.0, 1.5)
    w = float(np.clip((np.log(n_ind) - np.log(3.0)) / (np.log(1000.0) - np.log(3.0)), 0.0, 1.0))
    return (1 - w) * r_win + w * r_all


def _excess_at(m: dict, lag: float, tol: float, fs: float = FS_MATCH, n_competitors: int = 1, lag_penalty: float = 0.0) -> tuple[float, float]:
    """(r at lag, r - chance level for this fragment's overlap within the +-tol window). The chance level also grows
    with the number of scored fragments competing for the lag (the best of N unrelated people reaches more). With
    lag_penalty, the excess is reduced by lag_penalty x the distance (s) between the fragment's local peak and the
    shared lag: a co-walker whose bob aligns a few hundredths of a second off is penalised, the subject is not."""
    r = _r_at(m, lag, tol)
    if not np.isfinite(r):
        return np.nan, np.nan
    ex = r - chance_r(m.get("overlap_s", 0.0), (int(2 * tol * fs) + 1) * max(int(n_competitors), 1))
    if lag_penalty > 0:
        w = np.abs(m["lags_s"] - lag) <= tol
        v = np.where(w, m["r_fine_fn"], np.nan)
        if np.isfinite(v).any():
            i = int(np.nanargmax(v)); ex -= lag_penalty * abs(float(m["lags_s"][0] + parabolic_peak(np.nan_to_num(m["r_fine_fn"], nan=-1), i) / FS_MATCH) - lag)
    return r, ex


def _r_at(m: dict, lag: float, tol: float) -> float:
    if not m.get("ok"):
        return np.nan
    w = np.abs(m["lags_s"] - lag) <= tol
    v = np.where(w, m["r_fine_fn"], np.nan)
    return float(np.nanmax(v)) if np.isfinite(v).any() else np.nan


def _overlap(a: dict, b: dict) -> int:
    return len(np.intersect1d(a["pos"], b["pos"]))


def _median_iou(a: dict, b: dict) -> float:
    """Median box IoU over the frames two fragments share (duplicate detections of one person overlap heavily)."""
    common, ia, ib = np.intersect1d(a["pos"], b["pos"], return_indices=True)
    if len(common) < 3:
        return 0.0
    A = a["box"][ia]; B = b["box"][ib]
    x0 = np.maximum(A[:, 0], B[:, 0]); y0 = np.maximum(A[:, 1], B[:, 1]); x1 = np.minimum(A[:, 2], B[:, 2]); y1 = np.minimum(A[:, 3], B[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return float(np.median(inter / np.maximum(area, 1e-6)))


def _greedy_accept(frs: dict, lag: float, params: dict) -> tuple[set, float]:
    """Accept fragments at `lag` by excess correlation, best first, without temporal overlap; support = sum of
    excess x overlap (seconds, capped), so one long genuine match outweighs a chain of short chance matches."""
    acc, support = set(), 0.0
    n_comp = sum(1 for r in frs.values() if r["match"].get("ok"))
    ranked = sorted(frs.items(), key=lambda kv: -np.nan_to_num(_excess_at(kv[1]["match"], lag, params["lag_tol_s"], n_competitors=n_comp, lag_penalty=params["lag_penalty"])[1], nan=-9))
    for t, r in ranked:
        ra, ex = _excess_at(r["match"], lag, params["lag_tol_s"], n_competitors=n_comp, lag_penalty=params["lag_penalty"])
        if not np.isfinite(ra) or ra < params["accept_r"] or ex < params["min_excess"]:
            break
        if all(_overlap(r["tr"], frs[a]["tr"]) < 5 for a in acc):
            acc.add(t); support += ex * min(r["match"]["overlap_s"], 20.0)
    return acc, support


def _app_descriptor(frs: dict, tids) -> np.ndarray | None:
    A = np.concatenate([frs[t]["tr"]["app"] for t in tids]) if tids else np.zeros((0, 48))
    A = A[A.sum(1) > 0]
    return np.median(A, axis=0) if len(A) else None


def joint_selection(views: dict, imu: dict, params: dict | None = None, log=print) -> dict:
    """views: {view_id: TrackSet}. Returns the shared lag, accepted fragments per view, confidence and coverage."""
    p = {**DEFAULTS, **(params or {})}
    scored = {vid: score_fragments(ts, imu, p) for vid, ts in views.items()}
    out = {"method": "imu_xcorr_joint", "ok": False, "offset_s": None, "confidence": "none", "views": {}, "reason": "",
           "n_fragments": {vid: len(s) for vid, s in scored.items()}, "n_scored": {vid: sum(1 for r in s.values() if r["match"].get("ok")) for vid, s in scored.items()}}
    # candidate lags: the best lag of long fragments, ranked by excess over the chance level of a full-range search
    cands = []
    for vid, s in scored.items():
        ok = []
        for t, r in s.items():
            m = r["match"]
            if not m.get("ok") or m["overlap_s"] < p["seed_min_overlap_s"]:
                continue
            ex = m["r_fine"] - chance_r(m["overlap_s"], len(m["lags_s"]))
            ok.append((ex, t))
        ok.sort(reverse=True)
        for ex, t in ok[:6]:          # evaluated jointly below; acceptance and confidence do the filtering
            cands.append(float(s[t]["match"]["offset_s"]))
    if not cands:
        out["reason"] = f"no fragment of at least {p['seed_min_overlap_s']} s with visible hips correlates above chance level in any view"; return out
    # evaluate every candidate lag jointly over both views (lags within lag_tol_s of each other are one candidate)
    evals = []
    for L in cands:
        if any(abs(L - e[0]) <= p["lag_tol_s"] for e in evals):
            continue
        support = 0.0; acc = {}
        for vid, s in scored.items():
            a, sup = _greedy_accept(s, L, p); acc[vid] = a; support += sup
        evals.append((L, support, acc))
    evals.sort(key=lambda e: -e[1])
    L, support, acc = evals[0]
    out["lag_candidates"] = [{"offset_s": float(e[0]), "support": float(e[1])} for e in evals[:5]]
    out["lag_ambiguity"] = float(evals[1][1] / evals[0][1]) if len(evals) > 1 and evals[0][1] > 0 else 0.0
    # refine the lag on the combined curve of all accepted fragments, then re-accept
    for _ in range(2):
        curves = [(scored[vid][t]["match"]["lags_s"], scored[vid][t]["match"]["r_fine_fn"], scored[vid][t]["match"]["overlap_s"]) for vid in acc for t in acc[vid]]
        if not curves:
            break
        axis, R = combined_lag_curve(curves)
        w = np.abs(axis - L) <= 1.0; Rw = np.where(w, R, np.nan)
        if np.isfinite(Rw).any():
            i = int(np.nanargmax(Rw)); L = float(axis[0] + parabolic_peak(np.nan_to_num(R, nan=-1), i) / FS_MATCH)
        acc = {vid: _greedy_accept(s, L, p)[0] for vid, s in scored.items()}
    out["offset_s"] = L
    # per view: appearance attachment of unscored / short fragments, confidence, coverage
    overall_best, overall_margin = 0.0, 0.0
    any_ok = False
    for vid, s in scored.items():
        ts = views[vid]; A = acc[vid]
        subj_app = _app_descriptor(s, A)
        concurrent_all = [t for t in s if t not in A and any(_overlap(s[t]["tr"], s[a]["tr"]) >= 5 for a in A)]
        duplicates = [t for t in concurrent_all if max(_median_iou(s[t]["tr"], s[a]["tr"]) for a in A) > 0.4]   # second box on the subject
        concurrent = [t for t in concurrent_all if t not in duplicates]                                           # definitely other people
        other_apps = [np.median(s[t]["tr"]["app"][s[t]["tr"]["app"].sum(1) > 0], 0) for t in concurrent if (s[t]["tr"]["app"].sum(1) > 0).any()]
        attached = set()
        if subj_app is not None:
            for t, r in s.items():
                if t in A or t in concurrent or any(_overlap(r["tr"], s[a]["tr"]) >= 5 for a in A):
                    continue
                ra = _r_at(r["match"], L, p["lag_tol_s"])
                if np.isfinite(ra) and r["match"].get("overlap_s", 0) >= 3 and ra < 0.05:
                    continue   # scorable and contradicts the lag
                ap = r["tr"]["app"][r["tr"]["app"].sum(1) > 0]
                if not len(ap):
                    continue
                d_s = float(np.abs(np.median(ap, 0) - subj_app).sum())
                d_o = min([float(np.abs(np.median(ap, 0) - o).sum()) for o in other_apps], default=2.0)
                if d_s < p["app_attach"] and d_s < p["app_ratio"] * d_o:
                    attached.add(t)
        chosen = sorted(A | attached)
        # lag-consistency masks (identity switches inside a fragment)
        masks, mask_info = {}, {}
        for t in A:
            mk, info = consistency_mask(s[t]["tr"], imu, L)
            if info.get("kept_frac", 1.0) < 1.0:
                masks[int(t)] = mk
            mask_info[int(t)] = info
        n_comp = sum(1 for r in s.values() if r["match"].get("ok"))
        acc_scores = [_excess_at(s[t]["match"], L, p["lag_tol_s"], n_competitors=n_comp, lag_penalty=p["lag_penalty"]) for t in A]
        acc_scores = [(r, e) for r, e in acc_scores if np.isfinite(r)]
        best_r, best_e = (max(acc_scores, key=lambda x: x[1]) if acc_scores else (0.0, 0.0))
        # margin: at the chosen lag, the subject's excess against every other PERSON's excess (duplicate boxes excluded)
        others = []
        for t in s:
            if t in chosen or t in duplicates:
                continue
            r_o, e_o = _excess_at(s[t]["match"], L, p["lag_tol_s"], n_competitors=n_comp, lag_penalty=p["lag_penalty"])
            others.append(0.0 if (not np.isfinite(r_o) or r_o < p["accept_r"]) else max(e_o, 0.0))
        second = max(others, default=0.0)
        margin = float(best_e - second)
        conf = ("high" if (best_r >= p["min_r_primary"] and best_e >= p["min_excess"] + 0.05 and margin >= p["min_margin"])
                else ("low" if (best_r >= p["accept_r"] and best_e >= p["min_excess"]) else "none"))
        any_ok |= conf != "none"
        overall_best = max(overall_best, best_r); overall_margin = max(overall_margin, margin) if conf == "high" else overall_margin
        cov = coverage(ts, chosen, L, imu, p, masks)
        # this view's own estimate of the lag (its accepted fragments only), for the inter-view agreement check
        view_off = None
        vc = [(s[t]["match"]["lags_s"], s[t]["match"]["r_fine_fn"], s[t]["match"]["overlap_s"]) for t in A]
        if vc:
            ax, Rv = combined_lag_curve(vc); wv = np.abs(ax - L) <= 1.0; Rw = np.where(wv, Rv, np.nan)
            if np.isfinite(Rw).any():
                iv = int(np.nanargmax(Rw)); view_off = float(ax[0] + parabolic_peak(np.nan_to_num(Rv, nan=-1), iv) / FS_MATCH)
        out["views"][vid] = {"track_ids": [int(t) for t in chosen], "accepted_by_xcorr": sorted(int(t) for t in A), "attached_by_appearance": sorted(int(t) for t in attached),
                             "r_fine": float(best_r), "excess": float(best_e), "second_best_excess": float(second), "margin": float(margin), "confidence": conf, "view_offset_s": view_off,
                             "consistency": mask_info, "frame_masks": masks,
                             "fragments": {int(t): {"r_at_lag": (None if not np.isfinite(_r_at(s[t]["match"], L, p["lag_tol_s"])) else float(_r_at(s[t]["match"], L, p["lag_tol_s"]))),
                                                    "own_best_r": s[t]["match"].get("r_fine"), "own_offset_s": s[t]["match"].get("offset_s"), "r_env": s[t]["match"].get("r_env"),
                                                    "duration_s": s[t]["duration_s"], "locomotion_frac": s[t]["locomotion_frac"], "vis_full_frac": s[t]["vis_full_frac"],
                                                    "mean_box_h_frac": s[t]["mean_box_h_frac"], "role": ("subject" if t in chosen else ("duplicate_of_subject" if t in duplicates else ("concurrent_other" if t in concurrent else "unassigned"))),
                                                    "reason": s[t]["match"].get("reason", "")} for t in s},
                             "coverage": cov}
    out["ok"] = any_ok
    out["confidence"] = "high" if (overall_margin >= p["min_margin"] and overall_best >= p["min_r_primary"]) else ("low" if any_ok else "none")
    out["r_fine"] = overall_best; out["margin"] = overall_margin
    if not any_ok:
        out["reason"] = f"no fragment reaches accept_r {p['accept_r']} against the IMU pelvis signal in any view"
    return out


def consistency_mask(tr: dict, imu: dict, offset_s: float, win_s: float = 10.0, hop_s: float = 5.0, accept_r: float = 0.25) -> tuple[np.ndarray, dict]:
    """Which frames of an accepted fragment really follow the shared lag: the fragment's pelvis signal is correlated
    with the IMU reference in sliding windows (win_s, hop_s) at the shared lag (+-0.12 s); frames whose windows all
    fall below accept_r are dropped. Catches identity switches inside one tracker fragment (the subject and a
    co-walker crossing) that appearance cannot separate. Short fragments (< 1.5 win_s) are kept whole."""
    from .signals import bandpass, resample_to_grid, masked_xcorr
    t, y = hip_vertical_signal(tr)
    n = len(t); keep = np.ones(n, bool)
    span = float(np.nanmax(t) - np.nanmin(t)) if n else 0.0
    if span < 1.5 * win_s or np.isfinite(y).sum() < 60:
        return keep, {"windows": 0, "kept_frac": 1.0}
    ref = imu["disp_up"].astype(np.float64); fs = 100.0
    env = np.nan_to_num(np.asarray(imu.get("env"), dtype=np.float64)) if imu.get("env") is not None else None
    env_thr = 0.3 * np.nanpercentile(env, 90) if env is not None and np.isfinite(env).any() else None
    grid, v = resample_to_grid(t, y, fs, float(np.nanmin(t)), float(np.nanmax(t)))
    vb = bandpass(v, fs); ub = bandpass(ref, fs)
    starts = np.arange(float(np.nanmin(t)), float(np.nanmax(t)) - win_s + 1e-6, hop_s)
    win_r = []
    for s0 in starts:
        i0 = int((s0 - grid[0]) * fs); i1 = i0 + int(win_s * fs)
        seg = vb[i0:i1]
        if np.isfinite(seg).sum() < 0.5 * len(seg):
            win_r.append((s0, np.nan)); continue
        j0 = int((s0 - offset_s) * fs)                       # the IMU samples aligned with this window at the shared lag
        if j0 < 0 or j0 + len(seg) > len(ub):
            win_r.append((s0, np.nan)); continue
        if env_thr is not None and np.mean(env[j0:j0 + len(seg)]) < env_thr:
            win_r.append((s0, np.nan)); continue              # the IMU shows no gait here (standing): no evidence either way
        lags, r, nn = masked_xcorr(seg, ub[j0:j0 + len(seg)], 12, min_overlap=int(0.5 * len(seg)))
        win_r.append((s0, float(np.nanmax(r)) if np.isfinite(r).any() else np.nan))
    good = np.zeros(n, bool); scored = np.zeros(n, bool)
    for s0, r in win_r:
        m = (t >= s0) & (t < s0 + win_s)
        if np.isfinite(r):
            scored |= m
            if r >= accept_r:
                good |= m
    keep = good | ~scored
    return keep, {"windows": len(win_r), "kept_frac": float(keep.mean()), "window_r": [None if not np.isfinite(r) else round(r, 2) for _, r in win_r]}


def coverage(ts: TrackSet, tids: list[int], offset_s: float, imu: dict, p: dict, masks: dict | None = None) -> dict:
    """Which frames of the IMU recording this view covers with the subject, visibility classes, gaps, walk bouts."""
    t = ts.pts_ms / 1000.0 - offset_s
    inwin = (t >= 0) & (t <= imu["duration_s"])
    F = len(ts.frame_idx)
    vis = np.zeros(F, np.int8)
    for tid in tids:
        tr = ts.track(int(tid)); mk = (masks or {}).get(int(tid))
        pos = tr["pos"] if mk is None else tr["pos"][mk]; vv = tr["vis"] if mk is None else tr["vis"][mk]
        vis[pos] = np.maximum(vis[pos], vv + 0)   # vis class 0..2; presence counts as >=1
        vis[pos] = np.maximum(vis[pos], 1)
    n = int(inwin.sum())
    if n == 0:
        return {"frames_in_window": 0}
    present = vis[inwin] >= 1
    gaps = []
    tt = t[inwin]; m = np.r_[False, ~present, False]; d = np.diff(m.astype(int))
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if tt[min(b, len(tt) - 1)] - tt[a] >= p["gap_min_s"] or (a == 0 or b >= len(tt)):
            gaps.append([float(tt[a]), float(tt[min(b, len(tt) - 1)])])
    bouts = []
    for lab, a, b in imu["events"]:
        if lab != "Walk":
            continue
        sel = inwin & (t >= a) & (t <= b)
        nb = int(sel.sum()); cb = int((vis[sel] >= 1).sum())
        bouts.append({"t0": a, "t1": b, "covered_frac": (cb / nb) if nb else 0.0, "full_frac": (int((vis[sel] == VIS_FULL).sum()) / nb) if nb else 0.0})
    return {"frames_in_window": n, "present_frac": float(present.mean()), "full_frac": float((vis[inwin] == VIS_FULL).mean()),
            "partial_frac": float((vis[inwin] == VIS_PARTIAL).mean()), "absent_frac": float((vis[inwin] == 0).mean()),
            "gaps_imu_time_s": gaps[:100], "n_gaps": len(gaps), "walk_bouts": bouts}


# ----------------------------------------------------------------------------------------------------------------
def _poly_contains(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    import cv2
    return np.array([cv2.pointPolygonTest(poly.astype(np.float32), (float(x), float(y)), False) >= 0 for x, y in pts])


def roi_selection(ts: TrackSet, walkway: dict | None, params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    out = {"method": "walkway_roi", "ok": False, "track_ids": [], "reason": "", "candidates": {}}
    if not walkway:
        out["reason"] = "no walkway ROI for this view (optional; `wgvp set-roi`)"; return out
    poly = np.array(walkway["points"], np.float64); c = poly.mean(0); _, _, vt = np.linalg.svd(poly - c); axis = vt[0]
    extent = np.ptp((poly - c) @ axis); best = None
    for tid in ts.track_ids():
        tr = ts.track(int(tid)); feet = tr["kp"][:, [15, 16], :]
        ok = (feet[:, :, 2] >= 0.3).any(1)
        pts = np.where(ok[:, None], feet[:, :, :2].mean(1), tr["kp"][:, [11, 12], :2].mean(1))
        inside = _poly_contains(poly, pts); n_in = int(inside.sum())
        if n_in < 5:
            continue
        cross = float(np.ptp((pts[inside] - c) @ axis) / max(extent, 1e-6))
        out["candidates"][int(tid)] = {"frames_inside": n_in, "traversal": cross}
        if cross >= p["roi_min_cross"] and (best is None or n_in > out["candidates"][best]["frames_inside"]):
            best = int(tid)
    if best is None:
        out["reason"] = "no track traverses the walkway polygon"; return out
    out.update(ok=True, track_ids=[best]); return out


def pick_selection(pick: dict | None, ts: TrackSet) -> dict:
    out = {"method": "operator_pick", "ok": False, "track_ids": [], "reason": ""}
    if not pick:
        out["reason"] = "no operator pick (optional; `wgvp pick-subject`)"; return out
    tids = [int(t) for t in pick.get("track_ids", [])]; known = set(int(t) for t in ts.track_ids())
    bad = [t for t in tids if t not in known]
    if bad or not tids:
        out["reason"] = f"picked track ids {bad or tids} do not exist in this run's tracking (re-run pick-subject)"; return out
    out.update(ok=True, track_ids=tids, picked_at_frame=pick.get("frame")); return out


def decide_view(primary_view: dict | None, primary_conf: str, roi: dict, pick: dict) -> dict:
    """Per view: confident joint primary wins; else an operator pick; else a unique ROI candidate (flagged); else FAIL."""
    pv = primary_view or {}
    ptids = pv.get("track_ids", [])
    agree = {}
    if ptids and roi["ok"]:
        agree["primary_vs_roi"] = bool(set(roi["track_ids"]) & set(ptids))
    if ptids and pick["ok"]:
        agree["primary_vs_pick"] = bool(set(pick["track_ids"]) & set(ptids))
    if roi["ok"] and pick["ok"]:
        agree["roi_vs_pick"] = bool(set(pick["track_ids"]) & set(roi["track_ids"]))
    if ptids and primary_conf == "high":
        chosen, src, status = ptids, "imu_xcorr", "PASS"
        if pv.get("confidence") != "high":
            status = "WARN"; src = "imu_xcorr(this view weak; lag from the other view)"
        if agree.get("primary_vs_roi") is False or agree.get("primary_vs_pick") is False:
            status = "WARN"
    elif pick["ok"]:
        chosen, src, status = pick["track_ids"], "operator_pick", "WARN"
    elif ptids:
        chosen, src, status = ptids, "imu_xcorr(low)", "WARN"
    elif roi["ok"]:
        chosen, src, status = roi["track_ids"], "walkway_roi", "WARN"
    else:
        chosen, src, status = [], "none", "FAIL" if primary_conf == "none" else "WARN"
    msg = {"PASS": "subject identified by IMU cross-correlation with a clear margin",
           "WARN": "subject chosen but not confirmed by an independent method, or absent from this view - check the skeleton thumbnails; an operator pick (`wgvp pick-subject`) overrides",
           "FAIL": "no subject could be chosen: run `wgvp pick-subject` on a skeleton-only frame, or set a walkway ROI (`wgvp set-roi`)"}[status]
    return {"track_ids": chosen, "source": src, "status": status, "agreement": agree, "message": msg, "primary": pv, "walkway_roi": roi, "operator_pick": pick}


def merged_subject_track(ts: TrackSet, tids: list[int], masks: dict | None = None) -> dict:
    """Concatenate accepted fragments (optionally masked to their lag-consistent frames) into one subject track."""
    parts = []
    for t in tids:
        tr = ts.track(int(t)); mk = (masks or {}).get(int(t))
        if mk is not None:
            tr = {k: (v[mk] if isinstance(v, np.ndarray) and v.shape[:1] == mk.shape else v) for k, v in tr.items()}
        if len(tr["pos"]):
            parts.append(tr)
    if not parts:
        return {"tids": [int(t) for t in tids], "pos": np.zeros(0, int), "frame_idx": np.zeros(0, int), "pts_ms": np.zeros(0), "box": np.zeros((0, 4)), "conf": np.zeros(0), "kp": np.zeros((0, 17, 3)), "vis": np.zeros(0, np.int8), "app": np.zeros((0, 48)), "src_tid": np.zeros(0, int)}
    pos = np.concatenate([p["pos"] for p in parts]); order = np.argsort(pos, kind="stable")
    pos = pos[order]; keep = np.concatenate([[True], np.diff(pos) > 0]); sel = order[keep]
    cat = lambda k: np.concatenate([p[k] for p in parts])[sel]
    return {"tids": [int(t) for t in tids], "pos": pos[keep], "frame_idx": cat("frame_idx"), "pts_ms": cat("pts_ms"), "box": cat("box"),
            "conf": cat("conf"), "kp": cat("kp"), "vis": cat("vis"), "app": cat("app"),
            "src_tid": np.concatenate([np.full(len(p["pos"]), p["tid"]) for p in parts])[sel]}
