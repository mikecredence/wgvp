"""Two compromised views, one 3D pose: how much of the walker's true pose does each view recover, and does fusing
the two recover more? Runs on the synthetic session HC901 (sample_v2/, rendered with the walker's ground-truth SMPL
pose saved) after the pipeline has lifted both views with GVHMR.

    python tools/two_view_fusion.py <data root with wgvp_output/HC901> [--gt sample_v2/HC901_gt.npz] [--out docs/two_view_fusion]

Per view, per frame and per joint: the geodesic angle between GVHMR's joint rotation and the truth (SMPL-X body
joints 1-21 are the SMPL joints 1-21). Frames are classed by what the camera could see of the walker (from the
ground truth: head out of frame, shoulders out, hips out). Fusion rules, evaluated against the same truth:

  single view         each view alone (what a one-camera lab gets)
  mean                per-joint rotation average of the two views (chordal mean), equal weights
  confidence-weighted per-joint weights from what is available at run time: the view's ViTPose confidence for the
                      keypoints that joint moves, zero when the frame was bridged (visible = False)
  visibility-weighted per-joint weights 1 / 0 by whether the joint and the segment it moves are inside the view's
                      frame (here from the rendered truth's in-frame flags; in deployment the same test comes from
                      the 2D keypoints against the panel edges), averaged where both views see the joint
  segment rule        a joint only from the view(s) whose keypoints for it are confident and inside the panel,
                      averaged when both, nothing when neither; cross-faded over 15 frames at segment edges;
                      optionally each view's weights scaled by its share of the fused root (fuse_root) and the
                      result smoothed over 5 frames (smooth_rotations): this last variant is what the demo shows
  oracle pick         per joint, the view with the smaller error (an upper bound: needs the truth)

Root orientation is compared after removing the constant heading offset between each GVHMR world frame and the
render's (a single yaw per view, fitted on visible frames); body-joint rotations need no alignment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from wgvp.reconcile import SMPLX_BODY_JOINTS      # noqa: E402

LEGS = [0, 1, 3, 4, 6, 7]; ARMS = [12, 13, 15, 16, 17, 18, 19, 20]; SPINE = [2, 5, 8, 11, 14]
# SMPL-X body joint -> COCO-17 keypoints whose confidence says how well this joint's rotation is observed
# (the joint itself and the child segment it moves)
JOINT_KP = {0: [11, 13], 1: [12, 14], 2: [11, 12, 5, 6], 3: [13, 15], 4: [14, 16], 5: [5, 6, 11, 12], 6: [15], 7: [16], 8: [5, 6], 9: [15], 10: [16],
            11: [5, 6], 12: [5], 13: [6], 14: [5, 6], 15: [5, 7], 16: [6, 8], 17: [7, 9], 18: [8, 10], 19: [9], 20: [10]}
# SMPL joint ids used for the visibility classes of a frame (from the ground-truth in-frame flags)
SMPL_HEAD, SMPL_SHOULDERS, SMPL_HIPS, SMPL_ANKLES = 15, [16, 17], [1, 2], [7, 8]


def usable_joints_rt(kp2d: np.ndarray, view_box, visible: np.ndarray, min_conf: float = 0.3, margin: float = 8.0) -> np.ndarray:
    """Run-time test, no truth involved: SMPL-X body joint j is usable in a frame when the ViTPose keypoints it moves
    are confident and sit inside the view's panel with a margin, and the frame was not bridged. -> (T, 21) bool."""
    conf = np.nan_to_num(kp2d[:, :, 2], nan=0.0)
    if view_box is not None and len(view_box) == 4 and (view_box[2] - view_box[0]) > 0:
        x0, y0, x1, y1 = [float(b) for b in view_box]
        ins = (conf >= min_conf) & (kp2d[:, :, 0] > x0 + margin) & (kp2d[:, :, 0] < x1 - margin) & (kp2d[:, :, 1] > y0 + margin) & (kp2d[:, :, 1] < y1 - margin)
    else:
        ins = conf >= min_conf
    return np.stack([ins[:, JOINT_KP[j]].all(1) for j in range(21)], 1) & np.asarray(visible, bool)[:, None]


def ramp_inside(mask: np.ndarray, k: int) -> np.ndarray:
    """(T, J) bool -> (T, J) float weights that rise linearly from 0 to 1 over the first k frames of every run of
    True and fall over the last k, and are exactly 0 where the mask is False. Used to cross-fade between views at
    the edges of their usable segments; where only one view is usable the weights renormalise to that view and the
    ramp has no effect."""
    m = np.asarray(mask, bool); T, J = m.shape; w = np.zeros((T, J), float)
    for j in range(J):
        edges = np.flatnonzero(np.diff(np.r_[0, m[:, j].astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            n = b - a; i = np.arange(n)
            w[a:b, j] = np.minimum(1.0, np.minimum((i + 1) / k, (n - i) / k))
    return w


def smooth_rotations(aa: np.ndarray, window: int) -> np.ndarray:
    """Moving average of rotations over time, (T, J, 3) axis-angle -> same, computed as the normalised mean of unit
    quaternions (hemisphere-aligned to the window's centre) over `window` frames, within runs of emitted frames
    only: NaN gaps stay gaps and are not averaged across. window <= 1 returns the input."""
    if window <= 1:
        return aa
    T, J = aa.shape[:2]; out = np.full_like(aa, np.nan); h = window // 2
    for j in range(J):
        ok = np.isfinite(aa[:, j]).all(1)
        if not ok.any():
            continue
        q = np.full((T, 4), np.nan); q[ok] = R.from_rotvec(aa[ok, j]).as_quat()
        for t in np.flatnonzero(ok):
            a, b = max(0, t - h), min(T, t + h + 1)
            seg = q[a:b]; m = np.isfinite(seg).all(1)
            # stay inside this run of emitted frames
            idx = np.arange(a, b)[m]; left = idx[idx <= t]; right = idx[idx >= t]
            left = left[np.r_[True, np.diff(left) == 1][::-1].cumprod()[::-1].astype(bool)] if len(left) else left
            right = right[np.r_[True, np.diff(right) == 1].cumprod().astype(bool)] if len(right) else right
            use = np.union1d(left, right); qs = q[use]; qs = qs * np.sign(qs @ q[t] + 1e-12)[:, None]
            out[t, j] = R.from_quat(qs.mean(0) / np.linalg.norm(qs.mean(0))).as_rotvec()
    return out


def fuse_root(roots: list[np.ndarray], usable: list[np.ndarray], lifted: list[np.ndarray], ramp: int = 15,
              soft_lo: float = 30.0, soft_hi: float = 60.0, n_joints: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Root orientation of the merge and each view's share of it. Inputs per view, all in one world frame (per-view
    heading offsets already removed): roots (T, 3), usable (T, J) bool, lifted (T,) bool. Returns the fused root
    (T, 3) axis-angle, NaN where no view lifted the frame, and the normalised weights (V, T).

    Each view's root carries the weight  presence x (usable fraction)^2: presence ramps from 0 to 1 over the first
    and last `ramp` frames of every lifted run, so a view that enters or leaves cannot step the root, and the usable
    fraction is averaged over `ramp` frames, so joints flickering in and out of a view cannot either; a view that
    sees 5 of 21 joints carries 6 % of the weight of one that sees them all. When the roots disagree by more than
    soft_lo degrees one of them is wrong (two good lifts of the same walker agree to within ~10 deg), so the weights
    are sharpened towards the heaviest view, fully by soft_hi; the sharpening is a continuous function of the weights
    and the disagreement, so nothing here introduces a discontinuity that is not in the inputs. The weighted mean is
    taken on unit quaternions, hemisphere-aligned to the heaviest view.

    The share is also the right weight for the view's body joints: a lifter that got the root's pitch wrong (the
    along view of the sample at close range: +40 deg) has hips that compensate for it, and the rest of its body
    solved in the wrong gravity frame."""
    V = len(roots); T = roots[0].shape[0]
    L = np.stack([np.asarray(l, bool) for l in lifted])                                                     # (V, T)
    pres = np.stack([ramp_inside(L[i][:, None], ramp)[:, 0] for i in range(V)])
    frac = np.stack([_moving_average(np.asarray(u, bool).sum(1) / float(n_joints), ramp) for u in usable])
    w = pres * frac ** 2; w[~L] = 0.0
    Q = np.stack([np.where(np.isfinite(r).all(1)[:, None], R.from_rotvec(np.nan_to_num(r)).as_quat(), np.nan) for r in roots])
    out = np.full((T, 3), np.nan); share = np.zeros((V, T))
    for t in range(T):
        idx = np.flatnonzero(L[:, t] & np.isfinite(Q[:, t]).all(1))
        if not len(idx):
            continue
        ww = w[idx, t].copy()
        if ww.sum() <= 1e-9:                                   # lifted but nothing usable in any view (bridged frames): plain mean
            ww[:] = 1.0
        k = int(np.argmax(ww)); q = Q[idx, t].copy(); q *= np.sign(q @ q[k] + 1e-12)[:, None]
        d = np.degrees(2.0 * np.arccos(np.clip(np.abs(q @ q[k]), 0.0, 1.0)))                               # each root's angle to the heaviest
        p = 8.0 * np.clip((d - soft_lo) / (soft_hi - soft_lo), 0.0, 1.0)                                    # 0 below soft_lo, 8 at soft_hi
        ww = ww * (ww / ww[k]) ** p; share[idx, t] = ww / ww.sum()
        m = (ww[:, None] * q).sum(0); out[t] = R.from_quat(m / np.linalg.norm(m)).as_rotvec()
    return out, share


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(x, float)
    c = np.cumsum(np.r_[0.0, x]); h = k // 2; i = np.arange(len(x)); a_ = np.clip(i - h, 0, len(x)); b_ = np.clip(i + h + 1, 0, len(x))
    return (c[b_] - c[a_]) / (b_ - a_)


def export_mask(merged_out: np.ndarray, required=LEGS, min_run: int = 15) -> np.ndarray:
    """Frames an export keeps: every required joint is output (seen by at least one view), in runs of at least
    min_run frames. (T, 21, 3) with NaN for joints not output -> (T,) bool."""
    ok = np.isfinite(merged_out[:, list(required)]).all(axis=(1, 2))
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a_, b_ in zip(edges[::2], edges[1::2]):
        if b_ - a_ < min_run:
            ok[a_:b_] = False
    return ok


def step_stats(aa: np.ndarray) -> dict:
    """Frame-to-frame continuity of an estimate (T, 21, 3): largest per-joint rotation step between consecutive
    frames where both are emitted; p99 and the share of steps over 20 deg."""
    ok = np.isfinite(aa).all(2)                                          # (T, 21)
    both = ok[1:] & ok[:-1]
    if not both.any():
        return {"p99_deg": float("nan"), "frac_over_20": float("nan"), "n": 0}
    a0 = aa[:-1][both]; a1 = aa[1:][both]
    d = np.degrees((R.from_rotvec(a0).inv() * R.from_rotvec(a1)).magnitude())
    return {"p99_deg": float(np.percentile(d, 99)), "frac_over_20": float((d > 20).mean()), "n": int(len(d))}


def geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: (..., 3) axis-angle -> degrees; NaN rotations (no output) give NaN."""
    sh = a.shape[:-1]; a2 = a.reshape(-1, 3); ok = np.isfinite(a2).all(1); out = np.full(len(a2), np.nan)
    out[ok] = np.degrees((R.from_rotvec(a2[ok]).inv() * R.from_rotvec(b.reshape(-1, 3)[ok])).magnitude())
    return out.reshape(sh)


def chordal_mean(aa_list: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    """Weighted rotation average per (frame, joint): aa_list = [(T,J,3)] per view, w (V,T,J). Quaternion method
    (largest eigenvector of the weighted outer-product sum), exact for two rotations."""
    V = len(aa_list); T, J = aa_list[0].shape[:2]
    Q = np.stack([R.from_rotvec(a.reshape(-1, 3)).as_quat().reshape(T, J, 4) for a in aa_list])     # (V,T,J,4)
    out = np.zeros((T, J, 3), np.float32)
    for t in range(T):
        for j in range(J):
            ww = w[:, t, j]
            if ww.sum() <= 0:
                ww = np.ones(V)
            q = Q[:, t, j]; q = q * np.sign(np.sum(q * q[:1], axis=1, keepdims=True) + 1e-12)   # same hemisphere as view 0
            M = sum(wi * np.outer(qi, qi) for wi, qi in zip(ww, q))
            vals, vecs = np.linalg.eigh(M); out[t, j] = R.from_quat(vecs[:, -1]).as_rotvec()
    return out


def fit_yaw(aa_est: np.ndarray, aa_gt: np.ndarray, mask: np.ndarray) -> float:
    """Yaw (deg, about world y) that best maps the estimated root orientations onto the truth."""
    best = (1e9, 0.0)
    for th in np.arange(-180, 180, 1.0):
        Ry = R.from_euler("y", th, degrees=True)
        e = np.degrees((Ry * R.from_rotvec(aa_est[mask])).inv().__mul__(R.from_rotvec(aa_gt[mask])).magnitude()).mean()
        if e < best[0]:
            best = (e, th)
    return best[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("--session", default="HC901"); ap.add_argument("--recording", default="hc901_selfpace")
    ap.add_argument("--gt", default=str(PKG / "sample_v2" / "HC901_gt.npz")); ap.add_argument("--out", default=str(PKG / "docs" / "two_view_fusion"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    deliver = Path(a.root) / "wgvp_output" / a.session / "deliver" / a.recording
    gt = np.load(a.gt); views = sorted(p.name for p in deliver.iterdir() if (p / "smpl_global.npz").exists())
    assert len(views) == 2, f"need two lifted views under {deliver}, found {views}"
    Z = {v: np.load(deliver / v / "smpl_global.npz") for v in views}; K = {v: np.load(deliver / v / "kp2d_vitpose.npz") for v in views}
    # common frames (combined-video frame indices) between the truth and both views
    frames = np.array(sorted(set(gt["frame_idx"].tolist()) & set(Z[views[0]]["frame_idx"].tolist()) & set(Z[views[1]]["frame_idx"].tolist())))
    gi = {int(f): i for i, f in enumerate(gt["frame_idx"])}; sel_gt = [gi[int(f)] for f in frames]
    gt_body = gt["pose_aa"][sel_gt, 3:66].reshape(-1, 21, 3); gt_root = gt["pose_aa"][sel_gt, :3]
    est, root, w_conf, visible, in_frame, in_frame_rt = {}, {}, {}, {}, {}, {}
    for v in views:
        z, k = Z[v], K[v]; zi = {int(f): i for i, f in enumerate(z["frame_idx"])}; s = [zi[int(f)] for f in frames]
        est[v] = z["body_pose"][s].reshape(-1, 21, 3); root[v] = z["global_orient"][s]; visible[v] = z["visible"][s].astype(bool)
        conf = np.nan_to_num(k["kp2d"][s][:, :, 2], nan=0.0)                               # (T,17) ViTPose confidence (face points blanked -> 0)
        in_frame_rt[v] = usable_joints_rt(k["kp2d"][s], k["view_box"] if "view_box" in k.files else None, visible[v])
        jin = gt[f"{v}_joints_in"][sel_gt]                                                   # (T,24) truth: joint inside this view
        w = np.stack([conf[:, JOINT_KP[j]].mean(1) for j in range(21)], 1)                   # (T,21)
        w = w * visible[v][:, None]                                                          # bridged frames carry no weight
        w_conf[v] = w; in_frame[v] = jin
    # ---- visibility classes per view from the truth
    def vclass(jin):
        head = jin[:, SMPL_HEAD]; sh = jin[:, SMPL_SHOULDERS].all(1); hips = jin[:, SMPL_HIPS].all(1); ank = jin[:, SMPL_ANKLES].all(1)
        c = np.full(len(jin), "whole body", object); c[~head] = "head cut"; c[~sh] = "shoulders cut"; c[~hips] = "hips cut (legs only)"; c[~ank] = "feet cut"; c[~jin.any(1)] = "out of view"
        return c
    classes = {v: vclass(in_frame[v]) for v in views}
    # ---- errors
    err = {v: geodesic(est[v], gt_body) for v in views}                                       # (T,21)
    A, B = views
    # visibility weights: the SMPL-X body joint j is SMPL joint j+1; a joint's rotation is observed when the joint and
    # the child segment it moves are in frame (SMPL kinematic children)
    SMPL_CHILD = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 10: 10, 11: 11, 12: 15, 13: 16, 14: 17, 15: 15, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23}
    def vis_w(v):
        # 1 when the joint and the segment it moves are in frame, 0.1 when the person is in frame but that joint is
        # not (so a view that sees the joint outweighs one that does not by 10:1, and two views that both miss it
        # fall back to an even average), 0 when the frame was bridged
        jin = in_frame[v]
        w = np.stack([(jin[:, j + 1] & jin[:, SMPL_CHILD[j + 1]]).astype(float) for j in range(21)], 1)
        return np.maximum(w, 0.1) * visible[v][:, None]
    w_vis = {v: vis_w(v) for v in views}
    fused = {"mean": chordal_mean([est[A], est[B]], np.ones((2, len(frames), 21))),
             "confidence-weighted": chordal_mean([est[A], est[B]], np.stack([w_conf[A], w_conf[B]])),
             "visibility-weighted": chordal_mean([est[A], est[B]], np.stack([w_vis[A], w_vis[B]]))}
    def segment_rule(wa, wb, ramp_frames: int = 0, extra=None):
        """Per joint and frame: the view(s) whose flag is set, averaged when both; NaN (no output) when neither.
        With ramp_frames > 0 the weights cross-fade over that many frames at the edges of each usable segment;
        extra = (sa, sb) multiplies each view's weights by a per-frame factor (the view's share of the fused root)."""
        fa, fb = (ramp_inside(wa, ramp_frames), ramp_inside(wb, ramp_frames)) if ramp_frames else (wa.astype(float), wb.astype(float))
        if extra is not None:
            fa, fb = fa * extra[0][:, None], fb * extra[1][:, None]
        out = chordal_mean([est[A], est[B]], np.stack([fa, fb]))
        none = ~(wa | wb); out[none] = np.nan
        return out
    hard = {v: (w_vis[v] >= 1.0) for v in views}                                             # truth-based usable flags
    fused["segment rule (truth visibility)"] = segment_rule(hard[A], hard[B])
    fused["segment rule (run-time visibility)"] = segment_rule(in_frame_rt[A], in_frame_rt[B])
    fused["segment rule, cross-faded (run-time visibility)"] = segment_rule(in_frame_rt[A], in_frame_rt[B], ramp_frames=15)
    fused["segment rule, cross-faded + smoothed 5 fr"] = smooth_rotations(fused["segment rule, cross-faded (run-time visibility)"], 5)
    fused["segment rule, cross-faded + smoothed 9 fr"] = smooth_rotations(fused["segment rule, cross-faded (run-time visibility)"], 9)
    # the fused root and each view's share of it (fuse_root); the share also weights the view's body joints
    yaw = {v: fit_yaw(root[v], gt_root, visible[v]) for v in views}
    root_al = {v: (R.from_euler("y", yaw[v], degrees=True) * R.from_rotvec(root[v])).as_rotvec() for v in views}
    root_f, share = fuse_root([root_al[A], root_al[B]], [in_frame_rt[A], in_frame_rt[B]], [visible[A] | True, visible[B] | True], ramp=15)
    fused["segment rule, cross-faded x root share + smoothed 5 fr"] = smooth_rotations(segment_rule(in_frame_rt[A], in_frame_rt[B], ramp_frames=15, extra=(share[0], share[1])), 5)
    pick = np.where((err[A] <= err[B])[:, :, None], est[A], est[B]); fused["oracle pick"] = pick
    err.update({k: geodesic(v_, gt_body) for k, v_ in fused.items()})
    root_err = {v: geodesic(root_al[v], gt_root) for v in views}; root_err["fused"] = geodesic(root_f, gt_root)
    root_step = {k: step_stats(x[:, None, :]) for k, x in (list(root_al.items()) + [("fused", root_f)])}
    # root error decomposed (yaw / pitch / roll about the truth's axes) and the hip joints' error, both by what the
    # along camera saw: a lifter that pitches the root wrong has hips that compensate for it
    root_rpy = {k: (R.from_rotvec(gt_root).inv() * R.from_rotvec(x)).as_euler("YXZ", degrees=True) for k, x in (list(root_al.items()) + [("fused", root_f)])}
    root_by_class = {c: {k: {"yaw": float(np.median(e[classes[A] == c, 0])), "pitch": float(np.median(e[classes[A] == c, 1])), "roll": float(np.median(e[classes[A] == c, 2]))}
                         for k, e in root_rpy.items()} for c in sorted(set(classes[A]))}
    hips_by_class = {c: {k: float(np.nanmedian(err[k][classes[A] == c][:, :2])) for k in list(views) + ["segment rule, cross-faded + smoothed 5 fr", "segment rule, cross-faded x root share + smoothed 5 fr"]} for c in sorted(set(classes[A]))}
    # ---- knee flexion, the gait-relevant scalar
    def knee(aa): return np.degrees(np.linalg.norm(aa[:, [3, 4]], axis=2))
    kg = knee(gt_body); knee_rmse = {k: float(np.sqrt(np.nanmean((knee(x) - kg) ** 2))) for k, x in list(est.items()) + list(fused.items())}
    # ---- tables
    def row(e, m=None):
        m = np.ones(len(frames), bool) if m is None else m; em = e[m]; legs = em[:, LEGS]
        with np.errstate(all="ignore"):
            return {"n_frames": int(m.sum()), "legs": float(np.nanmedian(legs)), "arms": float(np.nanmedian(em[:, ARMS])), "spine": float(np.nanmedian(em[:, SPINE])), "all": float(np.nanmedian(em)),
                    "legs_p90": float(np.nanpercentile(legs, 90)), "frac_legs_over_20": float(np.nanmean(legs > 20)) if np.isfinite(legs).any() else float("nan"),
                    "coverage": float(np.isfinite(em).mean())}
    steps = {k: step_stats(x) for k, x in list(est.items()) + list(fused.items())}
    res = {"n_frames": int(len(frames)), "views": views, "yaw_offset_deg": yaw,
           "overall": {k: row(e) for k, e in err.items()}, "knee_flexion_rmse_deg": knee_rmse, "continuity": steps,
           "root_orientation_median_deg": {**{v: float(np.median(root_err[v][visible[v]])) for v in views}, "fused": float(np.median(root_err["fused"]))},
           "root_step_p99_deg": {k: v_["p99_deg"] for k, v_ in root_step.items()}, "root_share_mean": {v: float(share[i].mean()) for i, v in enumerate(views)},
           "root_error_signed_by_along_visibility": root_by_class, "hip_error_by_along_visibility": hips_by_class,
           "by_visibility": {v: {c: row(err[v], classes[v] == c) for c in sorted(set(classes[v]))} for v in views},
           "fused_by_along_visibility": {c: {k: row(err[k], classes[A] == c) for k in fused} for c in sorted(set(classes[A]))},
           "per_joint_median": {k: dict(zip(SMPLX_BODY_JOINTS, np.median(e, 0).round(2).tolist())) for k, e in err.items()}}
    (out / "two_view_fusion.json").write_text(json.dumps(res, indent=1))
    # ---- markdown
    L = [f"# Two compromised views, one pose: HC901\n", f"{len(frames)} frames with the truth and both views lifted. Errors are geodesic angles (deg) between GVHMR's joint rotation and the rendered truth; medians over frames.\n",
         "| estimate | output coverage | legs | legs p90 | legs > 20 deg | arms | spine | all 21 | knee-flexion RMSE | frame step p99 | steps > 20 deg |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in list(views) + list(fused):
        r = res["overall"][k]; st = steps[k]
        L.append(f"| {k} | {r['coverage']:.0%} | {r['legs']:.1f} | {r['legs_p90']:.1f} | {r['frac_legs_over_20']:.0%} | {r['arms']:.1f} | {r['spine']:.1f} | {r['all']:.1f} | {knee_rmse[k]:.1f} | {st['p99_deg']:.1f} | {st['frac_over_20']:.2%} |")
    L.append("\nErrors of the segment rules are over the joints they emit; coverage is the share of (frame, joint) cells with an output. Both views and the other fusions emit every cell. "
             "Continuity: the largest per-joint rotation step between consecutive emitted frames, its 99th percentile and the share of steps over 20 deg.")
    rm = res["root_orientation_median_deg"]; rs = res["root_step_p99_deg"]
    L.append(f"\nRoot orientation after a constant heading alignment of {yaw[A]:+.0f} deg ({A}) and {yaw[B]:+.0f} deg ({B}): median error {rm[A]:.1f} / {rm[B]:.1f} deg "
             f"(frame step p99 {rs[A]:.1f} / {rs[B]:.1f}); the fused root (weighted by each view's ramped presence and the square of its usable fraction, sharpened towards the heavier view when the two disagree) "
             f"{rm['fused']:.1f} deg (step p99 {rs['fused']:.1f}); mean share of the {A} view {res['root_share_mean'][A]:.2f}.\n")
    L += ["\nRoot error decomposed (signed medians about the truth's axes, yaw / pitch / roll) and the hip joints' error, by what the along camera saw:\n",
          "| along camera saw | n | " + " | ".join(f"root {k}" for k in root_rpy) + f" | hips {A} | hips {B} | hips merge (no share) | hips merge (root share) |", "|---|---|" + "---|" * (len(root_rpy) + 4)]
    for c in root_by_class:
        n_c = int((classes[A] == c).sum()); hb = hips_by_class[c]
        L.append(f"| {c} | {n_c} | " + " | ".join(f"{root_by_class[c][k]['yaw']:+.0f} / {root_by_class[c][k]['pitch']:+.0f} / {root_by_class[c][k]['roll']:+.0f}" for k in root_rpy)
                 + f" | {hb[A]:.1f} | {hb[B]:.1f} | {hb['segment rule, cross-faded + smoothed 5 fr']:.1f} | {hb['segment rule, cross-faded x root share + smoothed 5 fr']:.1f} |")
    for v in views:
        L += [f"\n## {v} view by what the camera saw (truth)\n", "| frames | n | legs | arms | spine | all |", "|---|---|---|---|---|---|"]
        for c, r in res["by_visibility"][v].items():
            L.append(f"| {c} | {r['n_frames']} | {r['legs']:.1f} | {r['arms']:.1f} | {r['spine']:.1f} | {r['all']:.1f} |")
    fk = list(fused)
    L += [f"\n## Fused estimates by what the {A} view saw (legs / all 21; segment rules: over emitted joints, with coverage)\n", "| frames | n | " + " | ".join(fk) + " |", "|---|---|" + "---|" * len(fk)]
    for c, d in res["fused_by_along_visibility"].items():
        cells = [f"{d[k]['legs']:.1f} / {d[k]['all']:.1f}" + (f" ({d[k]['coverage']:.0%})" if k.startswith("segment") else "") for k in fk]
        L.append(f"| {c} | {d[fk[0]]['n_frames']} | " + " | ".join(cells) + " |")
    (out / "two_view_fusion.md").write_text("\n".join(L) + "\n")
    # ---- figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={"height_ratios": [1, 1.2]})
        x = np.arange(21); wdt = 0.16
        names = list(views) + ["mean", "segment rule (run-time visibility)", "segment rule, cross-faded x root share + smoothed 5 fr", "oracle pick"]; wdt = 0.8 / len(names)
        for i, k in enumerate(names):
            ax[0].bar(x + (i - len(names) / 2 + 0.5) * wdt, np.nanmedian(err[k], 0), wdt, label=k)
        ax[0].set_xticks(x); ax[0].set_xticklabels(SMPLX_BODY_JOINTS, rotation=60, ha="right", fontsize=8); ax[0].set_ylabel("median error vs truth (deg)"); ax[0].legend(fontsize=8, ncol=5)
        ax[0].set_title("A. Per-joint error of each view and of the fused estimates", loc="left", fontsize=10.5)
        t = gt["imu_t_s"][sel_gt]
        for k, c in ((A, "tab:blue"), (B, "tab:orange"), ("segment rule, cross-faded x root share + smoothed 5 fr", "tab:green")):
            with np.errstate(all="ignore"):
                ax[1].plot(t, np.nanmedian(err[k][:, LEGS], 1), color=c, lw=1.0 if k in views else 1.6, label=f"{k}: median leg-joint error" + ("" if k in views else " (gaps = no usable view)"))
        cl = classes[A]
        for lab, col in (("hips cut (legs only)", "0.75"), ("shoulders cut", "0.85"), ("head cut", "0.92")):
            m = cl == lab
            if m.any():
                ax[1].fill_between(t, 0, 60, where=m, color=col, step="mid", lw=0, label=f"{A} view: {lab}")
        for lab, col in (("out of view", "mistyrose"),):
            m = classes[B] == lab
            if m.any():
                ax[1].fill_between(t, 0, 60, where=m, color=col, step="mid", lw=0, alpha=0.8, label=f"{B} view: {lab}")
        ax[1].set_ylim(0, 60); ax[1].set_xlabel("IMU time (s)"); ax[1].set_ylabel("leg-joint error (deg)"); ax[1].legend(fontsize=7.5, ncol=3, loc="upper right")
        ax[1].set_title("B. Leg-joint error over the walk, with what each camera could see", loc="left", fontsize=10.5)
        plt.tight_layout(); plt.savefig(out / "two_view_fusion.png", dpi=130); plt.close(fig)
    except Exception as e:   # matplotlib is optional
        print("figure skipped:", e)
    if not a.quiet:
        print("\n".join(L)); print(f"\n-> {out}")
    return res


if __name__ == "__main__":
    main()
