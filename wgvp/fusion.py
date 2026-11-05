"""Rotation-space fusion of two lifted views: the library behind the merge stage (`wgvp.merge`) and the analysis
tool `tools/two_view_fusion.py`. Everything here is run-time: no ground truth is involved.

  usable_joints_rt   which SMPL-X body joints a view can vouch for in a frame (its 2D keypoints confident and inside the panel)
  ramp_inside        cross-fade weights at the edges of usable runs
  chordal_mean       weighted rotation average per (frame, joint)
  fuse_root          the merged root orientation and each view's share of it (continuous in every term)
  smooth_rotations   short moving average in rotation space that never crosses a gap
  export_mask        the frames an export keeps
  step_stats, geodesic, fit_yaw   measurements
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

LEGS = [0, 1, 3, 4, 6, 7]; ARMS = [12, 13, 15, 16, 17, 18, 19, 20]; SPINE = [2, 5, 8, 11, 14]
# SMPL-X body joint -> COCO-17 keypoints whose confidence says how well this joint's rotation is observed
# (the joint itself and the child segment it moves)
JOINT_KP = {0: [11, 13], 1: [12, 14], 2: [11, 12, 5, 6], 3: [13, 15], 4: [14, 16], 5: [5, 6, 11, 12], 6: [15], 7: [16], 8: [5, 6], 9: [15], 10: [16],
            11: [5, 6], 12: [5], 13: [6], 14: [5, 6], 15: [5, 7], 16: [6, 8], 17: [7, 9], 18: [8, 10], 19: [9], 20: [10]}


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
    m = np.asarray(mask, bool); T, J = m.shape; w = np.zeros((T, J), float); k = max(int(k), 1)
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
              soft_lo: float = 30.0, soft_hi: float = 60.0, n_joints: int = 21, return_raw: bool = False):
    """Root orientation of the merge and each view's share of it. Inputs per view, all in one world frame (per-view
    heading offsets already removed): roots (T, 3), usable (T, J) bool, lifted (T,) bool. Returns the fused root
    (T, 3) axis-angle, NaN where no view lifted the frame, and the normalised weights (V, T); with return_raw also
    the weights before normalisation (V, T), each in [0, 1].

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
    return (out, share, w) if return_raw else (out, share)


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(x, float)
    c = np.cumsum(np.r_[0.0, x]); h = k // 2; i = np.arange(len(x)); a_ = np.clip(i - h, 0, len(x)); b_ = np.clip(i + h + 1, 0, len(x))
    return (c[b_] - c[a_]) / (b_ - a_)


def moving_average_in_runs(x: np.ndarray, k: int) -> np.ndarray:
    """(T,) or (T, D) with NaN gaps -> moving average over k frames inside each run of finite rows; gaps stay NaN."""
    x = np.asarray(x, float); one = x.ndim == 1; X = x[:, None] if one else x; out = np.full_like(X, np.nan)
    ok = np.isfinite(X).all(1); edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        for d in range(X.shape[1]):
            out[a:b, d] = _moving_average(X[a:b, d], k)
    return out[:, 0] if one else out


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
    sh = a.shape[:-1]; a2 = a.reshape(-1, 3); b2 = b.reshape(-1, 3); ok = np.isfinite(a2).all(1) & np.isfinite(b2).all(1); out = np.full(len(a2), np.nan)
    out[ok] = np.degrees((R.from_rotvec(a2[ok]).inv() * R.from_rotvec(b2[ok])).magnitude())
    return out.reshape(sh)


def chordal_mean(aa_list: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    """Weighted rotation average per (frame, joint): aa_list = [(T,J,3)] per view, w (V,T,J). Quaternion method
    (largest eigenvector of the weighted outer-product sum), exact for two rotations."""
    V = len(aa_list); T, J = aa_list[0].shape[:2]
    Q = np.stack([R.from_rotvec(np.nan_to_num(a.reshape(-1, 3))).as_quat().reshape(T, J, 4) for a in aa_list])     # (V,T,J,4)
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


# SMPL-X body joints 0-21 of the neutral mean shape, rest pose, relative to the pelvis (metres; from the SMPL-X neutral
# model: smplx.create(..., model_type="smplx").joints with zero pose and shape), and their parents. Enough for a
# model-free forward kinematics of the merged pose when only joint POSITIONS of the mean body are needed (the feet
# for the walkway calibration); the real body shape moves them by a few centimetres.
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SMPLX_REST_JOINTS = np.array([[0.0, 0.0, 0.0], [0.0582, -0.0928, -0.026], [-0.0633, -0.1039, -0.0213], [-0.0028, 0.1099, -0.0276], [0.1129, -0.4715, -0.0354],
                              [-0.1075, -0.4663, -0.0381], [0.0067, 0.2417, -0.0336], [0.0694, -0.8746, -0.0673], [-0.0921, -0.877, -0.0583], [-0.0046, 0.294, -0.0051],
                              [0.1167, -0.9326, 0.0509], [-0.1309, -0.9353, 0.0608], [-0.0168, 0.4591, -0.0367], [0.0417, 0.3789, -0.0123], [-0.0523, 0.3783, -0.0185],
                              [0.008, 0.6196, -0.016], [0.161, 0.4367, -0.0278], [-0.1549, 0.4318, -0.0312], [0.4151, 0.3645, -0.0703], [-0.4261, 0.3953, -0.0576],
                              [0.6671, 0.3877, -0.0727], [-0.6753, 0.3908, -0.073]])
L_FOOT, R_FOOT, L_ANKLE, R_ANKLE = 10, 11, 7, 8


def fk_joints(root_aa: np.ndarray, body_aa: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """World positions (T, 22, 3) of the SMPL-X body joints of the mean shape for root (T,3), body pose (T,21,3) and
    translation (T,3); the pelvis sits at the translation. Rows with a NaN root or translation are NaN; a NaN body
    joint is taken as identity (its children then hang in the rest direction of their parent)."""
    T = root_aa.shape[0]; out = np.full((T, 22, 3), np.nan)
    ok = np.isfinite(root_aa).all(1) & np.isfinite(transl).all(1)
    if not ok.any():
        return out
    Rg = [None] * 22; pos = [None] * 22
    with np.errstate(all="ignore"):
        Rg[0] = R.from_rotvec(root_aa[ok]); pos[0] = transl[ok].astype(float)
        for j in range(1, 22):
            par = SMPLX_PARENTS[j]; loc = np.nan_to_num(body_aa[ok, j - 1])
            pos[j] = pos[par] + Rg[par].apply(SMPLX_REST_JOINTS[j] - SMPLX_REST_JOINTS[par]); Rg[j] = Rg[par] * R.from_rotvec(loc)
    out[ok] = np.stack(pos, axis=1)
    return out


def heading_yaw_deg(root_aa: np.ndarray) -> np.ndarray:
    """Heading of a root orientation (T, 3) axis-angle in a y-up world: the yaw of the pelvis forward axis (SMPL
    faces +z in its rest pose) projected on the floor, degrees in (-180, 180]; NaN rows stay NaN."""
    root_aa = np.asarray(root_aa, float); ok = np.isfinite(root_aa).all(1); out = np.full(len(root_aa), np.nan)
    if ok.any():
        f = R.from_rotvec(root_aa[ok]).apply(np.array([0.0, 0.0, 1.0]))
        out[ok] = np.degrees(np.arctan2(f[:, 0], f[:, 2]))
    return out


def circular_median_deg(d: np.ndarray) -> float:
    """Median of angles (deg) that is insensitive to the +-180 wrap: centre on the circular mean, take the median
    of the wrapped deviations."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    c = np.degrees(np.arctan2(np.sin(np.radians(d)).mean(), np.cos(np.radians(d)).mean()))
    dev = (d - c + 180.0) % 360.0 - 180.0
    return float((c + np.median(dev) + 180.0) % 360.0 - 180.0)


def fit_yaw(aa_est: np.ndarray, aa_gt: np.ndarray, mask: np.ndarray) -> float:
    """Yaw (deg, about world y) that best maps the estimated root orientations onto the truth (analysis only)."""
    best = (1e9, 0.0)
    for th in np.arange(-180, 180, 1.0):
        Ry = R.from_euler("y", th, degrees=True)
        e = np.degrees((Ry * R.from_rotvec(aa_est[mask])).inv().__mul__(R.from_rotvec(aa_gt[mask])).magnitude()).mean()
        if e < best[0]:
            best = (e, th)
    return best[1]
