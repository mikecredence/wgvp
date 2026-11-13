"""Merge stage: the two views' GVHMR outputs become one pose per frame on the IMU timeline, written as the training
target `merged_smpl.npz` (+ `merge.json`) beside `reconcile.json`. Runs on the CPU after the GPU stage, with
run-time information only; the body IMUs never enter it (the recording CSV is used for the sync and the subject
identity, nothing else), so the file can supervise a sensor-to-pose model without circularity.

Per recording:
  1. Common timeline: both views come from one tiled video, so they share frame indices; the merge covers every
     frame some view lifted. Per view and frame, the joints it can vouch for (`usable`): ViTPose keypoints
     confident and inside the panel, frame not bridged.
  2. One world frame: each view's GVHMR world is gravity-aligned but has its own heading and origin. The view with
     more usable joints is the reference; every other view is turned onto it by one constant yaw (the circular
     median of the heading difference of the two roots) and one constant offset (median translation difference),
     fitted on frames where both views see most of the body. The residuals are reported.
  3. Root: a view's root that flips for an instant and comes back is interpolated across first
     (`repair_root_glitches`); then `fuse_root` (presence-ramped, usable-fraction-weighted, sharpened when the
     views disagree, the arbitration itself smoothed over the cross-fade time) and the same 5-frame moving
     average the body gets. The pelvis path is fused from the views' per-frame STEPS with the same shares and
     summed (`fusion.fuse_path`), then smoothed over 5 frames: each view's horizontal step counts in full across
     its camera's line of sight and PATH_ALONG_SIGHT of that along it, where it is a change of monocular depth
     (the lines of sight come from the camera-frame files, `sight_line`). A mean of the views' positions would
     slide whenever the shares change: two monocular paths sit up to metres apart while the walker moves.
  4. Body: per joint, the view(s) that can vouch for it, cross-faded over 0.5 s at segment edges, each view
     weighted by its share of the root; nothing where no view saw the joint; smoothed over 5 frames.
  5. Export rule: a frame is kept when every leg joint AND every torso joint (spine, neck, head) is output, in
     runs of at least 0.5 s (`export_ok`). A frame with a leg or the trunk unseen is dropped whole; a frame with
     only an arm unseen is kept, the arm's joints NaN with weight 0.
  6. Heading-relative root for learning from body-worn sensors: the heading yaw is taken out of the root
     (`global_orient_heading_rel`), and the root velocity is expressed in the heading frame; a sensor model is
     then asked for tilt, turning rate and velocity, never for the room's yaw.
  7. One shape per recording (`betas`): the median of GVHMR's per-frame betas over whole-body frames of all views.
  8. The walkway (when the recording CSV carries it: per sample the loaded cells of the pressure mat with foot
     labels, see imu_ref.parse_walkway): (a) each view's positions are scaled about the path's own footfall-time
     centre by a factor fitted, with the mat's yaw and offset in that view's world, to the footfalls against the
     landing foot (forward kinematics of the mean body), accepted under 0.5 m of residual (`transl_scaled`,
     `walkway_scale`; of little use, GVHMR's path is distorted rather than scaled); (b) the pelvis ON THE MAT,
     `transl_mat`: per stance the footprint's centre minus the pose's foot-to-pelvis offset turned into the mat
     frame, both feet in double support, ramped hand-overs, short gaps bridged; a measured position with no depth
     estimate in it (`transl_mat_video` is the video path in the mat frame, `transl_from_walkway` the mat path in
     the world; the heading-frame velocity uses it where it exists).
  8b. A view that is the mirror image of the reference (a camera's picture flipped by its software) is detected
     from the knees in the swing phase (as lifted vs sides swapped) and reflected back before merging
     (`view_mirrored`, fusion.mirror_body).
  9. Weights for a loss, each in [0, 1] and meaning "no evidence against this value": `weight` (T, 21) = the
     summed view weights that produced the joint (one confident view or two agreeing ones: 1) times an agreement
     factor 1 / (1 + ((d - 5 deg)+ / 20 deg)^2) where two views both vouched for the joint and disagreed by d, d
     scaled by twice the smaller view's share of the root, which is how much of the dissenting view the merged
     joint contains (the whole disagreement at an even split, none of it when the root rule set that view aside:
     a view in the wrong gravity frame that also solved the legs wrongly is not evidence against the other view's
     legs); zero outside the export. `root_weight` = the summed raw root weights (presence x usable fraction squared, <= 1);
     `transl_weight` = root_weight where the path exists. The distance between the two views' aligned pelvis
     positions is recorded (`transl_disagree_m`) and is no factor of the weight: the path is fused from steps,
     and on the rendered samples that distance tracks neither the velocity's error nor the path's.
     Translation from one far camera is the weakest field (see the README).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .fusion import (MIRROR_SWAP, mirror_body, mirror_check, mirror_root, mirror_transl, fuse_path,
                     ARMS, FLEX, L_ANKLE, L_FOOT, LEGS, R_ANKLE, R_FOOT, SPINE, TORSO_LEGS, chordal_mean, circular_median_deg, export_mask, fk_joints, flexion_two_band, fuse_root, geodesic, heading_frame, heading_rel, heading_yaw_deg,
                     moving_average_in_runs, ramp_inside, rate_in_runs, repair_root_glitches, seen_frames, side_on_score, sight_line_world, smooth_rotations, step_stats, usable_joints_rt)
from .reconcile import SMPLX_BODY_JOINTS

RAMP_S = 0.5           # cross-fade at the edges of usable segments and of lifted runs
MIN_RUN_S = 0.5        # an export keeps runs of at least this length
SMOOTH_FRAMES = 5      # moving average of the merged rotations / translation
ROOT_SMOOTH_FRAMES = 5 # the same for the merged root orientation
FAST_TURN_S = 0.25; FAST_TURN_DEG = 90.0   # QC: the merged root turning more than this within this long is no body's turn (360 deg/s; a quick turn on the spot peaks near 200)
AGREE_FREE_DEG = 5.0   # two views agreeing to within this carry full weight
AGREE_SCALE_DEG = 20.0 # the agreement factor is 0.5 at AGREE_FREE_DEG + this
BETAS_DIFF_WARN = 3.0  # |betas_a - betas_b| above which the two views may not hold the same person
ALIGN_MIN_FRAMES = 30
MOVING_M_S = 0.25       # the pelvis moves faster than this: a frame of walking, for the views' position residual
SLOPE_MIN_REACH_M = 2.0  # a view's floor slope is reported for a walk at least this long
WALKWAY_MIN_ROWS = 8   # footfalls (stances) needed before the mat's pose and the translation scale are fitted
MAT_GAP_S = 0.5        # the walkway-derived path is interpolated across gaps up to this long (a foot off the mat's edge)
MAT_RAMP_S = 0.12      # a stance's say in the walkway-derived path ramps in and out over this long

# hips' and knees' flexion: the posture (slower than a Gaussian of this sigma) is the weighted mean of the views, the
# stride's swing comes from a second mean that also weights each view by how side-on it sees the walker
FLEX_SLOW_S = 0.75
MAT_VEL_S = 1.5            # on the mat the root velocity's mean over this long (a stride and a half) is the mat path's; what is faster is the fused video path's
PATH_ALONG_SIGHT = 0.05   # weight of a view's pelvis step along its camera's line of sight (a change of monocular depth), against 1 for the step across it
SIDE_ON_FLOOR = 0.05      # weight of a view that sees the walker head-on, against 1.05 for one that sees them from the side
FIELDS_DOC = (
    "frame_idx: frame of the original combined video; imu_t_s: recording time (pts/1000 - offset); "
    "body_pose (T,63): merged SMPL-X body pose, axis-angle, NaN for joints no view saw; "
    "global_orient (T,3) / transl (T,3): merged root in the reference view's GVHMR world (y up, arbitrary heading and origin); the pelvis path is the views' "
    "per-frame steps fused and summed, each view's step counted across its camera's line of sight, set on the views' mean position per run of frames; "
    "heading_yaw_deg (T): yaw of the pelvis forward axis in that world, unwrapped inside runs; heading_yaw_rate_deg_s (T): its rate; "
    "global_orient_heading_rel (T,3): the root with the heading yaw removed (tilt only: what body-worn sensors can know); "
    "root_vel_heading_rel_m_s (T,3): root velocity in the heading frame (x lateral, y up, z forward), from the fused video path; where the pelvis has a mat "
    "path (transl_from_walkway) the velocity's mean over 1.5 s is the mat path's and only the faster part the video's; "
    "transl_scaled (T,3): the translation after the walkway scale calibration (equal to transl when no walkway data); "
    "transl_mat (T,3): the pelvis ON THE MAT, metres in the mat frame (X along the mat, height from the video, Y across): the walkway's footprint "
    "under each foot in stance plus the video pose's foot-to-pelvis offset, NaN off the mat or without walkway data (a measured position, no depth estimate in it); "
    "transl_mat_video (T,3): the video's own path turned into the mat frame, for comparison; transl_from_walkway (T,3): transl_mat back in the reference view's world; "
    "walkway_scale (V): each view's depth-scale factor (1 = unchanged); transl_calibrated: whether the walkway scale fit ran; "
    "view_mirrored (V): views found to be mirror images of the reference (left/right swapped by the camera software) and reflected back before merging; "
    "betas (10): one body shape for the recording; "
    "export_ok (T): frames the export keeps (every leg and torso joint output, i.e. hips, knees, ankles, feet, spine, neck, head; runs >= 0.5 s; an arm may be missing); "
    "weight (T,21): per-joint loss weight in [0,1] (0 outside the export or where the joint is NaN); root_weight (T): the same for the root orientation; "
    "transl_weight (T): for the translation / velocity (root_weight where the path exists); transl_disagree_m (T): distance between the two views' aligned pelvis "
    "positions (NaN with one view; recorded only: two monocular paths drift apart by decimetres to metres, and the merged path is fused from their steps); "
    "source (T,21): bit i set when view i (in `views` order) contributed to the joint; usable (V,T,21): which joints each view could vouch for; "
    "share (V,T): each view's share of the root per frame; side_on (V,T): how side-on each view saw the walker (1 = from the side, 0 = walking straight at "
    "the camera or away from it, NaN where not lifted): the stride's swing of the hips and knees is taken mostly from the view with the higher value; sight_line (V,T,3): each camera's line of sight to the pelvis as a unit vector in the merged world (NaN where not lifted or without the camera-frame file): a view's pelvis step counts in full across it and little along it; views, reference_view, view_alignment_yaw_deg (V), view_alignment_offset_m (V,3): "
    "how each view was turned onto the reference view's world; fps: frames per second of the merged timeline")


def moving_frames(transl: np.ndarray, fps: float, min_speed: float = MOVING_M_S, window_s: float = 0.5) -> np.ndarray:
    """Frames in which the pelvis moves: its displacement over `window_s` (centred) is faster than `min_speed`."""
    T = len(transl); k = max(1, int(round(0.5 * window_s * fps))); sp = np.zeros(T)
    if T > 2 * k:
        with np.errstate(all="ignore"):
            sp[k:T - k] = np.nan_to_num(np.linalg.norm(transl[2 * k:] - transl[:T - 2 * k], axis=1), nan=0.0) / (2 * k / fps)
    return sp > min_speed


def path_slope_deg(transl: np.ndarray, fps: float, min_reach_m: float = SLOPE_MIN_REACH_M) -> float | None:
    """Slope (deg) of the pelvis height along the walk's main horizontal direction, over the frames of walking.
    A walker on a level floor gives 0. A lift whose assumed field of view is wrong, from a camera that looks down
    or up at the walker, puts the path on a ramp instead: the walker stays upright and sinks or climbs with distance.
    None when the walk is shorter than `min_reach_m`. An indicator, not a measurement of the field of view: a camera
    held level gives 0 whatever the lens."""
    transl = np.asarray(transl, float); mv = moving_frames(transl, fps) & np.isfinite(transl).all(1)
    if mv.sum() < ALIGN_MIN_FRAMES:
        return None
    P = transl[mv]; xz = P[:, [0, 2]] - P[:, [0, 2]].mean(0)
    s = xz @ np.linalg.eigh(xz.T @ xz)[1][:, -1]
    if np.ptp(s) < min_reach_m:
        return None
    return float(np.degrees(np.arctan(abs(np.polyfit(s, P[:, 1], 1)[0]))))


def align_to_reference(root_ref, transl_ref, root_v, transl_v, mask, moving=None) -> dict:
    """One yaw (deg) and one offset (m) that turn view v's GVHMR world onto the reference view's, from frames in
    `mask` (both lifted and seeing most of the body). The root's residual is reported on the same frames. The pelvis
    position's is reported on those of them in which the walker moves (`moving`, when it leaves enough frames): the
    frames of a walker who stands still agree after any offset, and one who stands for half the recording would
    set the median alone."""
    n = int(mask.sum())
    if n == 0:
        return {"yaw_deg": 0.0, "offset_m": [0.0, 0.0, 0.0], "n_frames": 0, "root_residual_deg": None, "transl_residual_m": None}
    dyaw = heading_yaw_deg(root_ref[mask]) - heading_yaw_deg(root_v[mask])
    yaw = circular_median_deg(dyaw); Ry = R.from_euler("y", yaw, degrees=True)
    on = np.ones(n, bool) if moving is None else np.asarray(moving, bool)[mask]
    walking = bool(moving is not None and on.sum() >= ALIGN_MIN_FRAMES)
    with np.errstate(all="ignore"):                                  # (Accelerate on macOS raises spurious matmul warnings here)
        tv = Ry.apply(transl_v[mask]); off = np.median(transl_ref[mask] - tv, axis=0)
        rr = geodesic((Ry * R.from_rotvec(root_v[mask])).as_rotvec(), root_ref[mask])
        tr = np.linalg.norm(transl_ref[mask] - (tv + off), axis=1)
    return {"yaw_deg": float(yaw), "offset_m": [float(x) for x in off], "n_frames": n, "root_residual_deg": float(np.nanmedian(rr)),
            "transl_residual_m": float(np.nanmedian(tr[on] if walking else tr)), "transl_residual_frames": "moving" if walking else "all"}


def camera_centre_world(transl_world: np.ndarray, transl_cam: np.ndarray):
    """GVHMR's camera is static: world = R p_cam + c. Rigid (Kabsch) fit on the finite rows -> (c (3,), R (3,3), rms)."""
    ok = np.isfinite(transl_world).all(1) & np.isfinite(transl_cam).all(1)
    if ok.sum() < 10:
        return None
    P, Q = transl_cam[ok].astype(float), transl_world[ok].astype(float); pm, qm = P.mean(0), Q.mean(0)
    H = (P - pm).T @ (Q - qm); U, _, Vt = np.linalg.svd(H); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ D @ U.T; c = qm - Rm @ pm
    rms = float(np.sqrt(np.mean(np.sum((Q - (P @ Rm.T + c)) ** 2, axis=1))))
    return c, Rm, rms


def _interp_runs(t_src: np.ndarray, x_src: np.ndarray, t_q: np.ndarray, max_gap_s: float = 0.1) -> np.ndarray:
    """Linear interpolation of (T, D) rows that are finite, NaN where the query time is not inside a run of finite rows."""
    ok = np.isfinite(x_src).all(1); out = np.full((len(t_q), x_src.shape[1]), np.nan)
    if ok.sum() < 2:
        return out
    ts = t_src[ok]; xs = x_src[ok]
    for d in range(xs.shape[1]):
        out[:, d] = np.interp(t_q, ts, xs[:, d], left=np.nan, right=np.nan)
    i = np.clip(np.searchsorted(ts, t_q), 1, len(ts) - 1); gap = ts[i] - ts[i - 1]
    out[gap > max_gap_s] = np.nan
    return out


def _mat_to_world(th: float, o, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    cy, sy = np.cos(th), np.sin(th)
    return np.c_[cy * X + sy * Y + o[0], -sy * X + cy * Y + o[1]]


def fit_mat_pose(xz: np.ndarray, W: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Rigid 2D map (yaw, offset) taking walkway points W (N,2, metres on the mat) onto world positions xz (N,2):
    a 1-degree yaw grid with the offset in closed form, refined by robust least squares. Returns (yaw_rad, offset, rms)."""
    from scipy.optimize import least_squares
    best = None
    for th in np.radians(np.arange(-180.0, 180.0, 1.0)):
        o = xz.mean(0) - _mat_to_world(th, np.zeros(2), W[:, 0], W[:, 1]).mean(0); e = np.sum((_mat_to_world(th, o, W[:, 0], W[:, 1]) - xz) ** 2)
        if best is None or e < best[0]:
            best = (e, th, o)
    sol = least_squares(lambda q: (_mat_to_world(q[0], q[1:3], W[:, 0], W[:, 1]) - xz).ravel(), np.r_[best[1], best[2]], loss="soft_l1", f_scale=0.3)
    rms = float(np.sqrt(np.mean(np.sum((_mat_to_world(sol.x[0], sol.x[1:3], W[:, 0], W[:, 1]) - xz) ** 2, axis=1))))
    return float(sol.x[0]), sol.x[1:3].copy(), rms


def feet_positions(root_aa: np.ndarray, body_aa: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """(T, 2, 3): the left and right foot of the mean body (midpoint of ankle and foot joints) in the frame of `transl`."""
    J = fk_joints(root_aa, body_aa, transl)
    return np.stack([0.5 * (J[:, L_ANKLE] + J[:, L_FOOT]), 0.5 * (J[:, R_ANKLE] + J[:, R_FOOT])], axis=1)


def landed_foot_xz(feet_t: np.ndarray, foot_label: np.ndarray, W_world_guess=None) -> np.ndarray:
    """Pick per row the foot the walkway saw: the labelled one, else (label 0) the lower of the two (the stance foot
    is on the floor). feet_t (N, 2, 3) -> (N, 3)."""
    lab = np.asarray(foot_label, int); lower = np.argmin(feet_t[:, :, 1], axis=1)
    pick = np.where(lab == 1, 0, np.where(lab == 2, 1, lower))
    return feet_t[np.arange(len(feet_t)), pick]


def mat_pelvis_path(root_aa: np.ndarray, body_aa: np.ndarray, t_imu: np.ndarray, stances: dict, to_mat, fps: float):
    """The pelvis on the mat, (T, 2) metres in the mat frame: during each stance the walkway knows where that foot
    is, and the video pose knows where the pelvis is relative to that foot (forward kinematics of the mean body,
    turned into the mat frame by `to_mat`); both feet count during double support. Gaps shorter than MAT_GAP_S
    inside the mat's coverage are bridged linearly, the result smoothed like the rest. Returns (path, fraction of
    the frames covered, stances used)."""
    T = len(t_imu); F = feet_positions(root_aa, body_aa, np.zeros((T, 3)))                      # foot minus pelvis, world
    acc = np.zeros((T, 2)); cnt = np.zeros(T); used = 0
    for t0, t1, foot, xy in zip(stances["t0"], stances["t1"], stances["foot"], stances["xy"]):
        if foot not in (1, 2):
            continue
        if t1 - t0 < 0.05:                                                                        # a footfall instant (legacy rows): a nominal stance around it
            t0, t1 = t0 - 0.15, t1 + 0.15
        m = (t_imu >= t0) & (t_imu <= t1) & np.isfinite(F[:, foot - 1]).all(1)
        if not m.any():
            continue
        # each stance's say ramps in and out over MAT_RAMP_S, so the hand-over between feet (whose two estimates differ
        # by the pose error) does not step the path
        w = np.clip(np.minimum(t_imu[m] - t0, t1 - t_imu[m]) / MAT_RAMP_S + 1.0 / (2 * MAT_RAMP_S * max(fps, 1)), 0.0, 1.0)
        ox, oz = to_mat(F[m, foot - 1, 0], F[m, foot - 1, 2]); acc[m, 0] += w * (xy[0] - ox); acc[m, 1] += w * (xy[1] - oz); cnt[m] += w; used += 1
    with np.errstate(invalid="ignore"):
        path = np.where(cnt[:, None] > 1e-6, acc / np.maximum(cnt, 1e-6)[:, None], np.nan)
    okp = np.isfinite(path).all(1)
    if okp.sum() >= 2:                                                                            # bridge short gaps
        idx = np.flatnonzero(okp); gaps = np.flatnonzero(np.diff(idx) > 1)
        for g in gaps:
            a, b = idx[g], idx[g + 1]
            if t_imu[b] - t_imu[a] <= MAT_GAP_S:
                path[a:b + 1] = np.linspace(path[a], path[b], b - a + 1)
    path = moving_average_in_runs(path, SMOOTH_FRAMES, centred=True)
    return path, float(np.isfinite(path).all(1).mean()), used


def fit_view_depth_scale(root_aa: np.ndarray, body_aa: np.ndarray, transl_world: np.ndarray, t_view: np.ndarray, cam_c: np.ndarray,
                         ww_t: np.ndarray, ww_xy: np.ndarray, ww_foot: np.ndarray, max_rms_m: float = 0.5) -> dict | None:
    """One view on its own: the scale factor k of its world path about the path's own centre (p' = m + k (p - m),
    applied to the whole body) that, with the mat's yaw and offset in this view's world, best explains the walkway
    footfalls against the position of the foot that landed (forward kinematics of the mean body from this view's own
    pose). GVHMR's world path is integrated by its network, not projected from the camera (the camera-frame and
    world-frame translations are not a rigid pair: see `camera_fit_rms_m`), so the centre of the scaling is the path
    itself, not the camera. Robust least squares; a fit is accepted only when its residual is under `max_rms_m`.
    None without enough rows."""
    from scipy.optimize import least_squares
    F = feet_positions(root_aa, body_aa, transl_world)                                           # (T, 2, 3)
    Fq = np.stack([_interp_runs(t_view, F[:, i], ww_t) for i in range(2)], axis=1)              # (N, 2, 3)
    use = np.isfinite(Fq).all(axis=(1, 2))
    if use.sum() < WALKWAY_MIN_ROWS:
        return None
    lab = np.asarray(ww_foot[use], int).copy(); Fu = Fq[use]; W = ww_xy[use]
    cam_c = Fu.reshape(-1, 3).mean(0)                                                             # the centre of the scaling: the view's own footfall-time positions
    def pos_of(feet, k): return (cam_c + k * (feet - cam_c))[:, [0, 2]]
    def solve(lab_now, x_init):
        P = landed_foot_xz(Fu, lab_now)
        def resid(q): return (_mat_to_world(q[1], q[2:4], W[:, 0], W[:, 1]) - pos_of(P, np.exp(q[0]))).ravel()
        sol = least_squares(resid, x_init, loss="soft_l1", f_scale=0.15, max_nfev=2000)
        return sol, float(np.sqrt(np.mean(np.sum(resid(sol.x).reshape(-1, 2) ** 2, axis=1))))
    th0, o0, rms0 = fit_mat_pose(pos_of(landed_foot_xz(Fu, lab), 1.0), W); x0 = np.r_[0.0, th0, o0]
    sol, rms = solve(lab, x0)
    free = lab == 0
    for _ in range(3 if free.any() else 0):                      # unlabelled rows: re-associate each with the nearer foot under the current fit, refit
        m = _mat_to_world(sol.x[1], sol.x[2:4], W[:, 0], W[:, 1]); k_ = np.exp(sol.x[0])
        dl = np.linalg.norm(pos_of(Fu[:, 0], k_) - m, axis=1); dr = np.linalg.norm(pos_of(Fu[:, 1], k_) - m, axis=1)
        new = np.where(free, np.where(dl <= dr, 1, 2), lab)
        if np.array_equal(new, lab):
            break
        lab = new; sol, rms = solve(lab, sol.x)
    k = float(np.exp(sol.x[0]))
    return {"ok": bool(sol.success) and 0.5 < k < 2.0 and rms <= max_rms_m, "scale": k, "yaw_deg": float(np.degrees(sol.x[1])), "offset_m": [float(x) for x in sol.x[2:4]],
            "centre_m": [float(x) for x in cam_c], "rms_before_m": rms0, "rms_after_m": rms, "n_rows": int(use.sum()), "n_labelled": int((ww_foot[use] > 0).sum())}


def merge_recording(rec_dir: Path, params: dict | None = None, log=print, write: bool = True) -> dict | None:
    """Merge every lifted view under <deliver>/<recording>/<view>/ into merged_smpl.npz + merge.json. Returns the
    merge.json content (status PASS/WARN with flags), or None when no view has been lifted yet."""
    rec_dir = Path(rec_dir)
    views = sorted(p.name for p in rec_dir.iterdir() if p.is_dir() and (p / "smpl_global.npz").exists() and (p / "kp2d_vitpose.npz").exists())
    if not views:
        return None
    Z = {v: np.load(rec_dir / v / "smpl_global.npz") for v in views}; K = {v: np.load(rec_dir / v / "kp2d_vitpose.npz") for v in views}
    frames = np.array(sorted(set().union(*[set(Z[v]["frame_idx"].tolist()) for v in views])), np.int64); T = len(frames); V = len(views)
    fi = {int(f): i for i, f in enumerate(frames)}
    lifted = np.zeros((V, T), bool); visible = np.zeros((V, T), bool); usable = np.zeros((V, T, 21), bool)
    body = np.full((V, T, 21, 3), np.nan); root = np.full((V, T, 3), np.nan); transl = np.full((V, T, 3), np.nan); betas = np.full((V, T, 10), np.nan)
    imu_t = np.full((V, T), np.nan); fps = None; detected = np.zeros((V, T), bool); rest_held = np.zeros((V, T), bool); rows = []
    for i, v in enumerate(views):
        z, k = Z[v], K[v]; idx = np.array([fi[int(f)] for f in z["frame_idx"]]); rows.append(idx)
        # "visible" from here on: the subject is seen in the frame, detected or held at rest with keypoints that confirm it
        lifted[i, idx] = True; visible[i, idx] = seen_frames(z, k); detected[i, idx] = z["visible"].astype(bool)
        rest_held[i, idx] = z["rest_held"].astype(bool) if "rest_held" in z.files else False
        body[i, idx] = z["body_pose"].reshape(-1, 21, 3); root[i, idx] = z["global_orient"]; transl[i, idx] = z["transl"]; betas[i, idx] = z["betas"]
        imu_t[i, idx] = z["imu_t_s"]
        kp = np.full((T, 17, 3), np.nan); kp[idx] = k["kp2d"]
        usable[i] = usable_joints_rt(kp, k["view_box"] if "view_box" in k.files else None, visible[i], frame_idx=frames)
        if fps is None and "fps_source" in z.files:
            fps = float(z["fps_source"]) / float(z["frame_step"] if "frame_step" in z.files else 1)
    t_imu = np.nanmean(imu_t, axis=0)
    with np.errstate(all="ignore"):
        t_mismatch = float(np.nanmax(np.nanmax(imu_t, 0) - np.nanmin(imu_t, 0))) if V > 1 else 0.0
    if fps is None or not np.isfinite(fps) or fps <= 0:
        fps = float(1.0 / np.nanmedian(np.diff(t_imu)))
    ramp = max(1, int(round(RAMP_S * fps))); min_run = max(1, int(round(MIN_RUN_S * fps)))
    glitch = []                                                                                  # a view's root flipping for an instant and coming back: interpolated across
    for i, v in enumerate(views):
        root[i], fx = repair_root_glitches(root[i], fps); glitch.append(fx)
    frac = usable.sum(2) / 21.0                                                                  # (V, T)
    ref = int(np.argmax(usable.sum(axis=(1, 2)))); flags = []
    # ---- camera centres (each view's own world) and the walkway footfalls, if the recording has them
    cam_c = []; cam_rms = []          # diagnostic only: how far GVHMR's world path is from a rigid image of its camera-frame path
    side = np.full((V, T), np.nan)    # how side-on each view sees the walker (from the pose in its camera's frame)
    sight = np.full((V, T, 3), np.nan)   # each camera's line of sight to the pelvis, in the view's own world
    for i, v in enumerate(views):
        inc = rec_dir / v / "smpl_incam.npz"; fit = None
        if inc.exists():
            zc = np.load(inc); same = len(zc["transl"]) == len(Z[v]["transl"]); fit = camera_centre_world(Z[v]["transl"], zc["transl"]) if same else None
            if same:
                side[i, rows[i]] = side_on_score(zc["global_orient"], zc["transl"]); sight[i, rows[i]] = sight_line_world(Z[v]["global_orient"], zc["global_orient"], zc["transl"])
        cam_c.append(fit[0] if fit is not None else None); cam_rms.append(fit[2] if fit is not None else None)
    ww = None; stances = None; ww_info = {"rows": 0, "stances": 0, "format": "none"}; ref_npz = rec_dir.parent.parent / "work" / "imu_ref" / f"{rec_dir.name}_imu_ref.npz"
    if ref_npz.exists():
        zr = np.load(ref_npz, allow_pickle=False)
        if "walkway_stance_t0" in zr.files:
            stances = {"t0": zr["walkway_stance_t0"].astype(float), "t1": zr["walkway_stance_t1"].astype(float), "foot": zr["walkway_stance_foot"].astype(int),
                       "xy": np.c_[zr["walkway_stance_x_m"], zr["walkway_stance_y_m"]].astype(float)}
            ww_info = {"rows": int(len(zr["walkway_t"])), "stances": int(len(stances["t0"])), "format": str(zr["walkway_format"]) if "walkway_format" in zr.files else "cells"}
        elif "walkway_t" in zr.files and len(zr["walkway_t"]):                        # a reference written before stances existed
            stances = {"t0": zr["walkway_t"].astype(float), "t1": zr["walkway_t"].astype(float), "foot": zr["walkway_foot"].astype(int) if "walkway_foot" in zr.files else np.zeros(len(zr["walkway_t"]), int),
                       "xy": np.c_[zr["walkway_x_m"], zr["walkway_y_m"]].astype(float)}
            ww_info = {"rows": int(len(zr["walkway_t"])), "stances": int(len(zr["walkway_t"])), "format": "footfall_cm"}
        if stances is not None and len(stances["t0"]) >= WALKWAY_MIN_ROWS:
            ww = (0.5 * (stances["t0"] + stances["t1"]), stances["xy"], stances["foot"])     # one row per footfall, at mid-stance
    wfits = [None] * V; wscale = np.ones(V); transl_sc = transl.copy()           # per view, in its own world, scaled about its own path centre
    if ww is not None:
        for i, v in enumerate(views):
            wfits[i] = fit_view_depth_scale(root[i], body[i], transl[i], t_imu, None, ww[0], ww[1], ww[2])
            if wfits[i] and wfits[i]["ok"]:
                wscale[i] = wfits[i]["scale"]; li = lifted[i]; ctr = np.array(wfits[i]["centre_m"]); transl_sc[i, li] = ctr + wscale[i] * (transl[i, li] - ctr)
            elif wfits[i]:
                flags.append(f"{v}: walkway fit rejected (scale {wfits[i]['scale']:.2f}, residual {wfits[i]['rms_after_m']:.2f} m): its translation is left as lifted")
    calibrated = ww is not None and any(f is not None and f["ok"] for f in wfits)
    # ---- a view that is a mirror image of the reference (camera software mirroring the picture): its left is the
    # walker's right; reflect it back before anything is compared or fused
    mirrored = np.zeros(V, bool); mirror_info = {}
    for i, v in enumerate(views):
        if i == ref:
            continue
        both = lifted[ref] & lifted[i] & visible[ref] & visible[i] & usable[ref][:, LEGS].all(1) & usable[i][:, LEGS].all(1)
        chk = mirror_check(body[ref], body[i], both); mirror_info[v] = chk
        if chk["mirrored"]:
            mirrored[i] = True
            body[i] = mirror_body(body[i]); root[i] = mirror_root(root[i]); transl[i] = mirror_transl(transl[i]); transl_sc[i] = mirror_transl(transl_sc[i]); sight[i] = mirror_transl(sight[i])
            usable[i] = usable[i][:, MIRROR_SWAP]
            flags.append(f"{v} is a mirror image of {views[ref]} (its legs fit {chk['mirrored_deg']:.1f} deg with the sides swapped vs {chk['as_is_deg']:.1f} deg as lifted, "
                         f"{chk['n_frames']} frames): reflected back before merging; check that camera's mirror setting")
    # ---- one world frame
    align = []; root_al = root.copy(); transl_al = transl.copy(); transl_al_sc = transl_sc.copy(); sight_al = sight.copy(); moving = moving_frames(transl[ref], fps)
    for i, v in enumerate(views):
        if i == ref:
            align.append({"yaw_deg": 0.0, "offset_m": [0.0, 0.0, 0.0], "n_frames": int(lifted[i].sum()), "root_residual_deg": 0.0, "transl_residual_m": 0.0, "reference": True}); continue
        both = lifted[ref] & lifted[i] & visible[ref] & visible[i]
        mask = both & (frac[ref] >= 0.75) & (frac[i] >= 0.75)
        if mask.sum() < ALIGN_MIN_FRAMES:
            mask = both & (frac[ref] >= 0.5) & (frac[i] >= 0.5)
        if mask.sum() < ALIGN_MIN_FRAMES:
            mask = both
        a = align_to_reference(root[ref], transl[ref], root[i], transl[i], mask, moving); align.append(a)
        if a["n_frames"] < ALIGN_MIN_FRAMES:
            flags.append(f"{v}: aligned to {views[ref]} on only {a['n_frames']} shared frames (heading / origin uncertain)")
        if a["transl_residual_m"] is not None and a["transl_residual_m"] > 0.3:      # for the record: the merged path is fused from the views' steps, so this offset does not enter it
            flags.append(f"info: {v}: after alignment its pelvis position differs from {views[ref]} by {a['transl_residual_m']:.2f} m (median on shared frames{' in which the walker moves' if a.get('transl_residual_frames') == 'moving' else ''})")
        Ry = R.from_euler("y", a["yaw_deg"], degrees=True); li = lifted[i]
        with np.errstate(all="ignore"):
            root_al[i, li] = (Ry * R.from_rotvec(root[i, li])).as_rotvec(); transl_al[i, li] = Ry.apply(transl[i, li]) + np.array(a["offset_m"]); sight_al[i, li] = Ry.apply(sight[i, li])
            if calibrated:                                   # the scaled positions get their own offset onto the (scaled) reference view
                tv = Ry.apply(transl_sc[i, li]); mm = mask[li]
                off_sc = np.median(transl_sc[ref, li][mm] - tv[mm], axis=0) if mm.sum() >= 3 else np.array(a["offset_m"])
                transl_al_sc[i, li] = tv + off_sc; a["offset_scaled_m"] = [float(x) for x in off_sc]
    # ---- root, translation
    root_f, share, raw = fuse_root([root_al[i] for i in range(V)], [usable[i] for i in range(V)], [lifted[i] for i in range(V)], ramp=ramp, return_raw=True, fps=fps)
    root_f = smooth_rotations(root_f[:, None, :], ROOT_SMOOTH_FRAMES)[:, 0]                       # the same moving average the body gets
    for i, v in enumerate(views):
        a = align[i]
        if i != ref and a["root_residual_deg"] is not None and a["root_residual_deg"] > 20:
            msg = (f"{v}: its root differs from {views[ref]}'s by {a['root_residual_deg']:.0f} deg on shared frames (one of them is in the wrong gravity frame); "
                   f"the merge gave {v} a mean root share of {share[i].mean():.2f}")
            flags.append(msg if share[i].mean() > 0.35 else "info: " + msg)
    # the pelvis path: the views' steps fused (each across its camera's line of sight) and summed; a mean of their positions would slide with the shares
    path_rule = V > 1; sight_known = bool(np.isfinite(sight_al).all(2).any(1).all())
    tr = np.nansum(share[:, :, None] * np.nan_to_num(transl_al), axis=0); tr[~lifted.any(0)] = np.nan
    if path_rule:
        tr = fuse_path(transl_al, share, lifted, sight_al, PATH_ALONG_SIGHT)
    transl_f = moving_average_in_runs(tr, SMOOTH_FRAMES, centred=True)
    # ---- body
    w = np.stack([ramp_inside(usable[i], ramp) * share[i][:, None] for i in range(V)])         # (V, T, 21)
    body_raw = chordal_mean([body[i] for i in range(V)], w); none = ~usable.any(0); body_raw[none] = np.nan
    # the hips' and knees' flexion in two bands: the posture stays that mean's; the stride's swing comes from a mean that
    # also weights each view by how side-on it sees the walker (a camera the walker moves straight at reads it short)
    swing_rule = V > 1 and bool(np.isfinite(side).any(1).all())
    if swing_rule:
        wf = w[:, :, FLEX] * (SIDE_ON_FLOOR + np.nan_to_num(side))[:, :, None]
        alt = chordal_mean([body[i][:, FLEX] for i in range(V)], wf); alt[none[:, FLEX]] = np.nan
        body_raw[:, FLEX] = flexion_two_band(body_raw[:, FLEX], alt, FLEX_SLOW_S * fps)
    body_f = smooth_rotations(body_raw, SMOOTH_FRAMES)
    export_ok = export_mask(body_f, TORSO_LEGS, min_run)                                         # legs and torso; an arm may be missing
    # ---- the walkway: the scaled views fused with the same shares (when the scale fits were accepted), the mat's
    # pose in the world from the fused feet against the footfalls, the video path in the mat frame, and the pelvis
    # path ON THE MAT from the footprints plus the video pose (no depth estimate in it)
    transl_scaled = transl_f.copy(); transl_mat = np.full((T, 3), np.nan); transl_mat_video = np.full((T, 3), np.nan); transl_ww_world = np.full((T, 3), np.nan)
    wfit = None; mat_cov = 0.0
    if calibrated:
        ts = np.nansum(share[:, :, None] * np.nan_to_num(transl_al_sc), axis=0); ts[~lifted.any(0)] = np.nan
        if path_rule:
            ts = fuse_path(transl_al_sc, share, lifted, sight_al, PATH_ALONG_SIGHT)
        transl_scaled = moving_average_in_runs(ts, SMOOTH_FRAMES, centred=True)
        if max(abs(np.log(k)) for k in wscale) > np.log(1.5):
            flags.append(f"walkway calibration scales a view's depth by {max(wscale):.2f}x / {min(wscale):.2f}x")
    if ww is not None:
        Fm = feet_positions(root_f, body_f, transl_scaled); Fq = np.stack([_interp_runs(t_imu, Fm[:, i], ww[0]) for i in range(2)], axis=1); okq = np.isfinite(Fq).all(axis=(1, 2))
        if okq.sum() >= WALKWAY_MIN_ROWS:
            th, o2, rms_m = fit_mat_pose(landed_foot_xz(Fq[okq], ww[2][okq])[:, [0, 2]], ww[1][okq]); cy, sy = np.cos(th), np.sin(th)
            def to_mat(dx, dz): return cy * dx - sy * dz, sy * dx + cy * dz                 # inverse of the mat -> world map
            mx, mz = to_mat(transl_scaled[:, 0] - o2[0], transl_scaled[:, 2] - o2[1]); transl_mat_video = np.c_[mx, transl_scaled[:, 1], mz]
            pm, mat_cov, n_used = mat_pelvis_path(root_f, body_f, t_imu, stances, to_mat, fps)
            transl_mat = np.c_[pm[:, 0], transl_f[:, 1], pm[:, 1]]
            okm = np.isfinite(pm).all(1)
            if okm.any():
                w_ = _mat_to_world(th, o2, pm[okm, 0], pm[okm, 1]); transl_ww_world[okm] = np.c_[w_[:, 0], transl_f[okm, 1], w_[:, 1]]
            wfit = {"ok": True, "scale": [float(k) for k in wscale], "per_view": {v: wfits[i] for i, v in enumerate(views)}, "mat_yaw_deg": float(np.degrees(th)),
                    "mat_offset_m": [float(x) for x in o2], "rms_m": rms_m, "n_footfalls": int(okq.sum()), "stances_used_for_path": int(n_used),
                    "path_from_walkway_frac": float(mat_cov), "scale_fits_accepted": bool(calibrated)}
            if not calibrated:
                flags.append("info: walkway footfalls present but no view's depth-scale fit was accepted (" + "; ".join(f"{v}: {('rms %.2f m, scale %.2f' % (wfits[i]['rms_after_m'], wfits[i]['scale'])) if wfits[i] else 'too few rows'}" for i, v in enumerate(views)) + "); the mat path still comes from the footprints")
        else:
            calibrated = False; flags.append(f"walkway footfalls present ({len(ww[0])}) but fewer than {WALKWAY_MIN_ROWS} fall inside the lifted frames: no mat frame")
    elif stances is not None and len(stances["t0"]):
        flags.append(f"info: walkway data present but only {len(stances['t0'])} footfall(s): the mat is not used")
    # ---- heading-relative root
    yaw, yaw_u, root_hr = heading_frame(root_f)
    yaw_rate = rate_in_runs(yaw_u, t_imu); vel_w = rate_in_runs(transl_scaled, t_imu)      # d/dt inside runs, on the IMU clock; the fused video path's steps: the stride's profile
    if np.isfinite(transl_ww_world).any():                         # on the mat its speed over a stride and a half comes from the footprints: no depth estimate in it
        slow = moving_average_in_runs(rate_in_runs(transl_ww_world, t_imu) - vel_w, int(round(MAT_VEL_S * fps)) | 1); vel_w[:, [0, 2]] += np.nan_to_num(slow)[:, [0, 2]]
    vel_hr = heading_rel(vel_w, yaw)
    # ---- one shape
    bsel = (frac >= 0.999) & visible
    if bsel.sum() < ALIGN_MIN_FRAMES:
        bsel = (frac >= 0.75) & visible
    if bsel.sum() < ALIGN_MIN_FRAMES:
        bsel = visible & lifted
    betas_rec = np.nanmedian(betas[bsel], axis=0) if bsel.any() else np.zeros(10)
    betas_view = {v: [float(x) for x in np.nanmedian(betas[i][bsel[i]], axis=0)] for i, v in enumerate(views) if bsel[i].any()}
    if len(betas_view) == 2:
        d_b = float(np.linalg.norm(np.diff(np.array(list(betas_view.values())), axis=0)))
        if d_b > BETAS_DIFF_WARN:
            flags.append(f"the views' body shapes differ (|dbetas| {d_b:.2f}); check that both views hold the same person")
    else:
        d_b = None
    # ---- weights
    evidence = np.clip(w.sum(0), 0.0, 1.0)                                                         # (T, 21)
    agree = np.ones((T, 21))
    if V >= 2:
        pair = usable[0] & usable[1]
        if pair.any():
            d = geodesic(body[0], body[1]) * np.clip(2.0 * np.minimum(share[0], share[1]), 0.0, 1.0)[:, None]     # the disagreement the merged joint contains: all of it at an even split, none when the root rule set a view aside
            f = 1.0 / (1.0 + (np.maximum(np.nan_to_num(d, nan=0.0) - AGREE_FREE_DEG, 0.0) / AGREE_SCALE_DEG) ** 2); agree = np.where(pair, f, 1.0)
    weight = evidence * agree * export_ok[:, None]; weight[~np.isfinite(body_f).all(2)] = 0.0
    root_weight = np.clip(raw.sum(0), 0.0, 1.0); root_weight[~np.isfinite(root_f).all(1)] = 0.0
    transl_dis = np.full(T, np.nan)
    if V >= 2:
        both_l = lifted[0] & lifted[1]; transl_dis[both_l] = np.linalg.norm(transl_al[0, both_l] - transl_al[1, both_l], axis=1)
    transl_weight = root_weight.copy(); transl_weight[~np.isfinite(transl_f).all(1)] = 0.0       # the views' position disagreement is recorded, not weighted (module docstring, item 9)
    source = np.zeros((T, 21), np.int8)
    for i in range(V):
        source |= (usable[i].astype(np.int8) << i)
    # ---- QC numbers
    body_step = step_stats(body_f); root_step = step_stats(root_f[:, None, :])
    exp_frac = float(export_ok.mean()); n_runs = int((np.diff(np.r_[0, export_ok.astype(int), 0]) == 1).sum())
    has_j = np.isfinite(body_f).all(2); legs_ok = has_j[:, LEGS].all(1); torso_ok = has_j[:, SPINE].all(1)
    cov = {"legs": float(legs_ok.mean()), "torso": float(torso_ok.mean()), "legs_and_torso": float((legs_ok & torso_ok).mean()),
           "arms": float(has_j[:, ARMS].all(1).mean()), "all_joints": float(has_j.all(1).mean())}
    if (legs_ok & ~torso_ok).mean() > 0.05:
        flags.append(f"info: {(legs_ok & ~torso_ok).mean():.0%} of the lifted frames are not exported because no view saw the whole torso (the legs were seen): "
                     "the upper body is out of one view's panel while the other view has no subject")
    n_glitch = {v: int(glitch[i].sum()) for i, v in enumerate(views)}
    if sum(n_glitch.values()):
        flags.append("info: a view's root flipped for an instant and came back; interpolated across before merging (" + ", ".join(f"{v}: {n} frame(s)" for v, n in n_glitch.items() if n) + ")")
    if exp_frac < 0.5:
        flags.append(f"the export keeps only {exp_frac:.0%} of the lifted frames")
    if np.isfinite(root_step["p99_deg"]) and root_step["p99_deg"] > 15:
        flags.append(f"merged root steps {root_step['p99_deg']:.0f} deg between frames at the 99th percentile")
    okr_ = np.isfinite(root_f).all(1); n_flip = 0
    if okr_.sum() > 1:
        rr = root_f[okr_]; n_flip = int((np.degrees((R.from_rotvec(rr[:-1]).inv() * R.from_rotvec(rr[1:])).magnitude()) > 90.0).sum())
    if n_flip:
        flags.append(f"merged root turns more than 90 deg between consecutive frames {n_flip} time(s) (a view's lift flipping; see the demo's heading)")
    kq = max(1, int(round(FAST_TURN_S * fps))); n_fast = 0                                       # a turn no body makes: more than 90 deg inside a quarter second
    if T > kq:
        pair = okr_[:-kq] & okr_[kq:]; fast = np.zeros(T - kq, bool)
        if pair.any():
            fast[pair] = np.degrees((R.from_rotvec(root_f[:-kq][pair]).inv() * R.from_rotvec(root_f[kq:][pair])).magnitude()) > FAST_TURN_DEG
        n_fast = int((np.diff(np.r_[0, fast.astype(int)]) == 1).sum())
    if n_fast and not n_flip:
        flags.append(f"merged root turns more than {FAST_TURN_DEG:.0f} deg within {FAST_TURN_S} s {n_fast} time(s): faster than a body turns (one view's lift snapping round; see the demo's heading)")
    if t_mismatch > 0.002:
        flags.append(f"the views' IMU times for the same frame differ by up to {t_mismatch * 1000:.0f} ms")
    status = "WARN" if any(not f.startswith("info:") for f in flags) else "PASS"
    summary = {"status": status, "flags": flags, "views": views, "reference_view": views[ref], "n_frames": T, "fps": fps,
               "imu_span_s": [float(np.nanmin(t_imu)), float(np.nanmax(t_imu))],
               "export": {"frac_of_lifted_frames": exp_frac, "n_frames": int(export_ok.sum()), "n_runs": n_runs, "min_run_frames": min_run},
               "coverage_of_lifted_frames": cov, "per_view": {v: {"lifted_frames": int(lifted[i].sum()), "visible_frames": int(visible[i].sum()),
                                                                 "detected_frames": int(detected[i].sum()), "rest_held_frames": int(rest_held[i].sum()),
                                                                 "rest_held_seen_frames": int((rest_held[i] & visible[i] & ~detected[i]).sum()),
                                                                 "mean_usable_fraction": float(frac[i][lifted[i]].mean()) if lifted[i].any() else 0.0,
                                                                 "legs_usable_frac": float(usable[i][:, LEGS].all(1).mean()), "root_share_mean": float(share[i].mean()),
                                                                 "side_on_mean": (float(np.nanmean(side[i])) if np.isfinite(side[i]).any() else None),
                                                                 "alignment": align[i]} for i, v in enumerate(views)},
               "continuity": {"body_step_p99_deg": body_step["p99_deg"], "body_steps_over_20deg_frac": body_step["frac_over_20"], "root_step_p99_deg": root_step["p99_deg"], "root_flips_over_90deg": n_flip,
                              "root_turns_over_90deg_in_quarter_s": n_fast, "root_glitch_frames_repaired": n_glitch},
               "weight": {"mean_over_export": float(weight[export_ok].mean()) if export_ok.any() else 0.0, "frac_cells_over_0_5": float((weight > 0.5).mean()),
                          "root_weight_mean": float(root_weight.mean()), "transl_weight_mean": float(transl_weight.mean())},
               "flexion": {"rule": "two bands: posture from the views' weighted mean, stride swing weighted by side_on" if swing_rule else "the views' weighted mean",
                           "joints": [SMPLX_BODY_JOINTS[j] for j in FLEX], "slow_sigma_s": FLEX_SLOW_S, "side_on_floor": SIDE_ON_FLOOR},
               "translation": {"rule": (("the views' steps fused and summed, each across its camera's line of sight" if sight_known else "the views' steps fused by their shares and summed (no camera-frame files: no lines of sight)") if path_rule else "the one view's path"),
                               "along_sight_weight": PATH_ALONG_SIGHT, "views_disagree_median_m": (float(np.nanmedian(transl_dis)) if np.isfinite(transl_dis).any() else None),
                               "views_disagree_p90_m": (float(np.nanpercentile(transl_dis, 90)) if np.isfinite(transl_dis).any() else None),
                               "note": "GVHMR's translation is a monocular depth estimate; on the rendered session it is within 0.1 m near the mat centre and off by 1-3 m at the far ends of the walk, which the two views expose where both see the subject (transl_disagree_m); the merged path is fused from the views' steps, so that offset does not enter it, and transl_weight does not discount by it"},
               "betas": {"recording": [float(x) for x in betas_rec], "per_view": betas_view, "n_frames_used": int(bsel.sum()), "view_difference_norm": d_b},
               "walkway": ({**ww_info, "fit": wfit, "per_view": {v: wfits[i] for i, v in enumerate(views)}} if ww is not None else {**ww_info, "fit": None, "note": "no usable walkway footfalls in the recording CSV"}),
               "mirror_check": mirror_info, "views_mirrored": [v for i, v in enumerate(views) if mirrored[i]],
               "camera_fit_rms_m": {v: r_ for v, r_ in zip(views, cam_rms)},
               "floor_slope_deg": {v: path_slope_deg(transl[i], fps) for i, v in enumerate(views)},
               "imu_time_mismatch_s": t_mismatch, "files": ["merged_smpl.npz", "merge.json"],
               "note": "the body IMUs are not used anywhere in this file: the recording CSV serves the sync and the subject identity only"}
    if write:
        np.savez_compressed(rec_dir / "merged_smpl.npz", frame_idx=frames, imu_t_s=t_imu, body_pose=body_f.reshape(T, 63).astype(np.float32),
                            global_orient=root_f.astype(np.float32), transl=transl_f.astype(np.float32), transl_scaled=transl_scaled.astype(np.float32),
                            transl_mat=transl_mat.astype(np.float32), transl_mat_video=transl_mat_video.astype(np.float32), transl_from_walkway=transl_ww_world.astype(np.float32),
                            walkway_scale=wscale.astype(np.float32), transl_calibrated=bool(calibrated), view_mirrored=mirrored,
                            heading_yaw_deg=yaw_u.astype(np.float32),
                            heading_yaw_rate_deg_s=yaw_rate.astype(np.float32), global_orient_heading_rel=root_hr.astype(np.float32),
                            root_vel_heading_rel_m_s=vel_hr.astype(np.float32), betas=betas_rec.astype(np.float32), export_ok=export_ok,
                            weight=weight.astype(np.float32), root_weight=root_weight.astype(np.float32), transl_weight=transl_weight.astype(np.float32),
                            transl_disagree_m=transl_dis.astype(np.float32), source=source, usable=usable, share=share.astype(np.float32), side_on=side.astype(np.float32), sight_line=sight_al.astype(np.float32),
                            views=np.array(views), reference_view=views[ref], view_alignment_yaw_deg=np.array([a["yaw_deg"] for a in align], np.float32),
                            view_alignment_offset_m=np.array([a["offset_m"] for a in align], np.float32), joints=np.array(SMPLX_BODY_JOINTS), fps=fps,
                            fields_doc=FIELDS_DOC, note=summary["note"])
        (rec_dir / "merge.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"  [{rec_dir.name}] merge: {status} export {exp_frac:.0%} of {T} frames, root step p99 {root_step['p99_deg']:.1f} deg, ref {views[ref]} {flags}")
    return summary
