"""Joint two-view fit, an optional stage after the merge (`wgvp jointfit <root>`): one SMPL-X body per frame fitted
to BOTH views' 2D keypoints at once, written as `jointfit_smpl.npz` (+ `jointfit.json`) beside `merged_smpl.npz`.
The merge stays the default training target; this file is a second estimate of the same frames to compare with it.

Why a second estimate: the merge takes each joint from the view(s) that can vouch for it, but every view was lifted
on its own, so a limb's depth along a camera's line of sight is that lift's guess and the pelvis path is two
monocular depth estimates fused. Two views of one body fix both by triangulation, once the pose of one camera
against the other is known. No calibration is delivered with the videos, so the fit finds it from the walker.

Per recording (two lifted views; needs the camera-frame lifts `smpl_incam.npz` and the SMPL-X model file). Camera
a, whose frame hosts the fit, is the merge's reference view: the view that vouches for the most joints.
  1. Seeds. The pose of camera b against camera a: the rigid fit of the two lifts' in-camera joints on the frames
     both views detected the subject. The body: the merge's pose. The pelvis: the two lifts' in-camera pelvis,
     camera b's carried into camera a's frame. The world: the rotation that takes the merge's world root onto view
     a's in-camera root, on the frames where view a can vouch for the torso (a body cut off by the picture's edge
     gives a wrong root).
  2. Unknowns. Per frame a correction of the root and of each body joint (R = R_seed exp(d)) and the pelvis
     position in camera a's frame; per recording the body shape, the pose of camera b (unless given), the world
     rotation, and six left/right-mirrored offsets between ViTPose's keypoints and the model's joints (a keypoint
     marks the surface of a shoulder or hip, not the joint centre).
  3. Loss. Reprojection of the twelve body keypoints in both views, in metres at the body's depth (so a near and
     a far camera count alike), under a Geman-McClure loss that sets aside a keypoint more than ~3 sd (9 cm) off;
     a prior that holds every correction near zero (root 0.05 rad, limbs 0.20, spine and collars 0.10); smoothness
     of the corrections and of the pelvis's acceleration; the shape near the lifts' mean.
  4. The up direction. Two views cannot see the pelvis's pitch, and a world tilted by a few degrees costs the
     reprojection nothing, so the world's up is pinned. Each lift knows its camera's ROLL to 1-2 deg and not its
     PITCH (on the rendered sessions the high side camera's 31 deg of pitch is under-estimated by about 10): the
     image direction that is level in the world needs the roll only (`level_axis`), and two level directions from
     two cameras fix the up direction as their cross product (`up_from_levels`). The world's yaw stays free and
     is held by the root prior, so the output's heading is the merge's.
  5. Output in the merge's world: `global_orient` = the merge's root with the fitted correction, `transl` = the
     fitted pelvis carried into the world and set on the merge's path by one constant offset. Joints and frames
     the merge does not output are not output here either (`export_ok` is the merge's): the fit refines what the
     merge exports and never adds to it. The walkway path (`transl_mat`) is not refitted; take it from the merge.
     The output is a 3-frame moving average inside runs of emitted frames (the merge uses 5): the fit follows the
     keypoints frame by frame, and without it a standing body shimmers at 2-6 times the motion the truth has at
     rest. Three frames bring the rest jitter to the truth's and cost the knee's range 0.01 of gain; five cost 0.04.

What it buys and what it does not, measured on the six rendered sessions (docs/design.md, section 2.15):
the pelvis path and the limb segments' directions improve clearly; the root's rotation, the heading and the seated
phase of the TUG do not. Self-calibration is weaker when one view sees only part of the body for most of the
recording (`jointfit.json`: `views_complete_frac`, noted below 0.5); a known camera pose can be passed in.
The body IMUs are not used here either.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .fusion import SMPLX_PARENTS, SMPLX_REST_JOINTS, LEGS, ARMS, geodesic, mirror_body, mirror_root, mirror_transl, moving_average_in_runs, smooth_rotations

# COCO-17 keypoint -> SMPL-X body joint (index in the 22-joint tree); the face points are not delivered
KP_JOINT = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
KPS = sorted(KP_JOINT); KPJ = [KP_JOINT[k] for k in KPS]
KP_MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]          # left <-> right partner of every COCO-17 keypoint
# keypoint offsets: the joint whose global rotation carries each keypoint's offset, and the six left/right pairs
OFF_FRAME = {16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 1: 0, 2: 0, 4: 1, 5: 2, 7: 4, 8: 5}
PAIRS = [(16, 17), (18, 19), (20, 21), (1, 2), (4, 5), (7, 8)]                  # shoulders, elbows, wrists, hips, knees, ankles
OFF_SD_M = [0.05, 0.05, 0.05, 0.15, 0.05, 0.05]                                 # the hip keypoint sits furthest from its joint
TRUNK = [2, 5, 8, 12, 13]                                                       # spine and collars (body index): five joints seen through two shoulder points
SEED_TORSO = [0, 1, 2, 5, 8]                                                    # hips and spine: what a view must vouch for to seed the world from its root
PARAMS = {"iters": (150, 450), "lr": 0.01, "sd_kp_m": 0.03, "sd_root_rad": 0.05, "sd_body_rad": 0.20, "sd_trunk_rad": 0.10, "w_up": 100.0,
          "smooth_frames": 3, "min_conf": 0.3, "margin_px": 8.0, "min_kp": 4, "min_level_cross": 0.35, "min_shared_frames": 30, "complete_kp": 10, "complete_frac_warn": 0.5, "min_baseline_m": 0.3}
FIELDS_DOC = (
    "frame_idx, imu_t_s: the merge's timeline (same rows as merged_smpl.npz); "
    "body_pose (T,63): SMPL-X body pose from the joint two-view fit, axis-angle, NaN for the joints the merge does not output; "
    "global_orient (T,3) / transl (T,3): root in the merge's world (y up; the heading is the merge's, the pelvis path is triangulated from both views and set on "
    "the merge's path by one constant offset); betas (10): the fitted body shape; export_ok (T): the merge's export rule, unchanged; "
    "n_keypoints (2,T): body keypoints (of 12) each view gave the fit; seen (T): some view gave at least 4, seen_both (T): both did (elsewhere the pose rests on "
    "one view, or on the merge's pose and the neighbouring frames); reproj_px (2,T): median reprojection residual of each view's keypoints; "
    "cam_b_from_a_R (3,3), cam_b_from_a_t (3): the pose of camera b against camera a (x_b = R x_a + t; x right, y down, z ahead), found with the fit unless given; "
    "world_from_cam_a_R (3,3): camera a's frame turned into the merge's world; keypoint_offsets_m (6,3): fitted keypoint-to-joint offsets of the left side "
    "(shoulder, elbow, wrist, hip, knee, ankle; the right side mirrored in x), in the parent joint's frame; views: (a, b), camera a first (the merge's reference view): the order of every per-view array here; fps. "
    "Pose, root and path are a 3-frame moving average inside runs of emitted frames (jointfit.json: params.smooth_frames). "
    "The walkway path and the loss weights are not refitted: take transl_mat and weight from merged_smpl.npz (same frames, same joints)")


# ---------------------------------------------------------------------------------------------- skeleton and geometry
def load_skeleton(body_models: Path) -> tuple[np.ndarray, np.ndarray]:
    """Rest joints (22, 3) and their shape directions (22, 3, 10) of the neutral SMPL-X model under `body_models`
    (the folder that holds smplx/SMPLX_NEUTRAL.npz, as GVHMR's inputs/checkpoints/body_models does)."""
    f = Path(body_models) / "smplx" / "SMPLX_NEUTRAL.npz"
    if not f.exists():
        raise FileNotFoundError(f"SMPL-X model not found at {f}: pass --body-models <folder that holds smplx/SMPLX_NEUTRAL.npz> "
                                "(the GVHMR checkout has it under inputs/checkpoints/body_models)")
    z = np.load(f, allow_pickle=True); Jr = np.asarray(z["J_regressor"], np.float64)[:22]
    return Jr @ np.asarray(z["v_template"], np.float64), np.einsum("jv,vck->jck", Jr, np.asarray(z["shapedirs"][:, :, :10], np.float64))


def mean_skeleton() -> tuple[np.ndarray, np.ndarray]:
    """The mean body's rest joints with no shape freedom: for tests, which have no model file."""
    return SMPLX_REST_JOINTS.astype(np.float64).copy(), np.zeros((22, 3, 10))


def find_body_models(cfg: dict | None, body_models: str | None = None) -> Path:
    if body_models:
        return Path(body_models)
    cand = Path((cfg or {}).get("gvhmr", {}).get("root") or "") / "inputs" / "checkpoints" / "body_models"
    return cand if (cand / "smplx").exists() else Path.home() / "data" / "WearGait" / "body_models"


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid (R, t) with Q ~ P R^T + t, least squares. P, Q (N, 3)."""
    pm, qm = P.mean(0), Q.mean(0); U, _, Vt = np.linalg.svd((P - pm).T @ (Q - qm)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ D @ U.T; return Rm, qm - Rm @ pm


def level_axis(up_cam: np.ndarray) -> np.ndarray:
    """The image-plane direction that is level in the world, from the up direction in the camera frame (x right, y
    down, z ahead). It depends on the camera's roll only: a wrong pitch moves `up_cam` in the y-z plane's sense and
    leaves the direction of (-u_y, u_x) alone."""
    h = np.array([-up_cam[1], up_cam[0], 0.0]); return h / np.linalg.norm(h)


def up_from_levels(h_a: np.ndarray, h_b_in_a: np.ndarray, up_hint: np.ndarray, min_cross: float = 0.35) -> tuple[np.ndarray, bool]:
    """The up direction in camera a's frame from each camera's level axis (camera b's already carried into camera
    a's frame): their cross product, signed toward `up_hint`. Two cameras whose level axes are nearly parallel
    (they look along the same line) cannot fix it; then the hint is returned. -> (up (3,), from_levels)."""
    u = np.cross(h_a, h_b_in_a); n = np.linalg.norm(u)
    if n < min_cross:
        return np.asarray(up_hint, float) / np.linalg.norm(up_hint), False
    u = u / n; return u * np.sign(u @ up_hint), True


def lift_up(glob_root: np.ndarray, cam_root: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """A view's own up direction in its camera frame: GVHMR's gravity world against its camera, median over frames."""
    k = ok & np.isfinite(glob_root).all(1) & np.isfinite(cam_root).all(1)
    Rwc = R.from_rotvec(glob_root[k]) * R.from_rotvec(cam_root[k]).inv(); u = np.median(Rwc.inv().apply([0.0, 1.0, 0.0]), axis=0); return u / np.linalg.norm(u)


def _chordal(Rs: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(Rs).mean(0)); return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _fill(x: np.ndarray) -> np.ndarray:
    """Linear fill along time of the rows that hold a NaN (ends held)."""
    x = np.array(x, float); t = np.arange(len(x)); flat = x.reshape(len(x), -1); ok = np.isfinite(flat).all(1)
    if ok.all() or not ok.any():
        return x
    for i in range(flat.shape[1]):
        flat[~ok, i] = np.interp(t[~ok], t[ok], flat[ok, i])
    return flat.reshape(x.shape)


def _fill_rot(Rm: np.ndarray) -> np.ndarray:
    """Fill NaN rotation matrices along time (linear on the entries, then back onto a rotation)."""
    T = len(Rm); F = _fill(Rm.reshape(T, 9)).reshape(T, 3, 3); U, _, Vt = np.linalg.svd(F); D = np.tile(np.eye(3), (T, 1, 1)); D[:, 2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def _rodrigues(aa):
    """(..., 3) rotation vectors -> (..., 3, 3) (torch), with a series near zero so the gradient exists there."""
    import torch
    th2 = (aa * aa).sum(-1, keepdim=True); th = torch.sqrt(th2 + 1e-12)
    a = torch.where(th2 < 1e-8, 1 - th2 / 6, torch.sin(th) / th); b = torch.where(th2 < 1e-8, 0.5 - th2 / 24, (1 - torch.cos(th)) / (th2 + 1e-12))
    x, y, z = aa.unbind(-1); O = torch.zeros_like(x)
    K = torch.stack([O, -z, y, z, O, -x, -y, x, O], -1).reshape(aa.shape[:-1] + (3, 3))
    return torch.eye(3, dtype=aa.dtype).expand(K.shape) + a[..., None] * K + b[..., None] * (K @ K)


def _project(P, K):
    """Camera-frame points (..., 3) -> pixels (..., 2) (torch)."""
    import torch
    z = P[..., 2:3].clamp(min=0.05); return P[..., :2] / z * torch.stack([K[0, 0], K[1, 1]]) + torch.stack([K[0, 2], K[1, 2]])


def body_joints(root_aa: np.ndarray, body_aa: np.ndarray, transl: np.ndarray, betas: np.ndarray, skeleton) -> np.ndarray:
    """Joint positions (T, 22, 3) of the shaped body in smplx's convention (the pelvis at its rest place plus the
    translation). Rows with a NaN root or translation are NaN; a NaN body joint is taken as identity."""
    J0, Jd = skeleton; J = J0 + Jd @ np.asarray(betas, float); T = len(root_aa); out = np.full((T, 22, 3), np.nan)
    ok = np.isfinite(root_aa).all(1) & np.isfinite(transl).all(1)
    if not ok.any():
        return out
    Rg = [R.from_rotvec(root_aa[ok])]; pos = [J[0] + np.asarray(transl, float)[ok]]; loc = np.nan_to_num(np.asarray(body_aa, float).reshape(T, 21, 3)[ok])
    for j in range(1, 22):
        p = SMPLX_PARENTS[j]; pos.append(pos[p] + Rg[p].apply(J[j] - J[p])); Rg.append(Rg[p] * R.from_rotvec(loc[:, j - 1]))
    out[ok] = np.stack(pos, 1); return out


# ---------------------------------------------------------------------------------------------- inputs
def unmirror_view(v: dict) -> dict:
    """A view whose picture the camera software flipped left-right, turned back into what the camera saw: keypoints
    reflected across the panel with the left/right labels swapped, the lifts reflected about the camera's x = 0
    plane (fusion.mirror_*), the principal point reflected. An involution."""
    W = float(v["size"][0]); o = dict(v)
    kp = np.array(v["kp"], float)[:, KP_MIRROR]; kp[:, :, 0] = W - kp[:, :, 0]; o["kp"] = kp
    K = np.array(v["K"], float); K[0, 2] = W - K[0, 2]; o["K"] = K
    o["cam_root"] = mirror_root(v["cam_root"]); o["cam_body"] = mirror_body(v["cam_body"]); o["cam_transl"] = mirror_transl(v["cam_transl"]); o["glob_root"] = mirror_root(v["glob_root"])
    return o


def load_views(rec_dir: Path, merged) -> list[dict] | None:
    """The two views of a merged recording on the merge's timeline, as `joint_fit` takes them (keypoints in the cut
    frame's pixels, lifts NaN where a view has no frame). None when a view lacks its camera-frame file."""
    rec_dir = Path(rec_dir); frames = np.asarray(merged["frame_idx"]); T = len(frames); fi = {int(f): i for i, f in enumerate(frames)}; out = []
    mirrored = np.asarray(merged["view_mirrored"], bool) if "view_mirrored" in merged else np.zeros(len(merged["views"]), bool)
    for i, name in enumerate(str(x) for x in merged["views"]):
        d = rec_dir / name
        if not all((d / f).exists() for f in ("smpl_global.npz", "smpl_incam.npz", "kp2d_vitpose.npz")):
            return None
        g, c, k = np.load(d / "smpl_global.npz"), np.load(d / "smpl_incam.npz"), np.load(d / "kp2d_vitpose.npz")
        if "K_fullimg_cut" not in k.files or "view_box" not in k.files or len(c["transl"]) != len(g["transl"]):
            return None
        idx = np.array([fi[int(f)] for f in g["frame_idx"] if int(f) in fi]); src = np.array([j for j, f in enumerate(g["frame_idx"]) if int(f) in fi], int)
        box = k["view_box"].astype(float); sc = float(k["cut_scale"]) if "cut_scale" in k.files else 1.0
        kp = np.full((T, 17, 3), np.nan); kk = k["kp2d"].astype(np.float64)[src]; kk[:, :, 0] = (kk[:, :, 0] - box[0]) * sc; kk[:, :, 1] = (kk[:, :, 1] - box[1]) * sc; kp[idx] = kk
        def on(x, shape):
            o = np.full((T,) + shape, np.nan); o[idx] = np.asarray(x, float).reshape((-1,) + shape)[src]; return o
        det = np.zeros(T, bool); det[idx] = np.asarray(g["visible"], bool)[src]
        v = {"name": name, "kp": kp, "K": k["K_fullimg_cut"].astype(np.float64), "size": ((box[2] - box[0]) * sc, (box[3] - box[1]) * sc), "detected": det,
             "cam_root": on(c["global_orient"], (3,)), "cam_body": on(c["body_pose"], (21, 3)), "cam_transl": on(c["transl"], (3,)), "glob_root": on(g["global_orient"], (3,)),
             "betas": np.nanmean(np.asarray(c["betas"], float).reshape(-1, 10), axis=0), "usable": np.asarray(merged["usable"][i], bool), "mirrored": bool(mirrored[i])}
        out.append(unmirror_view(v) if v["mirrored"] else v)
    return out


def keypoint_mask(v: dict, min_conf: float = 0.3, margin: float = 8.0) -> np.ndarray:
    """(T, 12) bool: the body keypoints of a view that enter the fit (confident, inside the panel with a margin, on a
    frame where the subject was detected). The same test as the merge's `usable`, per keypoint."""
    kp = v["kp"][:, KPS]; W, H = v["size"]
    with np.errstate(invalid="ignore"):
        return (np.nan_to_num(kp[:, :, 2]) >= min_conf) & (kp[:, :, 0] > margin) & (kp[:, :, 0] < W - margin) & (kp[:, :, 1] > margin) & (kp[:, :, 1] < H - margin) & v["detected"][:, None]


# ---------------------------------------------------------------------------------------------- the fit
def joint_fit(views: list[dict], merged, skeleton, cams=None, params: dict | None = None, log=None) -> dict | None:
    """Fit one body per frame to both views. `views`: two dicts from `load_views` (camera a hosts the frame);
    `merged`: global_orient (T, 3) and body_pose (T, 63) of the merge; `skeleton`: (rest joints, shape directions);
    `cams`: an optional known pose (R, t) of camera b against camera a (x_b = R x_a + t), held fixed.
    Returns the body in camera a's frame with the camera pose and the world rotation, or None when the views never
    see the subject together."""
    import torch
    D = torch.float64; P = {**PARAMS, **(params or {})}; va, vb = views; T = len(va["kp"])
    J0n, Jdn = skeleton; J0 = torch.tensor(J0n, dtype=D); Jd = torch.tensor(Jdn, dtype=D)
    bet0 = np.nanmean(np.stack([va["betas"], vb["betas"]]), axis=0)
    Ja = body_joints(va["cam_root"], va["cam_body"], va["cam_transl"], va["betas"], skeleton); Jb = body_joints(vb["cam_root"], vb["cam_body"], vb["cam_transl"], vb["betas"], skeleton)
    oka = np.isfinite(Ja).all((1, 2)) & np.isfinite(va["cam_body"]).all((1, 2)); okb = np.isfinite(Jb).all((1, 2)) & np.isfinite(vb["cam_body"]).all((1, 2))
    wa, wb = oka & va["detected"], okb & vb["detected"]; both = wa & wb
    if cams is None and both.sum() < P["min_shared_frames"]:
        return None
    Rab, tab = kabsch(Ja[both].reshape(-1, 3), Jb[both].reshape(-1, 3)) if both.sum() >= 3 else (np.eye(3), np.zeros(3))
    Rab0, tab0 = Rab, tab
    if cams is not None:
        Rab, tab = np.asarray(cams[0], float), np.asarray(cams[1], float)
    # the root is the merge's world root seen through a world -> camera a rotation: what two views cannot see (the
    # pelvis's pitch) stays where the lifts put it against gravity
    mr = np.asarray(merged["global_orient"], float); okm = np.isfinite(mr).all(1); ra = R.from_rotvec(np.nan_to_num(va["cam_root"])).as_matrix()
    k0 = okm & oka; kt = k0 & va["usable"][:, SEED_TORSO].all(1) & va["detected"]; k0 = kt if kt.sum() >= 30 else k0
    if not k0.any():
        return None
    Raw0 = _chordal(np.einsum("tij,tkj->tik", ra[k0], R.from_rotvec(mr[k0]).as_matrix()))                 # world -> camera a
    ups = [lift_up(v["glob_root"], v["cam_root"], v["detected"]) for v in (va, vb)]; hz = [level_axis(u) for u in ups]
    u0, from_levels = up_from_levels(hz[0], Rab.T @ hz[1], ups[0], P["min_level_cross"])
    if from_levels:                                                                                        # start the world at the up the loss pins it to
        c = Raw0[:, 1]; ax = np.cross(c, u0); sn = np.linalg.norm(ax)
        if sn > 1e-9:
            Raw0 = R.from_rotvec(ax / sn * np.arctan2(sn, c @ u0)).as_matrix() @ Raw0
    rw = R.from_rotvec(np.nan_to_num(mr)).as_matrix(); alt_a = np.einsum("ji,tjk->tik", Raw0, ra)
    alt_b = np.einsum("ji,tjk->tik", Rab @ Raw0, R.from_rotvec(np.nan_to_num(vb["cam_root"])).as_matrix())
    rootw = np.where(okm[:, None, None], rw, np.where(oka[:, None, None], alt_a, alt_b)); rootw[~(okm | oka | okb)] = np.nan; rootw = _fill_rot(rootw)
    pel = (J0n + Jdn @ bet0)[0]; pa = Ja[:, 0]; pb = (Jb[:, 0] - tab) @ Rab                               # the pelvis in camera a's frame from each view
    p0 = np.where((wa & wb)[:, None], 0.5 * (pa + pb), np.where(wa[:, None], pa, pb)); p0[~(wa | wb)] = np.nan; tr0 = _fill(p0) - pel
    body0 = np.asarray(merged["body_pose"], float).reshape(T, 21, 3); body0 = np.where(np.isfinite(body0), body0, np.where(oka[:, None, None], va["cam_body"], vb["cam_body"]))
    body0 = _fill(body0) if np.isfinite(body0).all((1, 2)).any() else np.zeros_like(body0)

    msk = [keypoint_mask(v, P["min_conf"], P["margin_px"]) for v in views]
    kp = [torch.tensor(np.nan_to_num(v["kp"][:, KPS, :2]), dtype=D) for v in views]
    wgt = [torch.tensor(np.where(m_, np.nan_to_num(v["kp"][:, KPS, 2]).clip(0, 1), 0.0), dtype=D) for v, m_ in zip(views, msk)]
    Kt = [torch.tensor(v["K"], dtype=D) for v in views]
    Rb0 = _rodrigues(torch.tensor(body0, dtype=D)); Rw0 = torch.tensor(rootw, dtype=D); aw0 = torch.tensor(Raw0, dtype=D)
    hz_t = [torch.tensor(h, dtype=D) for h in hz]; up_a = torch.tensor(ups[0], dtype=D)
    d_aw = torch.zeros(3, dtype=D, requires_grad=True); d_root = torch.zeros(T, 3, dtype=D, requires_grad=True); d_body = torch.zeros(T, 21, 3, dtype=D, requires_grad=True)
    tr = torch.tensor(tr0, dtype=D, requires_grad=True); bet = torch.tensor(bet0, dtype=D, requires_grad=True); bet_ref = torch.tensor(bet0, dtype=D)
    cr = torch.tensor(R.from_matrix(Rab).as_rotvec(), dtype=D, requires_grad=True); ct = torch.tensor(tab, dtype=D, requires_grad=True)
    off = torch.zeros(6, 3, dtype=D, requires_grad=True); off_sd = torch.tensor(OFF_SD_M, dtype=D)[:, None]; mir = torch.tensor([-1.0, 1.0, 1.0], dtype=D)
    psd_n = np.full(21, P["sd_body_rad"]); psd_n[TRUNK] = P["sd_trunk_rad"]; psd = torch.tensor(psd_n, dtype=D)[None, :, None]
    kidx = {j: i for i, j in enumerate(KPJ)}; shape_free = bool(np.abs(Jdn).sum() > 0)

    def up_target(Rc):
        u = torch.linalg.cross(hz_t[0], Rc.T @ hz_t[1]); n = u.norm()
        if n < P["min_level_cross"]:
            return up_a
        u = u / n; return u * torch.sign((u * up_a).sum())

    def model():
        root = (aw0 @ _rodrigues(d_aw)) @ Rw0 @ _rodrigues(d_root); body = Rb0 @ _rodrigues(d_body)
        Jw = J0 + Jd @ bet; pos = [Jw[0].expand(T, 3) + tr]; Rg = [root]
        for j in range(1, 22):
            p = SMPLX_PARENTS[j]; pos.append(pos[p] + (Rg[p] @ (Jw[j] - Jw[p])[:, None]).squeeze(-1)); Rg.append(Rg[p] @ body[:, j - 1])
        X = [None] * len(KPJ)
        for pi, (jl, jr) in enumerate(PAIRS):
            for j, o in ((jl, off[pi]), (jr, off[pi] * mir)):
                X[kidx[j]] = pos[j] + (Rg[OFF_FRAME[j]] @ o[:, None]).squeeze(-1)
        return torch.stack(X, 1), root, body, torch.stack(pos, 1)

    def loss():
        X, _, _, _ = model(); Rc = _rodrigues(cr); tot = 0.0
        for i, Xv in enumerate((X, X @ Rc.T + ct)):
            r = (_project(Xv, Kt[i]) - kp[i]) * (Xv[..., 2:3].clamp(min=0.3) / Kt[i][0, 0]) / P["sd_kp_m"]     # metres at the body's depth, in sd
            n2 = (r * r).sum(-1); tot = tot + (wgt[i] * n2 / (1.0 + n2 / 9.0)).sum()                           # Geman-McClure: turns over near 3 sd
        tot = tot + ((d_body / psd) ** 2).sum() + ((d_root / P["sd_root_rad"]) ** 2).sum()
        tot = tot + (((d_body[1:] - d_body[:-1]) / 0.02) ** 2).sum() + (((d_root[1:] - d_root[:-1]) / 0.02) ** 2).sum() + (((tr[2:] - 2 * tr[1:-1] + tr[:-2]) / 0.005) ** 2).sum()
        tot = tot + T * 0.05 * ((bet - bet_ref) ** 2).sum() + T * 0.5 * ((off / off_sd) ** 2).sum()
        return tot + T * P["w_up"] * (((aw0 @ _rodrigues(d_aw))[:, 1] - up_target(Rc.detach())) ** 2).sum()

    cam_p = [cr, ct] if cams is None else []; t0 = time.time()
    for si, (prm, n) in enumerate((([d_aw, tr] + cam_p, P["iters"][0]), ([d_aw, d_root, tr, d_body, off] + ([bet] if shape_free else []) + cam_p, P["iters"][1]))):
        opt = torch.optim.Adam(prm, lr=P["lr"])
        for it in range(n):
            opt.zero_grad(); L = loss(); L.backward(); opt.step()
            if log is not None and (it % 150 == 0 or it == n - 1):
                log(f"    joint fit stage {si} iteration {it:4d} loss {L.item():.0f} ({time.time() - t0:.0f} s)")
    with torch.no_grad():
        X, root, body, pos = model(); Rc = _rodrigues(cr); res = []; far = []
        for i, Xv in enumerate((X, X @ Rc.T + ct)):
            e = (_project(Xv, Kt[i]) - kp[i]).norm(dim=-1); far.append(((e * Xv[..., 2].clamp(min=0.3) / Kt[i][0, 0] / P["sd_kp_m"]).numpy() > 3.0) & msk[i])
            e = e.numpy(); e[~msk[i]] = np.nan; res.append(e)
        Raw = (aw0 @ _rodrigues(d_aw)).numpy(); Rcn = Rc.numpy()
        n_kp = np.stack([m_.sum(1) for m_ in msk]); _, lv = up_from_levels(hz[0], Rcn.T @ hz[1], ups[0], P["min_level_cross"])
        return {"root_cam": R.from_matrix(root.numpy()).as_rotvec(), "root_world": R.from_matrix((Rw0 @ _rodrigues(d_root)).numpy()).as_rotvec(),
                "body": R.from_matrix(body.numpy().reshape(-1, 3, 3)).as_rotvec().reshape(T, 21, 3), "transl_cam": tr.numpy().copy(), "joints_cam": pos.numpy(),
                "betas": bet.numpy().copy(), "R_ab": Rcn, "t_ab": ct.numpy().copy(), "R_ab_seed": Rab0, "t_ab_seed": tab0, "R_aw": Raw, "R_aw_seed": Raw0,
                "offsets": off.numpy().copy(), "res_px": res, "set_aside": far, "mask": msk, "n_kp": n_kp, "up_from_levels": bool(lv),
                "level_axes_angle_deg": float(np.degrees(np.arcsin(min(1.0, np.linalg.norm(np.cross(hz[0], Rcn.T @ hz[1])))))),
                "seen": (n_kp >= P["min_kp"]).any(0), "seen_both": (n_kp >= P["min_kp"]).all(0), "cams_given": cams is not None, "seconds": time.time() - t0, "params": P}


def to_world(sol: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fit in the world it was seeded from (the merge's, up to the origin): root (T, 3), body (T, 21, 3), joints
    (T, 22, 3) with the camera a's centre at the origin."""
    return sol["root_world"], sol["body"], sol["joints_cam"] @ sol["R_aw"]


# ---------------------------------------------------------------------------------------------- one recording, one data folder
def _angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


def jointfit_recording(rec_dir: Path, skeleton, params: dict | None = None, cams=None, log=print, write: bool = True, return_arrays: bool = False):
    """Fit the merged recording under <deliver>/<recording>/ to both of its views and write jointfit_smpl.npz +
    jointfit.json there. Returns the jointfit.json content (with the arrays under "arrays" when asked), or None
    when there is no merged recording. A recording the fit cannot take (one view, three views, no camera-frame
    files, views that never see the subject together) gets status SKIP and no npz."""
    rec_dir = Path(rec_dir); mf = rec_dir / "merged_smpl.npz"
    if not mf.exists():
        return None
    m = np.load(mf); views = [str(v) for v in m["views"]]; T = len(m["frame_idx"])

    def skip(why):
        s = {"recording": rec_dir.name, "status": "SKIP", "flags": [why], "views": views}
        if write:
            (rec_dir / "jointfit.json").write_text(json.dumps(s, indent=1))
        log(f"  [{rec_dir.name}] joint fit: SKIP ({why})"); return s
    if len(views) != 2:
        return skip(f"the joint fit takes two views, this recording has {len(views)}")
    V = load_views(rec_dir, m)
    if V is None:
        return skip("a view lacks its camera-frame lift, keypoints or intrinsics (smpl_incam.npz, kp2d_vitpose.npz with K_fullimg_cut)")
    ref = str(m["reference_view"]) if "reference_view" in m.files else views[0]
    V = sorted(V, key=lambda v: v["name"] != ref); views = [v["name"] for v in V]                          # camera a = the merge's reference view (stable: the other keeps its place)
    P = {**PARAMS, **(params or {})}; sol = joint_fit(V, m, skeleton, cams, P)
    if sol is None:
        return skip("the two views never see the subject together for long enough to find the cameras' relative pose")
    mb = m["body_pose"].astype(float).reshape(T, 21, 3); mr = m["global_orient"].astype(float); mt = m["transl"].astype(float); ex = m["export_ok"].astype(bool)
    root, body, J = to_world(sol); J0, Jd = skeleton; pel = (J0 + Jd @ sol["betas"])[0]; transl = J[:, 0] - pel
    okm = np.isfinite(mr).all(1) & np.isfinite(mt).all(1); k = okm & (ex if ex.any() else True)
    shift = np.median(mt[k] - transl[k], axis=0) if k.any() else np.zeros(3); transl = transl + shift
    root = np.where(okm[:, None], root, np.nan); transl = np.where(okm[:, None], transl, np.nan); body = np.where(np.isfinite(mb), body, np.nan)
    w = int(P["smooth_frames"])                                                                          # the keypoints' frame-to-frame noise shows on a body at rest
    if w > 1:
        body = smooth_rotations(body, w); root = smooth_rotations(root[:, None], w)[:, 0]; transl = moving_average_in_runs(transl, w, centred=True)
    # ---- what moved against the merge, and how well the keypoints are met
    with np.errstate(all="ignore"):
        db = geodesic(body, mb); dr = geodesic(root, mr); dp = np.linalg.norm((transl - mt)[:, [0, 2]], axis=1)
        nk = sol["n_kp"]; full = (nk >= P["complete_kp"]).all(0); sb = sol["seen_both"]; cfrac = float(full[sb].mean()) if sb.any() else 0.0
        px = [{"median": float(np.nanmedian(e)), "p90": float(np.nanpercentile(e[np.isfinite(e)], 90)), "set_aside_frac": float(f.sum() / max(1, np.isfinite(e).sum()))}
              if np.isfinite(e).any() else {"median": None, "p90": None, "set_aside_frac": None} for e, f in zip(sol["res_px"], sol["set_aside"])]
        stats = {"root_deg_median": float(np.nanmedian(dr[ex])) if ex.any() else None, "legs_deg_median": float(np.nanmedian(db[ex][:, LEGS])) if ex.any() else None,
                 "arms_deg_median": float(np.nanmedian(db[ex][:, ARMS])) if ex.any() else None, "path_m_median": float(np.nanmedian(dp[ex])) if ex.any() else None,
                 "path_m_p90": float(np.nanpercentile(dp[ex & np.isfinite(dp)], 90)) if (ex & np.isfinite(dp)).any() else None}
    flags = []
    if cams is None and cfrac < P["complete_frac_warn"]:
        flags.append(f"info: both views show the whole body in only {cfrac:.0%} of the frames they share: the cameras' relative pose rests on a partial view and is less certain "
                     "(7 deg off the truth on the rendered session of this kind against 3-5 deg on the others; the body's pose did not suffer there)")
    if not sol["up_from_levels"]:
        flags.append("the two cameras' level axes are nearly parallel: the up direction is the first view's own gravity estimate, whose pitch is not reliable")
    if any(v["mirrored"] for v in V):
        flags.append(f"{[v['name'] for v in V if v['mirrored']]} mirrored: reflected back before the fit (checked on fabricated data only)")
    base = float(np.linalg.norm(sol["t_ab"]))
    if base < P["min_baseline_m"]:
        flags.append(f"the two cameras stand {base:.2f} m apart: the views are one viewpoint, so the distance to the walker (and with it the path's length) rests on the body's size in "
                     "the picture and on the lens, not on two lines of sight")
    summary = {"recording": rec_dir.name, "status": "WARN" if any(not f.startswith("info:") for f in flags) else "PASS", "flags": flags, "views": views, "n_frames": int(T), "n_export_frames": int(ex.sum()),
               "frames_seen_by_both": int(sb.sum()), "frames_seen_by_one": int((sol["seen"] & ~sb).sum()), "export_frames_without_keypoints": int((ex & ~sol["seen"]).sum()),
               "views_complete_frac": cfrac, "reprojection_px": {v: p for v, p in zip(views, px)},
               "cameras": {"given": bool(sol["cams_given"]), "b_from_a_rotvec_deg": [float(x) for x in np.degrees(R.from_matrix(sol["R_ab"]).as_rotvec())],
                           "b_from_a_t_m": [float(x) for x in sol["t_ab"]], "baseline_m": base,
                           "moved_from_seed_deg": _angle(sol["R_ab"], sol["R_ab_seed"]), "moved_from_seed_m": float(np.linalg.norm(sol["t_ab"] - sol["t_ab_seed"]))},
               "up": {"rule": "cross product of the two cameras' level axes" if sol["up_from_levels"] else "the first view's own gravity estimate",
                      "level_axes_angle_deg": sol["level_axes_angle_deg"], "world_moved_from_seed_deg": _angle(sol["R_aw"], sol["R_aw_seed"])},
               "change_from_merge": stats, "betas": [float(x) for x in sol["betas"]], "keypoint_offsets_m": [[float(x) for x in o] for o in sol["offsets"]],
               "world_offset_m": [float(x) for x in shift], "seconds": float(sol["seconds"]), "params": {k_: (list(v_) if isinstance(v_, tuple) else v_) for k_, v_ in P.items()},
               "files": ["jointfit_smpl.npz", "jointfit.json"],
               "note": "an optional second estimate of the merge's frames; merged_smpl.npz stays the default target. The body IMUs are not used"}
    arrays = dict(frame_idx=m["frame_idx"], imu_t_s=m["imu_t_s"], body_pose=body.reshape(T, 63).astype(np.float32), global_orient=root.astype(np.float32), transl=transl.astype(np.float32),
                  betas=sol["betas"].astype(np.float32), export_ok=ex, n_keypoints=nk.astype(np.int16), seen=sol["seen"], seen_both=sb,
                  reproj_px=np.stack([np.nanmedian(np.where(np.isfinite(e).any(1, keepdims=True), e, 0.0), axis=1) * np.where(np.isfinite(e).any(1), 1.0, np.nan) for e in sol["res_px"]]).astype(np.float32),
                  cam_b_from_a_R=sol["R_ab"], cam_b_from_a_t=sol["t_ab"], world_from_cam_a_R=sol["R_aw"].T, keypoint_offsets_m=sol["offsets"].astype(np.float32),
                  views=np.array(views), fps=float(m["fps"]) if "fps" in m.files else 30.0)
    if write:
        np.savez_compressed(rec_dir / "jointfit_smpl.npz", **arrays, fields_doc=FIELDS_DOC, note=summary["note"])
        (rec_dir / "jointfit.json").write_text(json.dumps(summary, indent=1))
    log(f"  [{rec_dir.name}] joint fit: {summary['status']} reprojection {px[0]['median']:.1f} / {px[1]['median']:.1f} px, cameras moved {summary['cameras']['moved_from_seed_deg']:.1f} deg from the seed, "
        f"path moved {stats['path_m_median'] if stats['path_m_median'] is not None else float('nan'):.2f} m against the merge, {sol['seconds']:.0f} s {flags}")
    if return_arrays:
        summary = {**summary, "arrays": arrays, "solution": sol}
    return summary


def jointfit_root(root: Path, body_models: str | None = None, session: str | None = None, recording: str | None = None, log=print) -> list[dict]:
    """Every merged recording under <root>/wgvp_output (or one session / recording)."""
    from .config import load_config
    root = Path(root); skel = load_skeleton(find_body_models(load_config(root), body_models)); out = []
    for mf in sorted((root / "wgvp_output").glob("*/deliver/*/merged_smpl.npz")):
        if (session and mf.parent.parent.parent.name != session) or (recording and mf.parent.name != recording):
            continue
        s = jointfit_recording(mf.parent, skel, log=log)
        if s is not None:
            out.append(s)
    return out
