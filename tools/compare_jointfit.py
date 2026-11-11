"""The merge against the joint two-view fit, both scored against a rendered sample's truth.

    python -m wgvp jointfit <root>                                    # writes jointfit_smpl.npz beside merged_smpl.npz
    python tools/compare_jointfit.py <root> [--true-cameras] [--pair "Sub 001/Sub001_selfpace=sample/HC900_gt.npz" ...] [--out docs/jointfit.json]

<root> is a data folder the pipeline has run on (GPU stage and merge included). Without --pair the six recordings of
the partner layout (`tools/make_partner_layout.py`) are scored against the bundled samples' truths. Each recording
is scored three ways on the SAME frames (the merge's export rule) and the SAME joints (a limb counts only where the
merge outputs every joint that places it):

  merge          merged_smpl.npz
  joint fit      jointfit_smpl.npz (the cameras' relative pose found from the walker)
  true cameras   the same fit with the render's true camera pose held fixed (--true-cameras; computed here, not
                 written): what a calibrated rig would give, the upper bound of the self-calibration

The truth is an SMPL body and the estimates are SMPL-X bodies: the two models' joints of the same name differ, so
joint ROTATIONS compare only down to a floor of several degrees (design.md, section 2.15). The measures
that do not depend on either model's conventions come from joint POSITIONS: the direction of a body segment (after
one yaw for the recording, the unknown heading of the world), the knee's angle between thigh and shank, the thigh's
angle in the sagittal plane, the pelvis's frame from its three rigid joints, and the pelvis path after one rigid
fit in the floor plane. Jitter is the RMS second difference of the limb joints against the pelvis, in cm per frame
squared, and is read against the truth's own value: lower than the truth means motion was smoothed away.
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
from wgvp.fusion import ARMS, LEGS, SMPLX_PARENTS, SPINE, fit_yaw, geodesic, heading_yaw_deg   # noqa: E402
from wgvp.jointfit import body_joints, find_body_models, jointfit_recording, load_skeleton      # noqa: E402

PAIRS = ["Sub 001/Sub001_selfpace=sample/HC900_gt.npz", "Sub 001/Sub001_tug=sample_v4/HC903_gt.npz", "Sub 002/Sub002_selfpace=sample_v2/HC901_gt.npz",
         "Sub 003/Sub003_selfpace=sample_v3/HC902_gt.npz", "Sub 003/Sub003_hurriedpace=sample_v5/HC904_gt.npz", "Sub 004/Sub004_selfpace_mat=sample_v6/HC905_gt.npz"]
SEGMENTS = {"thigh": [(1, 4), (2, 5)], "shank": [(4, 7), (5, 8)], "upper_arm": [(16, 18), (17, 19)], "forearm": [(18, 20), (19, 21)], "trunk": [(0, 12)]}
LIMB = [4, 5, 7, 8, 16, 17, 18, 19, 20, 21]                                     # knees, ankles, shoulders, elbows, wrists
ROWS = [("thigh direction (deg)", "dir_thigh", 2), ("shank direction (deg)", "dir_shank", 2), ("upper arm direction (deg)", "dir_upper_arm", 2), ("forearm direction (deg)", "dir_forearm", 2),
        ("trunk direction (deg)", "dir_trunk", 2), ("knee angle RMSE (deg)", "knee_rmse", 2), ("knee angle gain", "knee_gain", 2), ("thigh sagittal RMSE (deg)", "thigh_sag_rmse", 2),
        ("thigh sagittal gain", "thigh_sag_gain", 2), ("pelvis frame (deg)", "pelvis", 2), ("heading (deg)", "heading", 2), ("ankle position (cm)", "ankle_pos_cm", 2), ("limb position (cm)", "limb_pos_cm", 2),
        ("path median (m)", "path_med_m", 3), ("path p90 (m)", "path_p90_m", 3), ("speed ratio", "speed_ratio", 3), ("speed RMSE (m/s)", "speed_rmse", 3),
        ("root rotation (deg)", "rot_root", 2), ("leg rotations (deg)", "rot_legs", 2), ("spine rotations (deg)", "rot_spine", 2), ("arm rotations (deg)", "rot_arms", 2),
        ("leg directions at rest (deg)", "dirlegs_rest", 2), ("leg directions walking (deg)", "dirlegs_walk", 2), ("leg directions turning (deg)", "dirlegs_turn", 2),
        ("seated: root rotation (deg)", "sit_rot_root", 2), ("seated: thigh direction (deg)", "sit_dir_thigh", 2), ("seated: shank direction (deg)", "sit_dir_shank", 2),
        ("limb jitter at rest (cm)", "jitter_rest_cm", 2), ("limb jitter walking (cm)", "jitter_walk_cm", 2), ("limb jitter turning (cm)", "jitter_turn_cm", 2)]


def ancestors_ok(body: np.ndarray, j: int) -> np.ndarray:
    """(T,) bool: every body joint whose rotation places joint j (its ancestors below the root) has an output."""
    ok = np.ones(len(body), bool); p = SMPLX_PARENTS[j]
    while p > 0:
        ok &= np.isfinite(body[:, p - 1]).all(1); p = SMPLX_PARENTS[p]
    return ok


def _direc(J, p, c):
    d = J[:, c] - J[:, p]; return d / np.linalg.norm(d, axis=1, keepdims=True)


def _gain(est, tru):
    a, b = np.ravel(est), np.ravel(tru); ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50:
        return np.nan
    c = np.cov(a[ok], b[ok]); return float(c[0, 1] / c[1, 1])


def _pelvis_frame(J):
    """The pelvis's orientation from its own three rigid joints (the hips and the first spine joint)."""
    x = J[:, 1] - J[:, 2]; x = x / np.linalg.norm(x, axis=1, keepdims=True); y = J[:, 3] - 0.5 * (J[:, 1] + J[:, 2]); y = y - x * (x * y).sum(1, keepdims=True); y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], 2)


def _speed(p, fps, k=2):
    out = np.full(len(p), np.nan); out[k:-k] = np.linalg.norm(p[2 * k:, [0, 2]] - p[:-2 * k, [0, 2]], axis=1) * fps / (2 * k); return out


def score(est: dict, cells: np.ndarray, frames: np.ndarray, G: dict, sel: np.ndarray, phase: np.ndarray, skeleton, fps: float = 30.0) -> dict:
    """est: global_orient (T,3), body_pose (T,63), transl (T,3), betas (10,), frame_idx. cells (T,21): the body
    joints both arms output (the merge's); frames (T,): the frames scored; G: the truth; sel (T,): the truth's row
    of each frame (-1 where it has none); phase (T,): rest / walk / turn / other."""
    T = len(sel); has = sel >= 0; ex = frames & has; s = {"n": int(ex.sum())}
    root = np.asarray(est["global_orient"], float); body = np.where(cells[:, :, None], np.asarray(est["body_pose"], float).reshape(T, 21, 3), np.nan); tr = np.asarray(est["transl"], float)
    J = body_joints(root, body, tr, np.asarray(est["betas"], float), skeleton); gJ = G["joints3d"][sel.clip(0)].astype(float)[:, :22]
    gb = G["pose_aa"][sel.clip(0), 3:66].reshape(-1, 21, 3).astype(float); gr = G["pose_aa"][sel.clip(0), :3].astype(float)
    okr = np.isfinite(root).all(1) & has; yaw = fit_yaw(root, gr, okr & ex); Ry = R.from_euler("y", yaw, degrees=True)
    with np.errstate(all="ignore"):
        e = geodesic(body, gb); er = geodesic((Ry * R.from_rotvec(np.nan_to_num(root))).as_rotvec(), gr); er[~okr] = np.nan
        s.update(rot_root=np.nanmedian(er[ex]), rot_legs=np.nanmedian(e[ex][:, LEGS]), rot_spine=np.nanmedian(e[ex][:, SPINE]), rot_arms=np.nanmedian(e[ex][:, ARMS]))
        hd = (heading_yaw_deg(root) + yaw - heading_yaw_deg(gr) + 180) % 360 - 180; s["heading"] = np.nanmedian(np.abs(hd[ex & okr]))
        # segment directions, one yaw for the recording; a segment only where every joint that places it has an output
        seg_err = {}
        for nm, prs in SEGMENTS.items():
            a = []
            for p, c in prs:
                d = np.degrees(np.arccos(np.clip((Ry.apply(_direc(J, p, c)) * _direc(gJ, p, c)).sum(1), -1, 1))); d[~ancestors_ok(body, c)] = np.nan; a.append(d)
            seg_err[nm] = np.stack(a, 1); s[f"dir_{nm}"] = np.nanmedian(seg_err[nm][ex])
        legs = np.concatenate([seg_err["thigh"], seg_err["shank"]], 1)
        for p in ("rest", "walk", "turn"):
            k = ex & (phase == p); s[f"dirlegs_{p}"] = np.nanmedian(legs[k]) if k.any() else np.nan
        sit = ex & (phase == "sit")
        if sit.sum() > 30:
            s.update(sit_rot_root=np.nanmedian(er[sit]), sit_dir_thigh=np.nanmedian(seg_err["thigh"][sit]), sit_dir_shank=np.nanmedian(seg_err["shank"][sit]), sit_n=int(sit.sum()))
        # limb joints against the pelvis
        ep = np.linalg.norm(Ry.apply((J[:, LIMB] - J[:, :1]).reshape(-1, 3)).reshape(T, -1, 3) - (gJ[:, LIMB] - gJ[:, :1]), axis=2)
        for i, j in enumerate(LIMB):
            ep[~ancestors_ok(body, j), i] = np.nan
        s["limb_pos_cm"] = 100 * np.nanmedian(ep[ex]); s["ankle_pos_cm"] = 100 * np.nanmedian(ep[ex][:, 2:4])
        # frame-to-frame jitter: RMS second difference of the limb joints against the pelvis (cm per frame squared), beside the truth's own motion
        fi = np.asarray(est["frame_idx"]); c3 = np.zeros(T, bool); c3[1:-1] = (fi[2:] - fi[:-2]) == 2
        def jit(X, k):
            rel = X[:, LIMB] - X[:, :1]; a = np.linalg.norm(rel[2:] - 2 * rel[1:-1] + rel[:-2], axis=2); kk = k[1:-1] & k[2:] & k[:-2] & c3[1:-1]
            return 100 * np.sqrt(np.nanmean(a[kk] ** 2)) if kk.sum() > 20 else np.nan
        for p in ("rest", "walk", "turn"):
            k = ex & (phase == p); s[f"jitter_{p}_cm"] = jit(J, k); s[f"truth_jitter_{p}_cm"] = jit(gJ, k)
        # the pelvis's frame, up to the constant gap between the two models' pelvises
        okj = ex & np.isfinite(J).all((1, 2)); Fe = np.einsum("ij,tjk->tik", Ry.as_matrix(), _pelvis_frame(np.nan_to_num(J))); Fg = _pelvis_frame(gJ)
        C = np.einsum("tji,tjk->tik", Fe[okj], Fg[okj]).mean(0); U, _, Vt = np.linalg.svd(C); C = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        pe = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", Fe @ C, Fg) - 1) / 2, -1, 1))); s["pelvis"] = np.nanmedian(pe[okj])
        # knee angle (thigh against shank) and the thigh's sagittal angle, each without its constant offset between the models
        both = ex & np.isfinite(body[:, LEGS]).all((1, 2))
        ka = lambda X: np.stack([np.degrees(np.arccos(np.clip((_direc(X, h, k) * _direc(X, k, a)).sum(1), -1, 1))) for h, k, a in ((1, 4, 7), (2, 5, 8))], 1)   # noqa: E731
        ke, kt = ka(J), ka(gJ); s["knee_rmse"] = np.sqrt(np.nanmean((ke[both] - kt[both] - np.nanmedian(ke[both] - kt[both])) ** 2)); s["knee_gain"] = _gain(ke[both], kt[both])
        def sag(X, r):
            out = []
            for h, k in ((1, 4), (2, 5)):
                d = r.apply(_direc(X, h, k)); out.append(np.degrees(np.arctan2(d[:, 2], -d[:, 1])))
            return np.stack(out, 1)
        he = sag(np.nan_to_num(J, nan=1.0), R.from_euler("y", -np.nan_to_num(heading_yaw_deg(root)), degrees=True)); ht = sag(gJ, R.from_euler("y", -heading_yaw_deg(gr), degrees=True))
        s["thigh_sag_rmse"] = np.sqrt(np.nanmean((he[both] - ht[both] - np.nanmedian(he[both] - ht[both])) ** 2)); s["thigh_sag_gain"] = _gain(he[both], ht[both])
    # the pelvis path (one rigid fit in the floor plane) and the horizontal speed
    ok = ex & np.isfinite(tr).all(1); P = tr[ok][:, [0, 2]]; Q = G["transl"][sel[ok]][:, [0, 2]].astype(float); pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc)); Rm = U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt; d = np.linalg.norm((P - pc) @ Rm + qc - Q, axis=1)
    s["path_med_m"] = np.median(d); s["path_p90_m"] = np.percentile(d, 90)
    c5 = np.zeros(T, bool); c5[2:-2] = (fi[4:] - fi[:-4]) == 4
    sm = _speed(tr, fps); gs = _speed(G["transl"].astype(float), fps); st = np.full(T, np.nan); st[has] = gs[sel[has]]
    mov = ok & c5 & np.isfinite(sm) & np.isfinite(st); mov[mov] = st[mov] > 0.25
    if mov.sum() > 30:
        s["speed_ratio"] = np.mean(sm[mov]) / np.mean(st[mov]); s["speed_rmse"] = np.sqrt(np.mean((sm[mov] - st[mov]) ** 2))
    return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in s.items()}


def phases(root: Path, sess: str, rec: str, t: np.ndarray) -> np.ndarray:
    """rest / sit / walk / turn per frame from the IMU file's event labels (as the pipeline saved them); 'other' without."""
    ph = np.array(["other"] * len(t), dtype=object); f = root / "wgvp_output" / sess / "work" / "imu_ref" / f"{rec}_imu_ref.npz"
    if not f.exists():
        return ph
    ir = np.load(f, allow_pickle=True)
    if "event_label" not in ir.files:
        return ph
    for lab, a, b in zip([str(x) for x in ir["event_label"]], ir["event_t0"], ir["event_t1"]):
        ph[(t >= a) & (t < b)] = "sit" if lab == "Sitting" else ("rest" if lab == "Standing" else ("walk" if lab == "Walk" else "turn"))
    return ph


def truth_cameras(G: dict, a: str, b: str):
    """The render's camera b against camera a (x_b = R x_a + t; x right, y down, z ahead) and the true up direction in camera a's frame."""
    def cv(v):
        M = np.diag([1.0, -1.0, -1.0, 1.0]) @ np.linalg.inv(G[f"{v}_cam_pose"].astype(np.float64)); return M[:3, :3], M[:3, 3]
    (Ra, ta), (Rb, tb) = cv(a), cv(b); Rt = Rb @ Ra.T; return Rt, tb - Rt @ ta, Ra[:, 1]


def _ang(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def compare_recording(root: Path, sess: str, rec: str, gt: Path, skeleton, true_cameras: bool = False) -> dict | None:
    d = root / "wgvp_output" / sess / "deliver" / rec
    if not (d / "merged_smpl.npz").exists() or not (d / "jointfit_smpl.npz").exists():
        print(f"  {sess}/{rec}: no merged_smpl.npz + jointfit_smpl.npz (run `wgvp merge` and `wgvp jointfit` first)"); return None
    m = np.load(d / "merged_smpl.npz"); jf = np.load(d / "jointfit_smpl.npz"); G = dict(np.load(gt, allow_pickle=True)); T = len(m["frame_idx"])
    gi = {int(f): i for i, f in enumerate(G["frame_idx"])}; sel = np.array([gi.get(int(f), -1) for f in m["frame_idx"]])
    cells = np.isfinite(m["body_pose"].reshape(T, 21, 3)).all(2); ex = m["export_ok"].astype(bool); ph = phases(root, sess, rec, m["imu_t_s"]); fps = float(m["fps"]) if "fps" in m.files else 30.0
    ph_all = np.where(ph == "sit", "rest", ph)                                              # the seated phase is scored on its own and stays in the whole-recording numbers as 'rest'
    def both(est):
        s = score(est, cells, ex, G, sel, ph_all, skeleton, fps); t = score(est, cells, ex, G, sel, ph, skeleton, fps) if (ph == "sit").any() else {}
        s.update({k: v for k, v in t.items() if k.startswith("sit_")}); return s
    out = {"merge": both({k: m[k] for k in ("global_orient", "body_pose", "transl", "betas", "frame_idx")}), "joint fit": both({k: jf[k] for k in ("global_orient", "body_pose", "transl", "betas", "frame_idx")})}
    a, b = [str(v) for v in jf["views"]]; js = json.loads((d / "jointfit.json").read_text()) if (d / "jointfit.json").exists() else {}
    cal = {"views": [a, b], "views_complete_frac": js.get("views_complete_frac"), "reprojection_px": js.get("reprojection_px"), "seconds": js.get("seconds")}
    if f"{a}_cam_pose" in G and f"{b}_cam_pose" in G:
        Rt, tt, up = truth_cameras(G, a, b); Rf = jf["cam_b_from_a_R"]
        cal.update(camera_rotation_error_deg=float(np.degrees(np.arccos(np.clip((np.trace(Rf @ Rt.T) - 1) / 2, -1, 1)))), camera_position_error_m=float(np.linalg.norm(jf["cam_b_from_a_t"] - tt)),
                   true_baseline_m=float(np.linalg.norm(tt)), up_error_deg=_ang(jf["world_from_cam_a_R"][1], up))
        if true_cameras:
            r = jointfit_recording(d, skeleton, cams=(Rt, tt), log=lambda *_: None, write=False, return_arrays=True)
            out["true cameras"] = both({**{k: r["arrays"][k] for k in ("global_orient", "body_pose", "transl", "betas")}, "frame_idx": m["frame_idx"]})
            cal["up_error_true_cameras_deg"] = _ang(r["arrays"]["world_from_cam_a_R"][1], up)
    out["calibration"] = cal; return out


def table(res: dict, arms: list[str]) -> str:
    recs = list(res); L = [f"{'median of ' + str(len(recs)) + ' recordings':32s}" + "".join(f"{a:>14s}" for a in arms)]
    for label, key, nd in ROWS:
        v = [[res[r][a].get(key, np.nan) for r in recs] for a in arms]
        if not np.isfinite(np.array(v, float)).any():
            continue
        L.append(f"{label:32s}" + "".join(f"{np.nanmedian(x):14.{nd}f}" for x in v))
    L.append(f"{'the truth\'s own limb motion':32s}" + "  ".join(f"{p} {np.nanmedian([res[r][arms[0]].get(f'truth_jitter_{p}_cm', np.nan) for r in recs]):.2f}" for p in ("rest", "walk", "turn")) + " cm")
    L.append("")
    for label, key, nd in ROWS[:1] + ROWS[1:2] + ROWS[11:12] + ROWS[13:14] + ROWS[9:10] + ROWS[17:18]:
        L.append(f"{label:28s}" + " | ".join(f"{a}: " + " ".join(f"{res[r][a].get(key, np.nan):6.{nd}f}" for r in recs) for a in arms))
    L.append(f"{'camera rotation error (deg)':28s}" + " ".join(f"{res[r]['calibration'].get('camera_rotation_error_deg', np.nan):6.1f}" for r in recs))
    L.append(f"{'both views whole body':28s}" + " ".join(f"{(res[r]['calibration'].get('views_complete_frac') or np.nan):6.2f}" for r in recs))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("--pair", action="append", help='"<session>/<recording>=<truth npz>" (repeatable; default: the partner layout\'s six)')
    ap.add_argument("--true-cameras", action="store_true"); ap.add_argument("--body-models"); ap.add_argument("--out", help="write the numbers as JSON")
    a = ap.parse_args(argv); root = Path(a.root)
    from wgvp.config import load_config
    skel = load_skeleton(find_body_models(load_config(root), a.body_models)); res = {}
    for pr in a.pair or PAIRS:
        key, gt = pr.split("="); sess, rec = key.split("/"); gt = Path(gt) if Path(gt).is_absolute() else PKG / gt
        r = compare_recording(root, sess, rec, gt, skel, a.true_cameras)
        if r is not None:
            res[rec] = r; print(f"  {rec}: " + " | ".join(f"{k} thigh {v['dir_thigh']:.1f} ankle {v['ankle_pos_cm']:.1f} cm path {v['path_med_m']:.2f} m" for k, v in r.items() if k != "calibration"), flush=True)
    if not res:
        return 1
    arms = [k for k in ("merge", "joint fit", "true cameras") if all(k in r for r in res.values())]; print(); print(table(res, arms))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
