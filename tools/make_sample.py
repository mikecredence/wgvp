"""(Analysis side only.) Build the bundled dry-run sample as ONE combined two-view video shaped like the real WearGait
session videos (unequal panels, letterboxed side view, seated operator close to the camera, exits from one view).

Scene (rendered SMPL meshes, no real video):
  subject   WearGait hc100_selfpace (public) IK pose sequence walking a corridor (stance-foot-pinned root trajectory)
  co-walker hc102_selfpace, 1.5 m beside the subject, starting 0.8 s later (a distractor WITH gait)
  operator  a seated body close to the side camera, lower right, fidgeting (large in frame, no gait)
  bystander hc101's standing frames looped at the far wall
  light     a small square in the side panel, bright while the IMU records (optional cross-check only)
Panels: LEFT = along-walkway view 520x600 (camera at the walkway end, low: the head leaves the frame when close);
        RIGHT = oblique side view 900x500 letterboxed in 900x600 (black band below) seeing only the middle of the
        corridor, so the subject exits/re-enters that view at both ends. Combined frame 1420x600, 30 fps.
Files: HC900/SP/HC900_SP_DeID.mp4 + HC900/hc900_selfpace.csv (column subset of the public hc100_selfpace.csv) and
expected.json with the single sync offset, the panel boxes, and the subject's ground-truth hip projection per view.

    python tools/make_sample.py [--variant v1|v2|v3|v4|v5|v6]   (needs smplx, trimesh, pyrender, imageio[pyav], pandas and the inputs named below)
v2 (sample_v2/, session HC901) renders the same walk from two deliberately compromised views, after the real
session layout: the along view loses everything above the hips when the walker is close, the side view never
shows the head; the walker's ground-truth SMPL pose and 3D joints are saved beside it for the two-view
reconstruction experiment (tools/two_view_fusion.py). v3 (sample_v3/, session HC902) is a third room with another
walker (hc105), a co-walker, the along camera at the far end (legs cut when close) and the side camera on the other side.
v4 (HC903) is a Timed Up and Go with a chair, v5 (HC904) a hurried-pace walk, v6 (HC905) a self-paced walk on the mat
whose recording CSV carries walkway footfall positions fabricated from the truth (the public export has none).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
# Analysis-side inputs (not part of this repository): WearGait IK pose sequences (<name>_ik.npz with a (T, 24, 3, 3)
# "pose" array), the public WearGait recording CSVs, and the SMPL body model folder (smpl/SMPL_NEUTRAL.pkl).
RES = Path(os.environ.get("WGVP_SAMPLE_IK_DIR", Path.home() / "data/WearGait/results/transpose_smpl_ik_nodetrend_walkcal_v13"))
IMU = Path(os.environ.get("WGVP_SAMPLE_IMU_DIR", Path.home() / "data/WearGait/IMU data"))
BM = os.environ.get("WGVP_SAMPLE_BODY_MODELS", str(Path.home() / "data/WearGait/body_models"))
FPS = 30
LW, LH = 520, 600            # along-walkway panel
RW, RH, RC = 900, 600, 500   # side panel (content height RC, letterboxed to RH)


def _smplx_compat_shims():
    """chumpy (the legacy SMPL .pkl deserialiser pulled in by smplx) references numpy aliases (np.int, np.bool, ...)
    and inspect.getargspec, removed in modern Python / NumPy. Minimal shims instead of vendoring chumpy."""
    import inspect
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, alias in [("bool", bool), ("int", int), ("float", float), ("complex", complex), ("object", object), ("unicode", str), ("str", str)]:
        if not hasattr(np, name):
            setattr(np, name, alias)


def load_ik_aa(name: str) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rt
    P = np.load(RES / f"{name}_ik.npz")["pose"][::2]
    return Rt.from_matrix(P.reshape(-1, 3, 3)).as_rotvec().reshape(len(P), 72).astype(np.float32)


def fk(aa, smpl, transl=None):
    import torch
    V, J = [], []
    with torch.no_grad():
        for s in range(0, len(aa), 256):
            a = torch.tensor(aa[s:s + 256]); n = len(a)
            tr = torch.tensor(transl[s:s + 256]) if transl is not None else torch.zeros(n, 3)
            o = smpl(global_orient=a[:, :3], body_pose=a[:, 3:], betas=torch.zeros(n, 10), transl=tr)
            V.append(o.vertices.numpy()); J.append(o.joints[:, :24].numpy())
    return np.concatenate(V), np.concatenate(J)


def yaw_align(aa, smpl):
    from scipy.spatial.transform import Rotation as Rt
    _, J = fk(aa[:60], smpl)
    left = ((J[:, 1] - J[:, 2]) + (J[:, 16] - J[:, 17])).mean(0); left[1] = 0; left /= np.linalg.norm(left)
    fwd = np.cross(left, [0, 1, 0]); ang = np.arctan2(fwd[2], fwd[0])
    out = aa.copy(); out[:, :3] = (Rt.from_euler("y", ang) * Rt.from_rotvec(aa[:, :3])).as_rotvec().astype(np.float32)
    return out


def imu_vertical_displacement(csv_name: str, n_frames: int) -> np.ndarray:
    """The pelvis vertical displacement the accelerometer actually recorded (double-integrated LowerBack_FreeAcc_U,
    gait band, metres), resampled to the render's 30 fps. Imposing it on the mesh root makes the rendered pelvis
    motion consistent with the recording CSV, which is what the pipeline's selection and sync measure."""
    import pandas as pd
    sys.path.insert(0, str(PKG))
    from wgvp.signals import bandpass, highpass
    df = pd.read_csv(IMU / csv_name, usecols=lambda c: c in ("Time", "LowerBack_FreeAcc_U"))
    acc = np.nan_to_num(df["LowerBack_FreeAcc_U"].to_numpy(dtype=np.float64))
    v = highpass(np.cumsum(acc) / 100.0, 100.0, 0.5); d = highpass(np.cumsum(v) / 100.0, 100.0, 0.5); d = bandpass(d, 100.0)
    d = np.clip(np.nan_to_num(d), -0.06, 0.06)
    t100 = np.arange(len(d)) / 100.0; t30 = np.arange(n_frames) / FPS
    return np.interp(t30, t100, d).astype(np.float32)


def root_trajectory(aa, smpl, csv_name: str | None = None):
    """Horizontal path by stance-foot pinning; vertical = mean floor contact + the recorded IMU vertical displacement
    (csv_name) or the lowest-vertex-on-floor rule (csv_name None)."""
    from scipy.ndimage import uniform_filter1d
    V, J = fk(aa, smpl); feet = J[:, [10, 11]]; T = len(aa); tr = np.zeros((T, 3))
    for t in range(1, T):
        s = int(np.argmin(feet[t, :, 1] + feet[t - 1, :, 1])); d = feet[t, s] - feet[t - 1, s]
        tr[t] = tr[t - 1] - np.array([d[0], 0.0, d[2]])
    tr[:, 0] = uniform_filter1d(tr[:, 0], 9, mode="nearest"); tr[:, 2] = uniform_filter1d(tr[:, 2], 9, mode="nearest")
    floor = uniform_filter1d(V[..., 1].min(1), 5, mode="nearest")
    if csv_name is None:
        tr[:, 1] = -floor
    else:
        tr[:, 1] = -np.median(floor) + imu_vertical_displacement(csv_name, T)
    return tr.astype(np.float32)


def seated_sequence(T: int, rng) -> np.ndarray:
    """A seated body: hips and knees flexed ~90 deg, slow arm fidgeting (someone handling equipment)."""
    aa = np.zeros((T, 72), np.float32)
    aa[:, 3 + 0 * 3] = -1.45; aa[:, 3 + 1 * 3] = -1.45          # L/R hip flexion (joints 1, 2)
    aa[:, 3 + 3 * 3] = 1.5; aa[:, 3 + 4 * 3] = 1.5               # knees (joints 4, 5)
    aa[:, 3 + 15 * 3 + 2] = -0.9; aa[:, 3 + 16 * 3 + 2] = 0.9    # shoulders down (joints 16, 17)
    t = np.arange(T) / FPS
    aa[:, 3 + 17 * 3 + 1] = 0.6 + 0.35 * np.sin(2 * np.pi * 0.23 * t)      # L elbow
    aa[:, 3 + 18 * 3 + 1] = -0.6 - 0.35 * np.sin(2 * np.pi * 0.31 * t + 1)  # R elbow
    aa[:, 3 + 11 * 3] = 0.15 * np.sin(2 * np.pi * 0.11 * t)                # neck nod
    aa[:, :3] = [0.0, np.pi, 0.0]                                          # face -x... adjusted by placement below
    return aa


def look_at(eye, target):
    eye = np.asarray(eye, np.float64); target = np.asarray(target, np.float64)
    f = eye - target; f /= np.linalg.norm(f); r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r); u = np.cross(f, r)
    M = np.eye(4); M[:3, 0] = r; M[:3, 1] = u; M[:3, 2] = f; M[:3, 3] = eye
    return M


def project(P, cam_pose, fx, fy, cx, cy):
    Pc = (np.linalg.inv(cam_pose) @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    z = -Pc[:, 2]; u = fx * Pc[:, 0] / z + cx; v = cy - fy * Pc[:, 1] / z
    return np.c_[u, v], z


VARIANTS = {
    # v1: the dry-run sample. Along view: camera low at the walkway end (head cut when close); side view: oblique,
    # elevated, narrow (subject exits at both ends); a co-walker, a seated operator, a bystander.
    "v1": {"sid": "HC900", "subdir": "sample", "decoy": True, "seat": lambda xm, zm: [xm - 0.35, 0.0, zm - 1.55],
           "cams": lambda x0, x1, xm, zm: {"left": {"W": LW, "H": LH, "yfov": 55.0, "eye": [x0 - 1.6, 0.9, zm + 0.1], "target": [x0 + 2.2, 0.0, zm]},
                                           "right": {"W": RW, "H": RC, "yfov": 30.0, "eye": [xm + 1.0, 2.0, zm - 4.0], "target": [xm - 0.4, 0.5, zm + 0.4]}}},
    # v2: two deliberately compromised views of one walk, after the real session layout. Along view: camera 0.5 m
    # beyond the near end, 0.6 m high, looking down the mat: the whole body only at the far end, head and shoulders
    # gone mid-way, shins only at the near end (the pose is lost there). Side view: 6 m back, 1 m high, narrow and
    # tilted at the hips: the whole mat is in frame but the head is above the top edge for the entire pass and the
    # shoulders sit at it; the walker leaves the frame only at the very ends. No co-walker: the point of this sample
    # is reconstruction from two truncated views, and the ground-truth 3D pose is saved beside it.
    "v2": {"sid": "HC901", "subdir": "sample_v2", "decoy": False, "seat": lambda xm, zm: [xm + 2.4, 0.0, zm - 1.2],    # beside the far end, outside the side view's frustum: no occlusion of the mat
           "cams": lambda x0, x1, xm, zm: {"left": {"W": LW, "H": LH, "yfov": 60.0, "eye": [x0 - 0.5, 0.6, zm + 0.05], "target": [x0 + 1.2, 0.0, zm]},
                                           "right": {"W": RW, "H": RC, "yfov": 22.0, "eye": [xm, 1.0, zm - 6.0], "target": [xm, 0.35, zm]}}},
    # v3: a third room (sample_v3/, session HC902): a different walker (hc105: 6.3 m of walkway, the largest pelvis bob of
    # the public set) with hc100 as the co-walker on the far
    # side of the mat. The along camera stands at the FAR end and higher (1.5 m), so when the walker is close it is
    # the legs that leave its frame, not the head; the side camera is on the other side of the room, higher and
    # turned towards one end, so the walker leaves it at the other end only, where the along camera sees the whole
    # body. The operator sits close to the side camera, lower right; the bystander stands at the far wall.
    "v3": {"sid": "HC902", "subdir": "sample_v3", "decoy": True, "subject": "hc105_selfpace", "decoy_name": "hc100_selfpace", "decoy_dz": -1.5,
           "seat": lambda xm, zm: [xm + 0.6, 0.0, zm + 2.2], "bystander": lambda x0, x1, xm, zm: [xm - 1.0, 0.0, zm - 2.0],
           "cams": lambda x0, x1, xm, zm: {"left": {"W": LW, "H": LH, "yfov": 50.0, "eye": [x1 + 1.3, 1.5, zm - 0.15], "target": [x1 - 2.0, 0.4, zm]},
                                           "right": {"W": RW, "H": RC, "yfov": 40.0, "eye": [xm - 1.6, 2.2, zm + 3.5], "target": [xm + 1.0, 0.5, zm - 0.3]}}},   # inside the back wall (z = zm + 4)
    # v4: a Timed Up and Go (sample_v4/, session HC903, task TUG): hc142's repetitions of sit, stand, walk 3 m, turn,
    # walk back, turn and sit, with a chair at the start; v1's room, the along camera facing the chair, no co-walker. Short walks, a
    # large sit-to-stand excursion and a seated start are what a TUG asks of the selection and the sync. (hc100's TUG,
    # used first, has its left ankle 163 deg off in the inertial fit: a visibly wrong foot in the rendered truth.)
    "v4": {"sid": "HC903", "subdir": "sample_v4", "decoy": False, "subject": "hc142_tug", "task": {"csv": "tug", "dir": "TUG", "token": "TUG"}, "chair": True,
           "seat": lambda xm, zm: [xm - 0.35, 0.0, zm - 1.55],
           # the along camera faces the chair from beyond the turn-around point, as a TUG is usually filmed
           "cams": lambda x0, x1, xm, zm: {"left": {"W": LW, "H": LH, "yfov": 50.0, "eye": [x1 + 3.0, 1.0, zm + 0.1], "target": [x1 - 2.0, 0.5, zm]},
                                           "right": {"W": RW, "H": RC, "yfov": 30.0, "eye": [xm + 1.0, 2.0, zm - 4.0], "target": [xm - 0.4, 0.5, zm + 0.4]}}},
    # v5: a hurried-pace walk (sample_v5/, session HC904, task HP): hc105 at speed, with hc100 as the co-walker. The
    # along view is v2's low camera past the near end (feet and mat, head cut when close); the side view is laid out
    # like the real session's: elevated (2.2 m), 7 m back, wide (40 deg vertical, 66 deg across), the whole room and
    # mat in frame with the operators' desk in the foreground, so the walker is never out of this view and leaves
    # it only at the top edge, if at all. (The first render used v2's narrow side camera, which lost the walker for
    # 58 % of the recording; the real side view sees the entire room.)
    "v5": {"sid": "HC904", "subdir": "sample_v5", "decoy": True, "subject": "hc105_hurriedpace", "decoy_name": "hc100_hurriedpace", "decoy_dz": 1.5, "task": {"csv": "hurriedpace", "dir": "HP", "token": "HP"},
           "seat": lambda xm, zm: [xm - 2.0, 0.0, zm - 3.2], "bystander": lambda x0, x1, xm, zm: [x1 + 1.5, 0.0, zm + 2.5],
           "cams": lambda x0, x1, xm, zm: {"left": {"W": LW, "H": LH, "yfov": 60.0, "eye": [x0 - 0.5, 0.6, zm + 0.05], "target": [x0 + 1.2, 0.0, zm]},
                                           "right": {"W": RW, "H": RC, "yfov": 40.0, "eye": [xm, 2.2, zm - 7.0], "target": [xm, 0.0, zm]}}},
    # v6: a self-paced walk ON the instrumented mat (sample_v6/, session HC905, task SPM = selfpace_mat) with the walkway
    # columns of the recording CSV FABRICATED from the rendered truth in the export's own form: at every IMU sample the
    # loaded cells of the mat (Walkway_X/Y cell indices, WalkwayFoot labels, WalkwayPressureLevel) under each foot that
    # is on it, on a 384 x 48 grid of 1.27 cm cells centred on the walk. (The real recording's own walkway rows are
    # NOT used: the render walks the neutral SMPL body along a stance-pinned path 22 % shorter than the real one, so
    # the real footprints would contradict the video.) v3's room.
    "v6": {"sid": "HC905", "subdir": "sample_v6", "decoy": False, "subject": "hc105_selfpace_mat", "task": {"csv": "selfpace_mat", "dir": "SPM", "token": "SPM"}, "walkway_from_truth": True,
           "seat": lambda xm, zm: [xm + 0.6, 0.0, zm + 2.2], "bystander": lambda x0, x1, xm, zm: [xm - 1.0, 0.0, zm - 2.0],
           "cams": lambda x0, x1, xm, zm: {"left": {"W": LW, "H": LH, "yfov": 50.0, "eye": [x1 + 1.3, 1.5, zm - 0.15], "target": [x1 - 2.0, 0.4, zm]},
                                           "right": {"W": RW, "H": RC, "yfov": 40.0, "eye": [xm - 1.6, 2.2, zm + 3.5], "target": [xm + 1.0, 0.5, zm - 0.3]}}},
}


MAT_CELL_M = 0.0127; MAT_CELLS = (384, 48)        # the GAITRite grid: 4.88 x 0.61 m


def recording_csv_subset(var: dict, subj_name: str, J_s: np.ndarray, T_s: int, x0: float, x1: float, xm: float, zm: float):
    """The session's recording CSV: a column subset of the public recording with the walkway columns EMPTIED (the
    real footprints describe the real walk, 22 % longer than the rendered neutral-body one) or, for the mat variant,
    fabricated from the rendered truth in the export's own form. Returns (DataFrame, mat_frame | None); mat_frame
    gives the world position of cell (0, 0), the cell size and the grid, X along the walk (+x), Y across (+z)."""
    import pandas as pd
    cols = ["Time", "GeneralEvent", "ClinicalEvent", "L Foot Contact", "R Foot Contact", "Walkway_X", "Walkway_Y", "WalkwayPressureLevel", "WalkwayFoot",
            "LowerBack_FreeAcc_E", "LowerBack_FreeAcc_N", "LowerBack_FreeAcc_U"]
    dfc = pd.read_csv(IMU / f"{subj_name}.csv", usecols=lambda c: c in cols, low_memory=False)
    for c in ("Walkway_X", "Walkway_Y", "WalkwayPressureLevel", "WalkwayFoot"):
        dfc[c] = ""
    dfc = dfc[[c for c in cols if c in dfc.columns]]
    if not var.get("walkway_from_truth"):
        return dfc, None
    # the mat: 384 x 48 cells of 1.27 cm centred on the walk; a foot on it loads an ellipse of cells 24 x 9 cm around
    # the rendered truth's foot centre (midpoint of ankle and toe joints) for every sample its contact flag is on,
    # heel-weighted early in the stance and toe-weighted late, pressure levels 2-9
    tcsv_s = np.array([float(str(v).split()[0]) for v in dfc["Time"]]); n = len(dfc)
    mat_frame = {"x0": float(xm - 0.5 * MAT_CELLS[0] * MAT_CELL_M), "z0": float(zm - 0.5 * MAT_CELLS[1] * MAT_CELL_M), "cell_m": MAT_CELL_M, "cells": list(MAT_CELLS),
                 "about": "cell (0,0) corner in the render's world; Walkway_X = (x - x0) / cell_m along the walk, Walkway_Y = (z - z0) / cell_m across"}
    X = [""] * n; Y = [""] * n; P = [""] * n; F = [""] * n; rng_w = np.random.default_rng(1); n_rows = 0
    for col, lab, ja, jf in (("L Foot Contact", "L", 7, 10), ("R Foot Contact", "R", 8, 11)):
        c = np.nan_to_num(pd.to_numeric(dfc[col], errors="coerce").to_numpy(float)) > 0.5
        edges = np.flatnonzero(np.diff(np.r_[0, c.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            for i in range(a, b):
                fi = int(round(tcsv_s[i] * FPS))
                if not (0 <= fi < T_s):
                    continue
                cx, cz = 0.5 * (J_s[fi, ja, [0, 2]] + J_s[fi, jf, [0, 2]]); heading = J_s[fi, jf, [0, 2]] - J_s[fi, ja, [0, 2]]
                u = heading / max(np.linalg.norm(heading), 1e-6); v = np.array([-u[1], u[0]]); phase = (i - a) / max(b - a - 1, 1)
                centre = np.array([cx, cz]) + (0.06 * (phase - 0.5)) * u                      # heel to toe roll-over
                gx = (centre[0] - mat_frame["x0"]) / MAT_CELL_M; gz = (centre[1] - mat_frame["z0"]) / MAT_CELL_M
                xs, ys, ps = [], [], []
                for dx in range(-12, 13):
                    for dz in range(-12, 13):
                        q = np.array([dx, dz]) * MAT_CELL_M; along = q @ u; across = q @ v
                        if (along / 0.12) ** 2 + (across / 0.045) ** 2 <= 1.0:
                            ix, iz = int(round(gx)) + dx, int(round(gz)) + dz
                            if 0 <= ix < MAT_CELLS[0] and 0 <= iz < MAT_CELLS[1]:
                                xs.append(ix); ys.append(iz); ps.append(int(np.clip(round(6 - 20 * abs(across) - 10 * abs(along - 0.06 * (phase - 0.5)) + rng_w.normal(0, 1.0)), 2, 9)))
                if len(xs) >= 5:
                    X[i] = (X[i] + "|" if X[i] else "") + "|".join(map(str, xs)); Y[i] = (Y[i] + "|" if Y[i] else "") + "|".join(map(str, ys))
                    P[i] = (P[i] + "|" if P[i] else "") + "|".join(map(str, ps)); F[i] = (F[i] + "|" if F[i] else "") + "|".join([lab] * len(xs)); n_rows += 1
    dfc["Walkway_X"] = X; dfc["Walkway_Y"] = Y; dfc["WalkwayPressureLevel"] = P; dfc["WalkwayFoot"] = F
    print(f"walkway cells fabricated: {n_rows} foot-samples on a {MAT_CELLS[0]}x{MAT_CELLS[1]} grid of {100 * MAT_CELL_M} cm; mat x {mat_frame['x0']:.2f}..{mat_frame['x0'] + MAT_CELLS[0] * MAT_CELL_M:.2f}, walk {x0:.2f}..{x1:.2f}")
    return dfc, mat_frame


def refresh_csv(variant: str, out_dir: str | None = None):
    """Rewrite only the recording CSV (and the mat frame in expected.json) of an already rendered sample."""
    var = VARIANTS[variant]; sid = var["sid"]; subj_name = var.get("subject", "hc100_selfpace")
    task = var.get("task", {"csv": "selfpace", "dir": "SP", "token": "SP"}); tcsv = task["csv"]
    _smplx_compat_shims(); import smplx
    out = Path(out_dir) if out_dir else PKG / var["subdir"]; sess = out / "data" / sid
    smpl = smplx.create(BM, model_type="smpl", gender="neutral", batch_size=1)
    subj = yaw_align(load_ik_aa(subj_name), smpl); tr_s = root_trajectory(subj, smpl, f"{subj_name}.csv"); T_s = len(subj)
    x0, x1 = float(tr_s[:, 0].min()), float(tr_s[:, 0].max()); xm = 0.5 * (x0 + x1); zm = float(tr_s[:, 2].mean())
    _, J_s = fk(subj, smpl, tr_s)
    G = np.load(out / f"{sid}_gt.npz")
    assert np.allclose(G["transl"], tr_s, atol=1e-4) and len(G["pose_aa"]) == T_s, "the rendered truth differs from this build: re-render instead"
    dfc, mat_frame = recording_csv_subset(var, subj_name, J_s, T_s, x0, x1, xm, zm)
    dfc.to_csv(sess / f"{sid.lower()}_{tcsv}.csv", index=False)
    ej = out / "expected.json"; exp = json.loads(ej.read_text()); exp["mat_frame"] = mat_frame; ej.write_text(json.dumps(exp))
    print(f"{sid}: CSV rewritten ({len(dfc)} rows); mat_frame {mat_frame}")


def main(out_dir: str | None = None, variant: str = "v1"):
    var = VARIANTS[variant]; sid = var["sid"]; has_decoy = var["decoy"]
    subj_name = var.get("subject", "hc100_selfpace"); deco_name = var.get("decoy_name", "hc102_selfpace"); decoy_dz = var.get("decoy_dz", 1.5)
    task = var.get("task", {"csv": "selfpace", "dir": "SP", "token": "SP"}); tdir, ttok, tcsv = task["dir"], task["token"], task["csv"]
    _smplx_compat_shims(); os.environ.pop("PYOPENGL_PLATFORM", None)
    import smplx, trimesh, pyrender, imageio.v3 as iio
    from scipy.spatial.transform import Rotation as Rt
    out = Path(out_dir) if out_dir else PKG / var["subdir"]
    sess = out / "data" / sid; (sess / tdir).mkdir(parents=True, exist_ok=True)
    smpl = smplx.create(BM, model_type="smpl", gender="neutral", batch_size=1); faces = smpl.faces.astype(np.int32)
    rng = np.random.default_rng(0)

    subj = yaw_align(load_ik_aa(subj_name), smpl); tr_s = root_trajectory(subj, smpl, f"{subj_name}.csv"); T_s = len(subj)
    if has_decoy:
        deco = yaw_align(load_ik_aa(deco_name), smpl); tr_d = root_trajectory(deco, smpl, f"{deco_name}.csv") + np.array([0, 0, decoy_dz], np.float32); delay_d = int(0.8 * FPS)
    byst = yaw_align(load_ik_aa("hc101_selfpace"), smpl)[:45]; byst = np.concatenate([byst, byst[::-1]]); tr_b = root_trajectory(byst, smpl)
    x0, x1 = float(tr_s[:, 0].min()), float(tr_s[:, 0].max()); xm = 0.5 * (x0 + x1); zm = float(tr_s[:, 2].mean())
    tr_b = tr_b + np.array(var["bystander"](x0, x1, xm, zm) if "bystander" in var else [x1 + 2.2, 0.0, zm + 1.2], np.float32)
    pre = int(3.0 * FPS); post = int(1.5 * FPS); n_frames = pre + T_s + post
    seat = seated_sequence(n_frames, rng)
    V_s, J_s = fk(subj, smpl, tr_s); V_b, J_b = fk(byst, smpl, tr_b)
    if has_decoy:
        V_d, J_d = fk(deco, smpl, tr_d)
    # cameras: along-walkway = low at the x0 end looking down the mat (bodies truncate at the top when close);
    # side = oblique, elevated, narrow field so it sees only the middle of the corridor (exits at both ends)
    cams = var["cams"](x0, x1, xm, zm)
    # the seated operator sits close to the side camera, lower right of its view, facing the room (v2: beside the far
    # end of the mat instead, outside the side view, so no part of the walk is occluded in the reconstruction sample)
    seat_pos = np.array(var["seat"](xm, zm), np.float32)
    seat_rot = Rt.from_euler("y", np.deg2rad(25))
    seat[:, :3] = np.tile((seat_rot * Rt.from_rotvec(seat[0, :3])).as_rotvec().astype(np.float32), (n_frames, 1))
    V_o, J_o = fk(seat, smpl, np.tile(seat_pos, (n_frames, 1)) + np.array([0, 0.45, 0], np.float32))
    renderer = {cid: pyrender.OffscreenRenderer(viewport_width=c["W"], viewport_height=c["H"]) for cid, c in cams.items()}
    scenes, cam_poses, intr = {}, {}, {}
    for cid, c in cams.items():
        yfov = np.deg2rad(c["yfov"]); fy = 0.5 * c["H"] / np.tan(yfov / 2); intr[cid] = (fy, fy, c["W"] / 2, c["H"] / 2)
        cam_poses[cid] = look_at(c["eye"], c["target"])
        sc = pyrender.Scene(bg_color=(0.86, 0.86, 0.88, 1.0), ambient_light=(0.4, 0.4, 0.42))
        sc.add(pyrender.PerspectiveCamera(yfov=yfov, znear=0.1, zfar=60.0), pose=cam_poses[cid])
        key = np.eye(4); key[:3, 3] = (2.0, 4.0, -3.0); sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=key)
        fill = np.eye(4); fill[:3, 3] = (-2.0, 3.0, 3.0); sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.2), pose=fill)
        fs = 16.0; ff = np.array([[0, 1, 2], [0, 2, 3]])
        fv = np.array([[x0 - fs, 0, zm - fs], [x1 + fs, 0, zm - fs], [x1 + fs, 0, zm + fs], [x0 - fs, 0, zm + fs]], np.float32)
        fl = trimesh.Trimesh(fv, ff, process=False); fl.visual = trimesh.visual.ColorVisuals(mesh=fl, face_colors=np.tile([150, 140, 125, 255], (2, 1)))
        sc.add(pyrender.Mesh.from_trimesh(fl, smooth=False))
        # floor tiles (checker) for texture, a dark walkway mat along the path, two walls, a desk in front of the operator
        tiles = []
        for ix in range(-6, 14):
            for iz in range(-8, 8):
                if (ix + iz) % 2:
                    continue
                cx_, cz_ = x0 - 3 + ix * 0.9, zm + iz * 0.9
                tiles.append(np.array([[cx_, 0.002, cz_], [cx_ + 0.9, 0.002, cz_], [cx_ + 0.9, 0.002, cz_ + 0.9], [cx_, 0.002, cz_ + 0.9]], np.float32))
        tv = np.concatenate(tiles); tf = np.concatenate([ff + 4 * i for i in range(len(tiles))])
        tl = trimesh.Trimesh(tv, tf, process=False); tl.visual = trimesh.visual.ColorVisuals(mesh=tl, face_colors=np.tile([128, 120, 108, 255], (len(tf), 1)))
        sc.add(pyrender.Mesh.from_trimesh(tl, smooth=False))
        mv = np.array([[x0 - 0.8, 0.006, zm - 0.5], [x1 + 0.8, 0.006, zm - 0.5], [x1 + 0.8, 0.006, zm + 0.5], [x0 - 0.8, 0.006, zm + 0.5]], np.float32)
        mt = trimesh.Trimesh(mv, ff, process=False); mt.visual = trimesh.visual.ColorVisuals(mesh=mt, face_colors=np.tile([45, 42, 44, 255], (2, 1)))
        sc.add(pyrender.Mesh.from_trimesh(mt, smooth=False))
        for wv in (np.array([[x1 + 4.0, 0, zm - 8], [x1 + 4.0, 3.0, zm - 8], [x1 + 4.0, 3.0, zm + 8], [x1 + 4.0, 0, zm + 8]], np.float32),
                   np.array([[x0 - 8, 0, zm + 4.0], [x0 - 8, 3.0, zm + 4.0], [x1 + 8, 3.0, zm + 4.0], [x1 + 8, 0, zm + 4.0]], np.float32)):
            wl = trimesh.Trimesh(wv, np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]]), process=False)
            wl.visual = trimesh.visual.ColorVisuals(mesh=wl, face_colors=np.tile([205, 203, 198, 255], (4, 1)))
            sc.add(pyrender.Mesh.from_trimesh(wl, smooth=False))
        desk = trimesh.creation.box(extents=(1.8, 0.05, 0.7)); desk.apply_translation([seat_pos[0], 0.74, seat_pos[2] - 0.75])
        desk.visual = trimesh.visual.ColorVisuals(mesh=desk, face_colors=np.tile([150, 120, 90, 255], (len(desk.faces), 1)))
        sc.add(pyrender.Mesh.from_trimesh(desk, smooth=False))
        if var.get("chair"):                 # a chair under the walker's seated start (TUG): seat pan, four legs, a back behind the hips
            hx, hz = float(tr_s[0, 0]), float(tr_s[0, 2]); seat_h = float(J_s[0, 0, 1]) - 0.08
            parts = [trimesh.creation.box(extents=(0.46, 0.05, 0.46)).apply_translation([hx, seat_h, hz]),
                     trimesh.creation.box(extents=(0.46, 0.45, 0.05)).apply_translation([hx - 0.22 if tr_s[30, 0] > hx else hx + 0.22, seat_h + 0.25, hz]).apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0], [hx, 0, hz]))]
            for sx in (-0.2, 0.2):
                for sz in (-0.2, 0.2):
                    parts.append(trimesh.creation.box(extents=(0.04, seat_h, 0.04)).apply_translation([hx + sx, seat_h / 2, hz + sz]))
            for prt in parts:
                prt.visual = trimesh.visual.ColorVisuals(mesh=prt, face_colors=np.tile([90, 70, 60, 255], (len(prt.faces), 1))); sc.add(pyrender.Mesh.from_trimesh(prt, smooth=False))
        scenes[cid] = sc
    colors = {"s": [140, 150, 180], "d": [180, 150, 140], "b": [150, 175, 150], "o": [120, 120, 160]}
    writer = iio.imopen(str(sess / tdir / f"{sid}_{ttok}_DeID.mp4"), "w", plugin="pyav"); writer.init_video_stream("h264", fps=FPS)
    gt = {cid: {"frames": [], "hip": [], "vis": [], "decoy": [], "joints2d": [], "joints_in": []} for cid in cams}
    nodes = {cid: [] for cid in cams}
    FW, FH = LW + RW, LH
    light = [LW + RW - 70, 24, 30]
    for f in range(n_frames):
        t = f - pre; td = (t - delay_d) if has_decoy else -1
        bodies = []
        if 0 <= t < T_s:
            bodies.append(("s", V_s[t], J_s[t]))
        if has_decoy and 0 <= td < len(V_d):
            bodies.append(("d", V_d[td], J_d[td]))
        bodies.append(("b", V_b[f % len(V_b)], J_b[f % len(V_b)])); bodies.append(("o", V_o[f], J_o[f]))
        frame = np.zeros((FH, FW, 3), np.uint8)
        for cid, c in cams.items():
            sc = scenes[cid]
            for nd in nodes[cid]:
                sc.remove_node(nd)
            nodes[cid] = []
            for tag, v, j in bodies:
                tm = trimesh.Trimesh(v, faces, process=False); tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=np.tile(colors[tag] + [255], (len(v), 1)))
                nodes[cid].append(sc.add(pyrender.Mesh.from_trimesh(tm, smooth=True)))
            color, _ = renderer[cid].render(sc, flags=pyrender.RenderFlags.SKIP_CULL_FACES)   # the floor, tiles and mat are single-sided quads seen from above
            img = color[..., ::-1].copy()      # BGR for cv2-style tiling? we write RGB below; keep RGB order consistent
            img = color.copy()
            fx, fy, cx, cy = intr[cid]
            if cid == "left":
                frame[0:LH, 0:LW] = img
                off = (0, 0)
            else:
                frame[0:RC, LW:LW + RW] = img; off = (LW, 0)
                lx, ly, ls = light; frame[ly:ly + ls, lx:lx + ls] = (245, 240, 200) if 0 <= t < T_s else (40, 40, 44)
            if 0 <= t < T_s:
                pts3 = np.stack([J_s[t, [1, 2]].mean(0), J_s[t, 15], J_s[t, 7], J_s[t, 8], J_s[t, 16], J_s[t, 17]])   # hip, head, ankles, shoulders
                p2, z = project(pts3, cam_poses[cid], fx, fy, cx, cy)
                inside = (z > 0.2) & (p2[:, 0] >= 0) & (p2[:, 0] < c["W"]) & (p2[:, 1] >= 0) & (p2[:, 1] < c["H"])
                vis = 2 if inside.all() else (1 if inside[[0, 2, 3]].any() or inside[[4, 5]].any() else 0)
                if vis > 0 or inside[0]:
                    gt[cid]["frames"].append(f); gt[cid]["hip"].append([float(p2[0, 0] + off[0]), float(p2[0, 1] + off[1])]); gt[cid]["vis"].append(int(vis))
                else:
                    gt[cid]["vis"].append(0)
                # all 24 SMPL joints: image position (full-frame px) and whether each is inside this view
                pj, zj = project(J_s[t], cam_poses[cid], fx, fy, cx, cy)
                inj = (zj > 0.2) & (pj[:, 0] >= 0) & (pj[:, 0] < c["W"]) & (pj[:, 1] >= 0) & (pj[:, 1] < c["H"])
                gt[cid]["joints2d"].append((pj + np.array(off)).astype(np.float32)); gt[cid]["joints_in"].append(inj)
            if has_decoy and 0 <= td < len(V_d) and f % 10 == 0:
                p2, z = project(J_d[td, [1, 2]].mean(0, keepdims=True), cam_poses[cid], fx, fy, cx, cy)
                gt[cid]["decoy"].append([f, float(p2[0, 0] + off[0]), float(p2[0, 1] + off[1])])
        frame = np.clip(frame.astype(np.int16) + rng.normal(0, 2.5, frame.shape).astype(np.int16), 0, 255).astype(np.uint8)
        writer.write_frame(frame)
    writer.close()
    for r in renderer.values():
        r.delete()
    dfc, mat_frame = recording_csv_subset(var, subj_name, J_s, T_s, x0, x1, xm, zm)
    dfc.to_csv(sess / f"{sid.lower()}_{tcsv}.csv", index=False)
    (sess / "README.txt").write_text(f"Synthetic session {sid} (rendered SMPL meshes, no real video): one combined two-view video\n"
                                     f"{tdir}/{sid}_{ttok}_DeID.mp4 (left = along-walkway view 520x600, right = side view 900x500 letterboxed)" + (" with a co-walker,\n" if has_decoy else ",\n") +
                                     f"a seated operator near the side camera and a bystander; {sid.lower()}_{tcsv}.csv = column subset of the public {subj_name}.csv"
                                     + (" with the walkway columns fabricated from the rendered truth in the export's form (loaded cells per sample, 1.27 cm grid).\n" if mat_frame else "; its walkway columns are left empty (the real recording's footprints do not describe the rendered walk).\n"))
    # ground truth of the walker for the reconstruction experiment: SMPL pose, root trajectory, 3D joints, and per
    # view the image position and in-frame flag of every joint (no pixels)
    np.savez_compressed(out / f"{sid}_gt.npz", frame_idx=np.arange(pre, pre + T_s), imu_t_s=np.arange(T_s) / FPS, pose_aa=subj.astype(np.float32), transl=tr_s.astype(np.float32),
                        joints3d=J_s.astype(np.float32), betas=np.zeros(10, np.float32), fps=FPS, offset_s=pre / FPS,
                        **{f"{cid}_joints2d": np.stack(gt[cid]["joints2d"]) for cid in cams}, **{f"{cid}_joints_in": np.stack(gt[cid]["joints_in"]) for cid in cams},
                        **{f"{cid}_cam_pose": cam_poses[cid].astype(np.float32) for cid in cams}, **{f"{cid}_intrinsics": np.array(intr[cid], np.float32) for cid in cams},
                        **{f"{cid}_panel": np.array([0, 0, LW, LH] if cid == "left" else [LW, 0, LW + RW, RC]) for cid in cams},
                        note="SMPL (24 joints, axis-angle, neutral shape) in the render's world frame (y up); joints2d in full-frame pixels of the combined video")
    expected = {"session_id": sid, "recording_id": f"{sid.lower()}_{tcsv}", "video": f"{tdir}/{sid}_{ttok}_DeID.mp4", "task": task, "mat_frame": mat_frame, "fps": FPS, "n_frames": n_frames, "variant": variant, "gt_npz": f"{sid}_gt.npz",
                "offset_s": pre / FPS, "offset_frame": pre, "light_rect": light[:2] + [light[2], light[2]], "imu_duration_s": T_s / FPS, "n_frames_in_recording": T_s,
                "panels": {"left": [0, 0, LW, LH], "right": [LW, 0, LW + RW, RC]}, "layout": "horizontal", "seam": LW,
                "views": {cid: {"gt_frames": gt[cid]["frames"], "gt_hip_px": gt[cid]["hip"], "n_visible": len(gt[cid]["frames"]),
                                "vis_counts": {str(k): int(sum(1 for v in gt[cid]["vis"] if v == k)) for k in (0, 1, 2)}, "decoy_hip_px": gt[cid]["decoy"]} for cid in cams},
                "scene": {"subject": f"{subj_name} IK (public WearGait); pelvis vertical motion = double-integrated LowerBack_FreeAcc_U of the same recording", "decoy": (f"{deco_name} {decoy_dz:+.1f} m, +0.8 s (its own IMU vertical motion)" if has_decoy else None), "operator": "seated near side camera", "bystander": "hc101 standing loop",
                          "layout": {"mat_x": [float(x0 - 0.8), float(x1 + 0.8)], "mat_z": [float(zm - 0.5), float(zm + 0.5)], "operator_seat_xyz": [float(v) for v in seat_pos], "operator_yaw_deg": 25.0,
                                     "desk_center_xyz": [float(seat_pos[0]), 0.74, float(seat_pos[2] - 0.75)], "desk_extents_xyz": [1.8, 0.05, 0.7],
                                     "bystander_xyz": [float(tr_b[0, 0]), 0.0, float(tr_b[0, 2])], "walls": [{"x": float(x1 + 4.0), "z": [float(zm - 8), float(zm + 8)]}, {"z": float(zm + 4.0), "x": [float(x0 - 8), float(x1 + 8)]}],
                                     "cameras": {cid: {"eye": [float(v) for v in c["eye"]], "target": [float(v) for v in c["target"]], "yfov_deg": float(c["yfov"])} for cid, c in cams.items()}},
                          "walk_extent_m": [x0, x1]}}
    (out / "expected.json").write_text(json.dumps(expected))
    for cid in cams:
        print(f"{cid}: subject frames with hip in view {len(gt[cid]['frames'])}/{T_s}, vis counts {expected['views'][cid]['vis_counts']}")
    print(f"sample -> {out}: " + ", ".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in (sess / tdir).iterdir()))
    return 0


if __name__ == "__main__" and "--csv-only" in sys.argv:
    args = [a for a in sys.argv[1:] if a != "--csv-only"]
    var_ = args[args.index("--variant") + 1] if "--variant" in args else "v1"
    refresh_csv(var_, next((a for a in args if not a.startswith("--") and a != var_), None)); sys.exit(0)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None); ap.add_argument("--variant", choices=sorted(VARIANTS), default="v1")
    a = ap.parse_args(); sys.exit(main(a.out, a.variant))
