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

    python tools/make_sample.py [--variant v1|v2]   (needs smplx, trimesh, pyrender, imageio[pyav], pandas and the inputs named below)
v2 (sample_v2/, session HC901) renders the same walk from two deliberately compromised views, after the real
session layout: the along view loses everything above the hips when the walker is close, the side view never
shows the head; the walker's ground-truth SMPL pose and 3D joints are saved beside it for the two-view
reconstruction experiment (tools/two_view_fusion.py).
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
    "v1": {"sid": "HC900", "subdir": "sample", "decoy": True,
           "cams": lambda x0, x1, xm, zm: {"left": {"W": LW, "H": LH, "yfov": 55.0, "eye": [x0 - 1.6, 0.9, zm + 0.1], "target": [x0 + 2.2, 0.0, zm]},
                                           "right": {"W": RW, "H": RC, "yfov": 30.0, "eye": [xm + 1.0, 2.0, zm - 4.0], "target": [xm - 0.4, 0.5, zm + 0.4]}}},
    # v2: two deliberately compromised views of one walk, after the real session layout. Along view: camera 0.9 m
    # beyond the near end, 0.75 m high, wide: at the near end only the legs are in frame (the pose is lost), at the
    # far end the body is a tenth of the frame. Side view: low and narrow, aimed at hip height: the head is out of
    # frame for the whole pass and the shoulders sit at the top edge; the subject exits at both ends. No co-walker:
    # the point of this sample is reconstruction from two truncated views, and the ground-truth 3D pose is saved.
    "v2": {"sid": "HC901", "subdir": "sample_v2", "decoy": False,
           "cams": lambda x0, x1, xm, zm: {"left": {"W": LW, "H": LH, "yfov": 60.0, "eye": [x0 - 0.9, 0.75, zm + 0.05], "target": [x0 + 1.5, 0.3, zm]},
                                           "right": {"W": RW, "H": RC, "yfov": 22.0, "eye": [xm, 1.0, zm - 3.5], "target": [xm, 0.75, zm]}}},
}


def main(out_dir: str | None = None, variant: str = "v1"):
    var = VARIANTS[variant]; sid = var["sid"]; has_decoy = var["decoy"]
    _smplx_compat_shims(); os.environ.pop("PYOPENGL_PLATFORM", None)
    import smplx, trimesh, pyrender, imageio.v3 as iio
    from scipy.spatial.transform import Rotation as Rt
    out = Path(out_dir) if out_dir else PKG / var["subdir"]
    sess = out / "data" / sid; (sess / "SP").mkdir(parents=True, exist_ok=True)
    smpl = smplx.create(BM, model_type="smpl", gender="neutral", batch_size=1); faces = smpl.faces.astype(np.int32)
    rng = np.random.default_rng(0)

    subj = yaw_align(load_ik_aa("hc100_selfpace"), smpl); tr_s = root_trajectory(subj, smpl, "hc100_selfpace.csv"); T_s = len(subj)
    if has_decoy:
        deco = yaw_align(load_ik_aa("hc102_selfpace"), smpl); tr_d = root_trajectory(deco, smpl, "hc102_selfpace.csv") + np.array([0, 0, 1.5], np.float32); delay_d = int(0.8 * FPS)
    byst = yaw_align(load_ik_aa("hc101_selfpace"), smpl)[:45]; byst = np.concatenate([byst, byst[::-1]]); tr_b = root_trajectory(byst, smpl)
    x0, x1 = float(tr_s[:, 0].min()), float(tr_s[:, 0].max()); xm = 0.5 * (x0 + x1); zm = float(tr_s[:, 2].mean())
    tr_b = tr_b + np.array([x1 + 2.2, 0.0, zm + 1.2], np.float32)
    pre = int(3.0 * FPS); post = int(1.5 * FPS); n_frames = pre + T_s + post
    seat = seated_sequence(n_frames, rng)
    V_s, J_s = fk(subj, smpl, tr_s); V_b, J_b = fk(byst, smpl, tr_b)
    if has_decoy:
        V_d, J_d = fk(deco, smpl, tr_d)
    # cameras: along-walkway = low at the x0 end looking down the mat (bodies truncate at the top when close);
    # side = oblique, elevated, narrow field so it sees only the middle of the corridor (exits at both ends)
    cams = var["cams"](x0, x1, xm, zm)
    # the seated operator sits close to the side camera (image right = -x), lower right of its view, facing the room
    seat_pos = np.array([xm - 0.35, 0.0, zm - 1.55], np.float32)
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
        scenes[cid] = sc
    colors = {"s": [140, 150, 180], "d": [180, 150, 140], "b": [150, 175, 150], "o": [120, 120, 160]}
    writer = iio.imopen(str(sess / "SP" / f"{sid}_SP_DeID.mp4"), "w", plugin="pyav"); writer.init_video_stream("h264", fps=FPS)
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
            color, _ = renderer[cid].render(sc)
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
    import pandas as pd
    cols = ["Time", "GeneralEvent", "ClinicalEvent", "L Foot Contact", "R Foot Contact", "Walkway_X", "Walkway_Y", "LowerBack_FreeAcc_E", "LowerBack_FreeAcc_N", "LowerBack_FreeAcc_U"]
    pd.read_csv(IMU / "hc100_selfpace.csv", usecols=lambda c: c in cols).to_csv(sess / f"{sid.lower()}_selfpace.csv", index=False)
    (sess / "README.txt").write_text(f"Synthetic session {sid} (rendered SMPL meshes, no real video): one combined two-view video\n"
                                     f"SP/{sid}_SP_DeID.mp4 (left = along-walkway view 520x600, right = side view 900x500 letterboxed)" + (" with a co-walker,\n" if has_decoy else ",\n") +
                                     f"a seated operator near the side camera and a bystander; {sid.lower()}_selfpace.csv = column subset of the public hc100_selfpace.csv.\n")
    # ground truth of the walker for the reconstruction experiment: SMPL pose, root trajectory, 3D joints, and per
    # view the image position and in-frame flag of every joint (no pixels)
    np.savez_compressed(out / f"{sid}_gt.npz", frame_idx=np.arange(pre, pre + T_s), imu_t_s=np.arange(T_s) / FPS, pose_aa=subj.astype(np.float32), transl=tr_s.astype(np.float32),
                        joints3d=J_s.astype(np.float32), betas=np.zeros(10, np.float32), fps=FPS, offset_s=pre / FPS,
                        **{f"{cid}_joints2d": np.stack(gt[cid]["joints2d"]) for cid in cams}, **{f"{cid}_joints_in": np.stack(gt[cid]["joints_in"]) for cid in cams},
                        **{f"{cid}_cam_pose": cam_poses[cid].astype(np.float32) for cid in cams}, **{f"{cid}_intrinsics": np.array(intr[cid], np.float32) for cid in cams},
                        **{f"{cid}_panel": np.array([0, 0, LW, LH] if cid == "left" else [LW, 0, LW + RW, RC]) for cid in cams},
                        note="SMPL (24 joints, axis-angle, neutral shape) in the render's world frame (y up); joints2d in full-frame pixels of the combined video")
    expected = {"session_id": sid, "recording_id": f"{sid.lower()}_selfpace", "video": f"SP/{sid}_SP_DeID.mp4", "fps": FPS, "n_frames": n_frames, "variant": variant, "gt_npz": f"{sid}_gt.npz",
                "offset_s": pre / FPS, "offset_frame": pre, "light_rect": light[:2] + [light[2], light[2]], "imu_duration_s": T_s / FPS, "n_frames_in_recording": T_s,
                "panels": {"left": [0, 0, LW, LH], "right": [LW, 0, LW + RW, RC]}, "layout": "horizontal", "seam": LW,
                "views": {cid: {"gt_frames": gt[cid]["frames"], "gt_hip_px": gt[cid]["hip"], "n_visible": len(gt[cid]["frames"]),
                                "vis_counts": {str(k): int(sum(1 for v in gt[cid]["vis"] if v == k)) for k in (0, 1, 2)}, "decoy_hip_px": gt[cid]["decoy"]} for cid in cams},
                "scene": {"subject": "hc100_selfpace IK (public WearGait); pelvis vertical motion = double-integrated LowerBack_FreeAcc_U of the same recording", "decoy": ("hc102_selfpace +1.5 m, +0.8 s (its own IMU vertical motion)" if has_decoy else None), "operator": "seated near side camera", "bystander": "hc101 standing loop",
                          "walk_extent_m": [x0, x1]}}
    (out / "expected.json").write_text(json.dumps(expected))
    for cid in cams:
        print(f"{cid}: subject frames with hip in view {len(gt[cid]['frames'])}/{T_s}, vis counts {expected['views'][cid]['vis_counts']}")
    print(f"sample -> {out}: " + ", ".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in (sess / 'SP').iterdir()))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None); ap.add_argument("--variant", choices=sorted(VARIANTS), default="v1")
    a = ap.parse_args(); sys.exit(main(a.out, a.variant))
