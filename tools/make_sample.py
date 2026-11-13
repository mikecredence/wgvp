"""(Analysis side only.) Build the bundled dry-run sample as ONE combined two-view video shaped like the real WearGait
session videos (unequal panels, letterboxed side view, seated operator close to the camera, exits from one view).

Scene (rendered SMPL meshes, no real video):
  subject   WearGait hc100_selfpace (public) IK pose sequence walking a corridor (stance-foot-pinned root trajectory)
  co-walker hc102_selfpace, 1.5 m beside the subject, starting 0.8 s later (a distractor WITH gait)
  operator  a seated body close to the side camera, lower right, fidgeting (large in frame, no gait)
  bystander hc101's standing frames looped at the far wall
  light     a small square in the side panel, bright while the IMU records (optional cross-check only)
Panels: LEFT = along-walkway view 520x600 (camera at the walkway end, low: the head leaves the frame when close);
        RIGHT = the room-wide side view 900x375 letterboxed in 900x600 (black band below): elevated, 4 m from the
        mat with the operators' desk in the foreground, the walker's head outside its top edge on the mat, and the
        walker out of it only beyond the mat's ends. Combined frame 1420x600, 30 fps.
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
RW, RH, RC = 900, 600, 375   # side panel (content height RC, letterboxed to RH): the real side view is a wide strip, 2.4:1
MAT_L, MAT_W = 4.88, 0.61    # the instrumented mat: 16 x 2 ft (GAITRite), centred on the walk; self-paced and hurried walks step off its ends to turn

# The two cameras of a real session (from the Video Labeler frame of NLS036): the SIDE view is elevated (~2.3 m), ~4 m
# from the mat's centre line (the mat spans ~60 % of the frame's width), very wide (the whole room in frame, the
# operators' desk in the foreground) and aimed low, so a walker on the mat has the head outside the top edge and shows
# the face only when dipping or stepping towards the camera;
# the ALONG view sits at the mat's end about 1 m high looking down the mat, so a walker 2-3 m away shows from the chest
# down and is whole only beyond ~3.5 m. Every sample follows this layout; v2 keeps a lower, closer along camera as the
# deliberate stress test of truncation in that view.
def side_cam(xm, zm, side=-1.0, dx=0.8):
    # 4.0 m from the mat's line (the mat spans ~60 % of the panel's width, as in the real frame) and aimed at -0.21 m
    # on it: the top edge of the picture crosses the mat 1.55 m above the floor, so the walker's head is outside the
    # frame along the mat and the face shows only when the walker dips or steps towards the camera (the real frame)
    return {"W": RW, "H": RC, "yfov": 43.0, "eye": [xm + dx, 2.3, zm + side * 4.0], "target": [xm - 0.2, -0.21, zm]}
def close_along_cam(mat_end, walk_end, zm):
    # v2's stress view: 0.6 m high, 0.3 m beyond the mat's end and at least 0.5 m beyond the walk's own end (the walker
    # steps off the mat to turn), so the standing walker 1 m away shows from the thighs down and still reads as a person
    e = min(mat_end - 0.3, walk_end - 0.5)
    return {"W": LW, "H": LH, "yfov": 60.0, "eye": [e, 0.6, zm + 0.05], "target": [e + 1.7, 0.0, zm]}


def along_cam(mat_end, walk_end, direction, zm, height=1.0, back_mat=1.0, back_walk=1.2):
    """Camera beyond the mat's end `mat_end` and beyond the walker's turn point `walk_end` (whichever is further
    out, so a walk that overshoots the mat still turns in front of the lens), `height` m high, looking along the mat
    in `direction` (+1 towards +x)."""
    ex = min(mat_end - back_mat, walk_end - back_walk) if direction > 0 else max(mat_end + back_mat, walk_end + back_walk)
    return {"W": LW, "H": LH, "yfov": 60.0, "eye": [ex, height, zm], "target": [ex + direction * 3.5, 0.0, zm]}
def desk_seat(xm, zm, side=-1.0):
    return [xm - 1.8, 0.0, zm + side * 2.8]                                       # the operators' desk 1.2 m in front of the side camera, low in its frame


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


def facing_xz(J: np.ndarray) -> np.ndarray:
    """Unit facing direction in the floor plane (T, 2: x, z), from the hip and shoulder lines."""
    left = (J[:, 1] - J[:, 2]) + (J[:, 16] - J[:, 17]); f = np.stack([-left[:, 2], left[:, 0]], 1)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)


def stance_sequence(feet: np.ndarray, fwd: np.ndarray, h_scale: float = 0.03, f_scale: float = 0.02, f_cap: float = 3.0, a_scale: float = 0.02,
                    switch: float = 0.5, cap: float = 25.0):
    """Stance foot per frame (0 left, 1 right) by dynamic programming over the two feet's toe joints (T, 2, 3, the
    root at the origin). Per frame the lower foot is preferred (h_scale m of height costs 1) and so is the foot
    that moves backwards under the body, which carries the root forwards along its facing fwd (T, 2; f_scale m/frame
    less than the other foot costs 1, capped at f_cap); between frames a change of foot costs `switch`, and the root
    velocity the pinned foot implies has to be continuous (a_scale m/frame of change costs 1, squared, capped).
    The inertial fit has no floor: a swinging foot's toe can dip below the planted one for a few frames, and
    choosing the lower foot frame by frame then pins the swinging foot and throws the root backwards.
    Returns (stance, vel) with vel (T, 2, 2) the horizontal root step each foot implies."""
    T = len(feet); vel = np.zeros((T, 2, 2)); vel[1:] = -(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]])
    along = (vel * fwd[:, None, :]).sum(-1)
    un = (feet[:, :, 1] - feet[:, :, 1].min(1, keepdims=True)) / h_scale + np.minimum(f_cap, (along.max(1, keepdims=True) - along) / f_scale)
    cost = np.zeros((T, 2)); back = np.zeros((T, 2), int)
    if T > 1:
        cost[1] = un[1]
    for t in range(2, T):
        for s in range(2):
            c = [cost[t - 1, p] + min(cap, (np.linalg.norm(vel[t, s] - vel[t - 1, p]) / a_scale) ** 2) + (switch if p != s else 0.0) for p in range(2)]
            back[t, s] = int(np.argmin(c)); cost[t, s] = min(c) + un[t, s]
    st = np.zeros(T, int); st[-1] = int(np.argmin(cost[-1]))
    for t in range(T - 1, 1, -1):
        st[t - 1] = back[t, st[t]]
    if T > 1:
        st[0] = st[1]
    return st, vel


def pinned_path(aa, smpl):
    """Horizontal root path by pinning the stance foot (stance_sequence), 9-frame smoothed. Returns (tr, V, J) with
    the vertical left at zero."""
    from scipy.ndimage import uniform_filter1d
    V, J = fk(aa, smpl); st, vel = stance_sequence(J[:, [10, 11]], facing_xz(J)); T = len(aa); tr = np.zeros((T, 3))
    for t in range(1, T):
        tr[t, [0, 2]] = tr[t - 1, [0, 2]] + vel[t, st[t]]
    tr[:, 0] = uniform_filter1d(tr[:, 0], 9, mode="nearest"); tr[:, 2] = uniform_filter1d(tr[:, 2], 9, mode="nearest")
    return tr, V, J


def straight_bouts(tr, min_len: float = 0.4, tol_deg: float = 35.0, min_s: float = 1.0, trim_s: float = 0.3):
    """Runs of straight walking along the room's axis: the displacement over 1 s is longer than min_len m and within
    tol_deg of +x or of -x, for min_s or longer. Returns [(a, b, angle)]: the frames (trimmed by trim_s at each end)
    and the angle of the run's net displacement from its own direction of the axis (rad)."""
    h = FPS // 2; T = len(tr); p = tr[:, [0, 2]]; d = np.zeros((T, 2)); d[h:T - h] = p[2 * h:] - p[:T - 2 * h]
    L = np.linalg.norm(d, axis=1); th = np.arctan2(d[:, 1], d[:, 0]); out = []
    for ref in (0.0, np.pi):
        ok = (L > min_len) & (np.abs(np.angle(np.exp(1j * (th - ref)))) < np.radians(tol_deg)); e = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for a, b in zip(e[::2], e[1::2]):
            if b - a >= int(min_s * FPS):
                a2, b2 = a + int(trim_s * FPS), b - int(trim_s * FPS); dd = p[b2 - 1] - p[a2]
                out.append((int(a2), int(b2), float(np.angle(np.exp(1j * (np.arctan2(dd[1], dd[0]) - ref))))))
    return sorted(out)


def seated_runs(tr, J, min_s: float = 0.5):
    """Frame runs with the walker on the chair: the pelvis less than 75 % of its standing height above the lower
    foot, and the root at rest (under 0.2 m/s over 0.5 s; rising from the chair carries the pelvis forwards)."""
    from scipy.ndimage import uniform_filter1d
    hgt = J[:, 0, 1] - J[:, [10, 11], 1].min(1); sp = uniform_filter1d(np.linalg.norm(np.gradient(tr[:, [0, 2]], axis=0), axis=1) * FPS, FPS // 2, mode="nearest")
    ok = (hgt < 0.75 * np.percentile(hgt, 90)) & (sp < 0.2); e = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return [(int(a), int(b)) for a, b in zip(e[::2], e[1::2]) if b - a >= int(min_s * FPS)]


def carried_curve(T: int, anchors, s: np.ndarray | None = None) -> np.ndarray:
    """Per-frame value from anchors (a, b, value): the value inside each anchor's frames, held before the first and
    after the last, and carried from one anchor to the next with a cosine ease in time or, with s (the root's
    cumulative path length, T), in proportion to the distance travelled: a walker who stands still is not moved."""
    if not anchors:
        return np.zeros(T)
    anchors = [list(x) for x in sorted(anchors)]; gap = FPS // 3
    for p, q in zip(anchors[:-1], anchors[1:]):                       # two anchors that touch are drawn apart: no step between them
        short = gap - (q[0] - p[1])
        if short > 0:
            p[1] = max(p[0] + 1, p[1] - (short + 1) // 2); q[0] = min(q[1] - 1, q[0] + short // 2)
    out = np.full(T, anchors[-1][2], float); out[:anchors[0][1]] = anchors[0][2]
    for (a0, b0, v0), (a1, b1, v1) in zip(anchors[:-1], anchors[1:]):
        out[a0:b0] = v0; n = a1 - b0
        if n > 0:
            if s is not None and s[a1] - s[b0] > 1e-3:
                w = (s[b0:a1] - s[b0]) / (s[a1] - s[b0])
            else:
                w = 0.5 * (1 - np.cos(np.pi * (np.arange(n) + 1) / (n + 1)))
            out[b0:a1] = v0 + (v1 - v0) * w
    out[anchors[-1][0]:] = anchors[-1][2]
    return out


def walker_path(aa, smpl, csv_name: str | None = None, n_iter: int = 2):
    """The walker's pose sequence and root path in the room. Returns (aa, tr).

    The path is the stance-foot-pinned one (pinned_path), made to fit a room with a mat along x:
      heading   the inertial fit's heading is a few degrees off the direction the body travels, and drifts; every
                straight bout is turned onto the room's axis (a yaw per bout, eased through the turn between two
                bouts; a seated walker faces the way of the first seat), and the path is pinned again;
      line      a pinned turn leaves the next bout 0.1-0.3 m to the side; every bout is brought back to one line
                (the first seat's, or the bouts' mean);
      chair     every seated run is brought back to the first seat's position (a Timed Up and Go returns to its
                chair).
    The line and chair shifts are spread between their anchors in proportion to the distance the root travels (a
    slightly longer or shorter step), so no foot slides while the walker stands.
    Vertical = mean floor contact + the recorded IMU vertical displacement (csv_name), or the lowest vertex on the
    floor (csv_name None)."""
    from scipy.ndimage import uniform_filter1d
    from scipy.spatial.transform import Rotation as Rt
    aa = aa.copy(); T = len(aa)
    for _ in range(n_iter):
        tr, V, J = pinned_path(aa, smpl); anc = list(straight_bouts(tr)); S = seated_runs(tr, J)
        if S:
            f = facing_xz(J); f = np.arctan2(f[:, 1], f[:, 0])                                            # a yaw of +a about y turns a floor direction at angle a onto +x
            f0 = np.angle(np.exp(1j * f[S[0][0]:S[0][1]]).mean()); anc += [(a, b, float(np.angle(np.exp(1j * (f[a:b] - f0)).mean()))) for a, b in S]
        if not anc:
            break
        aa[:, :3] = (Rt.from_euler("y", carried_curve(T, anc)) * Rt.from_rotvec(aa[:, :3])).as_rotvec().astype(np.float32)
    tr, V, J = pinned_path(aa, smpl); B = straight_bouts(tr); S = seated_runs(tr, J)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(tr[:, [0, 2]], axis=0), axis=1))]
    zb = [(a, b, float(tr[a:b, 2].mean())) for a, b, _ in B]; zs = [(a, b, float(tr[a:b, 2].mean())) for a, b in S]
    if zb or zs:
        line = zs[0][2] if zs else float(np.average([z for _, _, z in zb], weights=[b - a for a, b, _ in zb]))
        dz = carried_curve(T, [(a, b, line - z) for a, b, z in zb + zs], s)
    else:
        dz = 0.0
    dx = carried_curve(T, [(a, b, float(tr[S[0][0]:S[0][1], 0].mean() - tr[a:b, 0].mean())) for a, b in S], s) if S else 0.0
    tr[:, 0] += dx; tr[:, 2] += dz
    floor = uniform_filter1d(V[..., 1].min(1), 5, mode="nearest")
    if csv_name is None:
        tr[:, 1] = -floor
    else:
        tr[:, 1] = -np.median(floor) + imu_vertical_displacement(csv_name, T)
    return aa.astype(np.float32), tr.astype(np.float32)


def walk_line(tr) -> float:
    """z of the line the walker keeps to: the mean over the straight bouts (over the whole path if there are none)."""
    B = straight_bouts(tr)
    return float(np.concatenate([tr[a:b, 2] for a, b, _ in B]).mean()) if B else float(tr[:, 2].mean())


CLOTHES = {"s": ([170, 60, 55], [45, 50, 75]), "d": ([60, 115, 75], [80, 70, 60]), "b": ([205, 170, 70], [50, 50, 55]), "o": ([95, 95, 155], [65, 65, 70])}   # (top, trousers) per person


def clothed_colours(smpl, tag: str) -> np.ndarray:
    """Per-vertex RGBA: a top, trousers, shoes, skin and hair, by the joint each vertex follows most. A uniformly pale
    body is missed by the person detector while it stands small and still in the side view; a clothed one is found,
    and the tops differ per person as in a real room (the pipeline's appearance descriptor is a torso histogram)."""
    part = smpl.lbs_weights.detach().cpu().numpy().argmax(1); top, trousers = CLOTHES[tag]
    col = np.tile(np.array([205, 165, 140, 255], np.uint8), (len(part), 1))                       # skin: neck, head, lower arms, hands
    col[np.isin(part, [3, 6, 9, 13, 14, 16, 17]), :3] = top; col[np.isin(part, [0, 1, 2, 4, 5]), :3] = trousers; col[np.isin(part, [7, 8, 10, 11]), :3] = [40, 38, 40]
    v = smpl.v_template.detach().cpu().numpy(); head = part == 15
    col[head & (v[:, 1] > np.percentile(v[head, 1], 55)) & (v[:, 2] < np.percentile(v[head, 2], 60)), :3] = [70, 52, 40]   # hair: the upper back of the head
    return col


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
    # v1: the dry-run sample. Along view: camera at the walkway end, 1 m high (chest down at 2-3 m, head cut when
    # close); side view: the room-wide one (side_cam), the walker out of it only past the mat's ends; a co-walker, a
    # seated operator, a bystander.
    "v1": {"sid": "HC900", "subdir": "sample", "decoy": True, "seat": lambda xm, zm: desk_seat(xm, zm), "bystander": lambda x0, x1, xm, zm: [xm + 1.5, 0.0, zm + 3.0],
           "cams": lambda x0, x1, xm, zm: {"left": along_cam(xm - 0.5 * MAT_L, x0, +1, zm), "right": side_cam(xm, zm)}},
    # v2: two deliberately compromised views of one walk, after the real session layout. Along view: camera 0.5 m
    # beyond the near end, 0.6 m high, looking down the mat: the whole body only at the far end, head and shoulders
    # gone mid-way, shins only at the near end (the pose is lost there). Side view: the room-wide one, the head
    # above the top edge along the whole mat. No co-walker: the point of this sample is reconstruction from two
    # truncated views, and the ground-truth 3D pose is saved beside it.
    "v2": {"sid": "HC901", "subdir": "sample_v2", "decoy": False, "seat": lambda xm, zm: desk_seat(xm, zm), "bystander": lambda x0, x1, xm, zm: [xm + 1.5, 0.0, zm + 3.0],
           "cams": lambda x0, x1, xm, zm: {"left": close_along_cam(xm - 0.5 * MAT_L, x0, zm),   # the stress test: low and close, shins only when near
                                           "right": side_cam(xm, zm)}},
    # v3: a third room (sample_v3/, session HC902): a different walker (hc105: 6.3 m of walkway, the largest pelvis bob of
    # the public set) with hc100 as the co-walker on the far
    # side of the mat. The along camera stands at the OTHER end of the mat (the walker comes towards it on the
    # return) and the side camera on the other side of the room, the same room-wide framing mirrored, so the walker
    # leaves it beyond the mat's end nearest the along camera. The operator sits close to the side camera, lower
    # right; the bystander stands at the far wall.
    "v3": {"sid": "HC902", "subdir": "sample_v3", "decoy": True, "subject": "hc105_selfpace", "decoy_name": "hc100_selfpace", "decoy_dz": 1.5,
           "seat": lambda xm, zm: desk_seat(xm, zm), "bystander": lambda x0, x1, xm, zm: [xm - 1.0, 0.0, zm + 3.0],
           "cams": lambda x0, x1, xm, zm: {"left": along_cam(xm + 0.5 * MAT_L, x1, -1, zm), "right": side_cam(xm, zm, dx=-0.8)}},    # the along camera at the other end
    # v4: a Timed Up and Go (sample_v4/, session HC903, task TUG): hc142's repetitions of sit, stand, walk 3 m, turn,
    # walk back, turn and sit, with a chair at the start; v1's room, the along camera facing the chair, no co-walker. Short walks, a
    # large sit-to-stand excursion and a seated start are what a TUG asks of the selection and the sync. (hc100's TUG,
    # used first, has its left ankle 163 deg off in the inertial fit: a visibly wrong foot in the rendered truth.)
    "v4": {"sid": "HC903", "subdir": "sample_v4", "decoy": False, "subject": "hc142_tug", "task": {"csv": "tug", "dir": "TUG", "token": "TUG"}, "chair": True,
           "seat": lambda xm, zm: desk_seat(xm, zm), "bystander": lambda x0, x1, xm, zm: [xm + 1.5, 0.0, zm + 3.0],
           # the along camera faces the chair from beyond the turn-around point, as a TUG is usually filmed
           "cams": lambda x0, x1, xm, zm: {"left": along_cam(x1 + 0.5, x1, -1, zm), "right": side_cam(xm, zm)}},
    # v5: a hurried-pace walk (sample_v5/, session HC904, task HP): hc105 at speed, with hc100 as the co-walker. The
    # along view is v1's camera past the near end (feet and mat, head cut when close); the side view is the room-wide
    # one every sample shares (side_cam: 2.3 m high, 4 m from the mat, the desk in the foreground); the hurried
    # walker overshoots the mat far enough to leave it at the near end for a third of the recording.
    "v5": {"sid": "HC904", "subdir": "sample_v5", "decoy": True, "subject": "hc105_hurriedpace", "decoy_name": "hc100_hurriedpace", "decoy_dz": 1.5, "task": {"csv": "hurriedpace", "dir": "HP", "token": "HP"},
           "seat": lambda xm, zm: desk_seat(xm, zm), "bystander": lambda x0, x1, xm, zm: [x1 + 1.5, 0.0, zm + 3.0],
           "cams": lambda x0, x1, xm, zm: {"left": along_cam(xm - 0.5 * MAT_L, x0, +1, zm), "right": side_cam(xm, zm)}},
    # v6: a self-paced walk ON the instrumented mat (sample_v6/, session HC905, task SPM = selfpace_mat) with the walkway
    # columns of the recording CSV FABRICATED from the rendered truth in the export's own form: at every IMU sample the
    # loaded cells of the mat (Walkway_X/Y cell indices, WalkwayFoot labels, WalkwayPressureLevel) under each foot that
    # is on it, on a 384 x 48 grid of 1.27 cm cells centred on the walk. (The real recording's own walkway rows are
    # NOT used: the render walks the neutral SMPL body along a stance-pinned path 22 % shorter than the real one, so
    # the real footprints would contradict the video.) v3's room.
    "v6": {"sid": "HC905", "subdir": "sample_v6", "decoy": False, "subject": "hc105_selfpace_mat", "task": {"csv": "selfpace_mat", "dir": "SPM", "token": "SPM"}, "walkway_from_truth": True,
           "seat": lambda xm, zm: desk_seat(xm, zm), "bystander": lambda x0, x1, xm, zm: [xm - 1.0, 0.0, zm + 3.0],
           "cams": lambda x0, x1, xm, zm: {"left": along_cam(xm + 0.5 * MAT_L, x1, -1, zm), "right": side_cam(xm, zm, dx=-0.8)}},
}


MAT_CELL_M = 0.0127; MAT_CELLS = (384, 48)        # the GAITRite grid: 4.88 x 0.61 m


def recording_csv_subset(var: dict, subj_name: str, J_s: np.ndarray, T_s: int, x0: float, x1: float, xm: float, zm: float):
    """The session's recording CSV: a column subset of the public recording with the walkway columns EMPTIED (the
    real footprints describe the real walk, about a quarter longer than the rendered neutral-body one) or, for the mat variant,
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
    mat_frame = {"x0": float(xm - 0.5 * MAT_L), "z0": float(zm - 0.5 * MAT_W), "cell_m": MAT_CELL_M, "cells": list(MAT_CELLS),
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
    subj, tr_s = walker_path(yaw_align(load_ik_aa(subj_name), smpl), smpl, f"{subj_name}.csv"); T_s = len(subj)
    x0, x1 = float(tr_s[:, 0].min()), float(tr_s[:, 0].max()); xm = 0.5 * (x0 + x1); zm = walk_line(tr_s)
    _, J_s = fk(subj, smpl, tr_s)
    G = np.load(out / f"{sid}_gt.npz")
    assert np.allclose(G["transl"], tr_s, atol=1e-4) and len(G["pose_aa"]) == T_s, "the rendered truth differs from this build: re-render instead"
    dfc, mat_frame = recording_csv_subset(var, subj_name, J_s, T_s, x0, x1, xm, zm)
    dfc.to_csv(sess / f"{sid.lower()}_{tcsv}.csv", index=False)
    ej = out / "expected.json"; exp = json.loads(ej.read_text(encoding="utf-8")); exp["mat_frame"] = mat_frame; ej.write_text(json.dumps(exp), encoding="utf-8")
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

    subj, tr_s = walker_path(yaw_align(load_ik_aa(subj_name), smpl), smpl, f"{subj_name}.csv"); T_s = len(subj)
    if has_decoy:
        deco, tr_d = walker_path(yaw_align(load_ik_aa(deco_name), smpl), smpl, f"{deco_name}.csv"); tr_d = tr_d + np.array([0, 0, walk_line(tr_s) + decoy_dz - walk_line(tr_d)], np.float32); delay_d = int(0.8 * FPS)   # the co-walker keeps decoy_dz beside the subject
    byst = yaw_align(load_ik_aa("hc101_selfpace"), smpl)[:45]; byst, tr_b = walker_path(np.concatenate([byst, byst[::-1]]), smpl)
    x0, x1 = float(tr_s[:, 0].min()), float(tr_s[:, 0].max()); xm = 0.5 * (x0 + x1); zm = walk_line(tr_s)
    tr_b = tr_b + np.array(var["bystander"](x0, x1, xm, zm) if "bystander" in var else [x1 + 2.2, 0.0, zm + 1.2], np.float32)
    pre = int(3.0 * FPS); post = int(1.5 * FPS); n_frames = pre + T_s + post
    seat = seated_sequence(n_frames, rng)
    V_s, J_s = fk(subj, smpl, tr_s); V_b, J_b = fk(byst, smpl, tr_b)
    if has_decoy:
        V_d, J_d = fk(deco, smpl, tr_d)
    # cameras: along-walkway = low at the x0 end looking down the mat (bodies truncate at the top when close);
    # side = the room-wide camera (side_cam): elevated, 4 m from the mat, the walker out of it only past the mat's ends
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
        mv = np.array([[xm - 0.5 * MAT_L, 0.006, zm - 0.5 * MAT_W], [xm + 0.5 * MAT_L, 0.006, zm - 0.5 * MAT_W], [xm + 0.5 * MAT_L, 0.006, zm + 0.5 * MAT_W], [xm - 0.5 * MAT_L, 0.006, zm + 0.5 * MAT_W]], np.float32)
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
    colors = {tag: clothed_colours(smpl, tag) for tag in CLOTHES}
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
                tm = trimesh.Trimesh(v, faces, process=False); tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=colors[tag])
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
                                     + (" with the walkway columns fabricated from the rendered truth in the export's form (loaded cells per sample, 1.27 cm grid).\n" if mat_frame else "; its walkway columns are left empty (the real recording's footprints do not describe the rendered walk).\n"), encoding="utf-8")
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
                          "layout": {"mat_x": [float(xm - 0.5 * MAT_L), float(xm + 0.5 * MAT_L)], "mat_z": [float(zm - 0.5 * MAT_W), float(zm + 0.5 * MAT_W)], "mat_size_m": [MAT_L, MAT_W], "operator_seat_xyz": [float(v) for v in seat_pos], "operator_yaw_deg": 25.0,
                                     "desk_center_xyz": [float(seat_pos[0]), 0.74, float(seat_pos[2] - 0.75)], "desk_extents_xyz": [1.8, 0.05, 0.7],
                                     "bystander_xyz": [float(tr_b[0, 0]), 0.0, float(tr_b[0, 2])], "walls": [{"x": float(x1 + 4.0), "z": [float(zm - 8), float(zm + 8)]}, {"z": float(zm + 4.0), "x": [float(x0 - 8), float(x1 + 8)]}],
                                     "cameras": {cid: {"eye": [float(v) for v in c["eye"]], "target": [float(v) for v in c["target"]], "yfov_deg": float(c["yfov"])} for cid, c in cams.items()}},
                          "walk_extent_m": [x0, x1]}}
    (out / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
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
