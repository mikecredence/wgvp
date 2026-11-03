"""Demo video: 3D reconstruction from two partial views (HC901).

    python tools/make_demo.py <data root with wgvp_output/HC901> --out demo.mp4 [--gif docs/figures/two_view_demo.gif]

Layout per frame (IMU time t):
  top      the along view and the side view at t with the chosen subject's 2D skeleton; the border says whether the
           view is usable for the legs at t (green), lifted but with the leg joints at a frame edge (amber), or
           without the subject (red)
  middle   left: the rendered truth. Centre: the reconstruction by the cross-faded segment rule: every joint comes
           from the view(s) whose ViTPose keypoints for it are inside the frame, with the weights ramped over the
           first and last half-second of each usable segment so hand-overs between views are continuous, averaged
           where both qualify. Body parts driven by joints that no view could see are drawn red and are NOT part of
           the output (filled only so the mesh can be drawn). When the body is in neither view nothing is drawn.
           Right: the export, what a user would take away: only frames in which every leg joint is output, in runs
           of at least half a second, and on those frames only the parts some view saw (the rest is absent).
  bottom   the readout on the shared IMU axis: a filmstrip of small renders every --strip-step seconds (what each
           view outputs, the merge, the truth) with the well-tracked blocks marked at frame resolution (bars) and
           boxes around the consecutive thumbnails inside one block; then knee-flexion charts (each view, the
           merge, the truth); a cursor marks t.
The reconstruction's root is the weighted quaternion mean of the views' roots (weight = ramped presence x the
square of the fraction of joints the view finds usable, sharpened towards the heavier view when the two disagree;
tools/two_view_fusion.fuse_root), after each view's constant heading offset to the truth has been removed (the two
cameras' world frames differ by that heading). Root position is not part of the experiment; bodies are drawn at
the origin. The 2D skeletons on the views are ViTPose's keypoints (the lifter's input), drawn with the left/right
labels of the arms and legs kept consistent from frame to frame (2D estimators flip sides on a side-on walker;
the 3D stage does not use those labels).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from tools.two_view_fusion import LEGS, chordal_mean, export_mask, fit_yaw, fuse_root, geodesic, ramp_inside, smooth_rotations, usable_joints_rt   # noqa: E402
from wgvp.tracking import FACE_KP, SKELETON                                                             # noqa: E402

COL = {"ok": [150, 160, 200], "missing": [220, 70, 70], "truth": [170, 170, 170]}
RAMP_FRAMES = 15        # cross-fade at the edges of usable segments (0.5 s at 30 fps)
MIN_RUN = 15            # an export keeps runs of at least this many frames
ROW_BGR = {"along": (180, 119, 31), "side": (14, 127, 255), "merge": (63, 209, 63)}     # tab:blue, tab:orange, the merge green


def knee_deg(body):   # (T, 21, 3) -> (T, 2) left, right knee flexion magnitude
    return np.degrees(np.linalg.norm(body[:, [3, 4]], axis=2))


def side_consistent(kp: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """Display only. (T, 17, 3) COCO keypoints (NaN where absent) -> same, with the left/right labels of the legs,
    and separately of the arms, swapped in a frame when the swapped assignment continues the previously drawn
    frame more closely. 2D pose estimators flip sides on a side-on walker; the 3D stage does not use these labels,
    so the flips only matter for the overlay."""
    out = kp.copy(); prev = None; prev_t = -10 ** 9
    for t in range(len(out)):
        cur = out[t]
        if not np.isfinite(cur[:, :2]).any():
            continue
        if prev is not None and t - prev_t <= max_gap:
            for L_ids, R_ids in (([11, 13, 15], [12, 14, 16]), ([5, 7, 9], [6, 8, 10])):
                ok = np.isfinite(cur[L_ids, :2]).all(1) & np.isfinite(cur[R_ids, :2]).all(1) & np.isfinite(prev[L_ids, :2]).all(1) & np.isfinite(prev[R_ids, :2]).all(1)
                if not ok.any():
                    continue
                Li, Ri = np.array(L_ids)[ok], np.array(R_ids)[ok]
                keep = np.linalg.norm(cur[Li, :2] - prev[Li, :2], axis=1).sum() + np.linalg.norm(cur[Ri, :2] - prev[Ri, :2], axis=1).sum()
                swap = np.linalg.norm(cur[Li, :2] - prev[Ri, :2], axis=1).sum() + np.linalg.norm(cur[Ri, :2] - prev[Li, :2], axis=1).sum()
                if swap < 0.7 * keep:
                    cur[L_ids + R_ids] = cur[R_ids + L_ids]
        prev = cur.copy(); prev_t = t
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("--session", default="HC901"); ap.add_argument("--recording", default="hc901_selfpace")
    ap.add_argument("--video", default=str(PKG / "sample_v2/data/HC901/SP/HC901_SP_DeID.mp4")); ap.add_argument("--gt", default=str(PKG / "sample_v2/HC901_gt.npz"))
    ap.add_argument("--body-models", default=str(Path.home() / "data/WearGait/body_models")); ap.add_argument("--out", default=str(PKG / "docs/figures/two_view_demo.mp4"))
    ap.add_argument("--gif", default=None); ap.add_argument("--gif-span", default="28,42"); ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--t0", type=float, default=None); ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--smooth", type=int, default=5, help="rotation moving-average window (frames) on the merge; 0 = off")
    ap.add_argument("--readout", choices=["both", "charts", "strips"], default="both", help="bottom readout: knee-flexion charts, a filmstrip of small renders every --strip-step seconds, or both")
    ap.add_argument("--strip-step", type=float, default=2.0)
    a = ap.parse_args()
    deliver = Path(a.root) / "wgvp_output" / a.session / "deliver" / a.recording
    views = sorted(p.name for p in deliver.iterdir() if (p / "smpl_global.npz").exists()); A, B = views
    Z = {v: np.load(deliver / v / "smpl_global.npz") for v in views}; K = {v: np.load(deliver / v / "kp2d_vitpose.npz") for v in views}; S = {v: np.load(deliver / v / "subject_track.npz") for v in views}
    gt = np.load(a.gt); gi = {int(f): i for i, f in enumerate(gt["frame_idx"])}
    offset = float(json.loads((deliver / "sync.json").read_text())["offset_s"])
    cap = cv2.VideoCapture(a.video); n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = 30.0; t_all = np.arange(n_video) / fps - offset
    # ---- per view, dense over the video's frames: body, root, usable joints, presence
    body, root, usable, lifted, vis = {}, {}, {}, {}, {}
    for v in views:
        z, k = Z[v], K[v]; fr = z["frame_idx"].astype(int)
        body[v] = np.full((n_video, 21, 3), np.nan); root[v] = np.full((n_video, 3), np.nan); usable[v] = np.zeros((n_video, 21), bool); lifted[v] = np.zeros(n_video, bool); vis[v] = np.zeros(n_video, bool)
        body[v][fr] = z["body_pose"].reshape(-1, 21, 3); root[v][fr] = z["global_orient"]; lifted[v][fr] = True; vis[v][fr] = z["visible"]
        usable[v][fr] = usable_joints_rt(k["kp2d"], k["view_box"] if "view_box" in k.files else None, z["visible"])
    common = [f for f in range(n_video) if f in gi and lifted[A][f] and lifted[B][f]]
    yaw = {v: fit_yaw(root[v][common], gt["pose_aa"][[gi[f] for f in common], :3], vis[v][common]) for v in views}
    print("heading offsets (deg):", yaw)
    # ---- the reconstruction's root, dense: per view after its heading offset, then the weighted quaternion mean of
    # the lifted views (two_view_fusion.fuse_root: ramped presence x usable fraction squared, sharpened towards the
    # heavier view when the two disagree), then the same smoothing
    root_al = {v: np.full((n_video, 3), np.nan) for v in views}
    for v in views:
        ok = lifted[v]; root_al[v][ok] = (R.from_euler("y", yaw[v], degrees=True) * R.from_rotvec(root[v][ok])).as_rotvec()
    root_fused, share = fuse_root([root_al[A], root_al[B]], [usable[A], usable[B]], [lifted[A], lifted[B]], ramp=RAMP_FRAMES)
    if a.smooth and a.smooth > 1:
        root_fused = smooth_rotations(root_fused[:, None, :], a.smooth)[:, 0, :]
    # ---- the merge, dense: cross-faded segment rule, each view's weights scaled by its share of the root (a view
    # whose root disagrees with the view that sees more of the body solved its joints in the wrong gravity frame)
    w = {v: ramp_inside(usable[v], RAMP_FRAMES) * share[i][:, None] for i, v in enumerate(views)}
    est = {v: np.nan_to_num(body[v]) for v in views}
    merged = chordal_mean([est[A], est[B]], np.stack([w[A], w[B]]))
    none = ~(usable[A] | usable[B]); fill = chordal_mean([est[A], est[B]], np.stack([lifted[A][:, None].repeat(21, 1).astype(float), lifted[B][:, None].repeat(21, 1).astype(float)]))
    merged_out = merged.copy(); merged_out[none] = np.nan                      # the output: nothing where no view saw the joint
    merged_draw = np.where(none[:, :, None], fill, merged)                       # for drawing only: unseen parts filled, shown red
    if a.smooth and a.smooth > 1:                                                # per-frame jitter of the lifter: short moving average, gaps respected
        merged_out = smooth_rotations(merged_out, a.smooth)
        anyl = (lifted[A] | lifted[B])[:, None].repeat(21, 1)
        md = merged_draw.copy(); md[~anyl] = np.nan; merged_draw = np.nan_to_num(smooth_rotations(md, a.smooth))
    # ---- what an export keeps: frames with every leg joint output, in runs of at least MIN_RUN frames
    export_ok = export_mask(merged_out, LEGS, MIN_RUN)
    # ---- 2D keypoints drawn on the views: ViTPose (the lifter's input), sides kept consistent for display
    kp_draw = {}
    for v in views:
        arr = np.full((n_video, 17, 3), np.nan); arr[K[v]["frame_idx"].astype(int)] = K[v]["kp2d"]; kp_draw[v] = side_consistent(arr)
    # ---- truth series
    truth_body = np.full((n_video, 21, 3), np.nan)
    for f, i in gi.items():
        if f < n_video:
            truth_body[f] = gt["pose_aa"][i, 3:66].reshape(21, 3)
    kn = {"truth": knee_deg(truth_body), A: knee_deg(body[A]), B: knee_deg(body[B]), "merge": knee_deg(merged_out)}
    for v in views:                                                             # a view's readout is only what it would output: its usable joints
        kn[v][~usable[v][:, 3], 0] = np.nan; kn[v][~usable[v][:, 4], 1] = np.nan
    # ---- frames to draw
    t0 = a.t0 if a.t0 is not None else 0.0; t1 = a.t1 if a.t1 is not None else float(gt["imu_t_s"][-1])
    frames = [f for f in range(n_video) if t0 <= t_all[f] <= t1][::a.step]
    # ---- SMPL-X + renderer
    os.environ.pop("PYOPENGL_PLATFORM", None)
    import torch, smplx, trimesh, pyrender
    bm = smplx.create(a.body_models, model_type="smplx", gender="neutral", use_pca=False, batch_size=1); faces = bm.faces.astype(np.int32)
    dom = bm.lbs_weights.detach().cpu().numpy()[:, :22].argmax(1)
    def verts(root_aa, body_aa):
        with torch.no_grad():
            o = bm(global_orient=torch.tensor(root_aa, dtype=torch.float32)[None], body_pose=torch.tensor(body_aa.reshape(63), dtype=torch.float32)[None], betas=torch.zeros(1, 10), transl=torch.zeros(1, 3))
        return o.vertices[0].numpy()
    RW, RH = 340, 442; AW, AH = 360, 250
    renderer = pyrender.OffscreenRenderer(RW, RH); aerial_r = pyrender.OffscreenRenderer(AW, AH)
    # the scene's mat: the walk's extent along x (from the sample's expected answers) and the mat's lateral position
    exp = json.loads((Path(a.gt).parent / "expected.json").read_text()) if (Path(a.gt).parent / "expected.json").exists() else {}
    x0m, x1m = exp.get("scene", {}).get("walk_extent_m", [-2.45, 2.45]); zm = float(np.median(gt["transl"][:, 2])) if "transl" in gt.files else 0.0
    cams = {v: gt[f"{v}_cam_pose"] for v in views if f"{v}_cam_pose" in gt.files}
    xm_ = 0.5 * (x0m + x1m)
    lay = exp.get("scene", {}).get("layout") or {"operator_seat_xyz": [xm_ + 2.4, 0.0, zm - 1.2], "operator_yaw_deg": 25.0, "desk_center_xyz": [xm_ + 2.4, 0.74, zm - 1.95], "desk_extents_xyz": [1.8, 0.05, 0.7],
                                                 "bystander_xyz": [x1m + 2.2, 0.0, zm + 1.2]}
    # the other people in the room, as static bodies (the operator seated at the desk, the bystander standing)
    seat = np.zeros(63, np.float32); seat[[0, 3]] = -1.45; seat[[9, 12]] = 1.5          # L/R hip flexion, L/R knee flexion (SMPL-X body joints 1,2 / 4,5)
    with torch.no_grad():
        o = bm(global_orient=torch.tensor(R.from_euler("y", lay["operator_yaw_deg"] + 180.0, degrees=True).as_rotvec(), dtype=torch.float32)[None], body_pose=torch.tensor(seat)[None], betas=torch.zeros(1, 10),
               transl=torch.tensor(np.array(lay["operator_seat_xyz"]) + [0.0, 0.45, 0.0], dtype=torch.float32)[None])
        V_op = o.vertices[0].numpy(); V_op[:, 1] -= V_op[:, 1].min()                   # feet on the floor
        o = bm(global_orient=torch.tensor(R.from_euler("y", 180.0, degrees=True).as_rotvec(), dtype=torch.float32)[None], body_pose=torch.zeros(1, 63), betas=torch.zeros(1, 10),
               transl=torch.tensor(np.array(lay["bystander_xyz"]) + [0.0, 1.0, 0.0], dtype=torch.float32)[None])
        V_by = o.vertices[0].numpy(); V_by[:, 1] -= V_by[:, 1].min()
    desk = trimesh.creation.box(extents=lay["desk_extents_xyz"]); desk.apply_translation(lay["desk_center_xyz"]); desk.visual = trimesh.visual.ColorVisuals(mesh=desk, face_colors=np.tile([150, 120, 90, 255], (len(desk.faces), 1)))
    def quad(pts, rgba):
        m = trimesh.Trimesh(np.asarray(pts, np.float32), np.array([[0, 1, 2], [0, 2, 3]]), process=False); m.visual = trimesh.visual.ColorVisuals(mesh=m, face_colors=np.tile(rgba, (2, 1)))
        return pyrender.Mesh.from_trimesh(m, smooth=False)
    def add_floor_and_mat(sc, floor_y, cx, half_len, cz, big=12.0):
        sc.add(quad([[cx - big, floor_y, cz - big], [cx + big, floor_y, cz - big], [cx + big, floor_y, cz + big], [cx - big, floor_y, cz + big]], [205, 200, 190, 255]))
        sc.add(quad([[cx - half_len - 0.8, floor_y + 0.004, cz - 0.5], [cx + half_len + 0.8, floor_y + 0.004, cz - 0.5], [cx + half_len + 0.8, floor_y + 0.004, cz + 0.5], [cx - half_len - 0.8, floor_y + 0.004, cz + 0.5]], [45, 42, 44, 255]))
    def look_at(eye, target, up=(0, 1, 0)):
        eye = np.asarray(eye, float); fwd = eye - np.asarray(target, float); fwd /= np.linalg.norm(fwd)
        right = np.cross(np.asarray(up, float), fwd); right /= np.linalg.norm(right); up_ = np.cross(fwd, right)
        M = np.eye(4); M[:3, 0] = right; M[:3, 1] = up_; M[:3, 2] = fwd; M[:3, 3] = eye; return M
    def lights(sc):
        for p_ in ((2, 4, 3), (-2, 3, -2)):
            Lm = np.eye(4); Lm[:3, 3] = p_; sc.add(pyrender.DirectionalLight(intensity=2.5), pose=Lm)
    def finish(rd, sc):
        col, _ = rd.render(sc, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES); col = col.astype(np.float32); al = col[..., 3:4] / 255.0
        return (col[..., :3] * al + 255 * (1 - al)).astype(np.uint8)[:, :, ::-1]
    def body_mesh(V, colours, keep_faces=None):
        tm = trimesh.Trimesh(V, faces if keep_faces is None else faces[keep_faces], process=False)
        tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=np.c_[colours, np.full(len(V), 255)].astype(np.uint8))
        return pyrender.Mesh.from_trimesh(tm, smooth=True)
    view_ref = {"floor": -1.17, "c": np.array([0.0, -0.25, 0.0])}
    def render(V, colours, keep_faces=None):
        """Close-up: the body at the origin on a floor with the mat running along x under it; V None = floor only
        (the camera and floor height of the last body drawn, so an empty frame looks like the same room)."""
        sc = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.45, 0.45, 0.45))
        if V is not None:
            sc.add(body_mesh(V, colours, keep_faces)); view_ref["floor"] = float(V[:, 1].min()) - 0.01; view_ref["c"] = np.array([0.0, V[:, 1].mean(), 0.0])
        add_floor_and_mat(sc, view_ref["floor"], 0.0, 2.45, 0.0)
        c = view_ref["c"]; az = np.deg2rad(40); eye = c + 3.3 * np.array([np.sin(az), 0.25, np.cos(az)])
        sc.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(40)), pose=look_at(eye, c)); lights(sc); return finish(renderer, sc)
    def render_aerial(V_world):
        """The whole scene from above: floor, mat, the truth body at its real position, the two cameras."""
        sc = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.5, 0.5, 0.5))
        add_floor_and_mat(sc, 0.0, 0.5 * (x0m + x1m), 0.5 * (x1m - x0m), zm)
        if V_world is not None:
            sc.add(body_mesh(V_world, np.tile(COL["truth"], (len(V_world), 1))))
        sc.add(body_mesh(V_op, np.tile([120, 120, 160], (len(V_op), 1)))); sc.add(body_mesh(V_by, np.tile([150, 175, 150], (len(V_by), 1)))); sc.add(pyrender.Mesh.from_trimesh(desk, smooth=False))
        for v, cm in cams.items():
            box = trimesh.creation.box(extents=(0.25, 0.25, 0.25)); box.apply_translation(cm[:3, 3]); box.visual = trimesh.visual.ColorVisuals(mesh=box, face_colors=np.tile([60, 110, 230, 255] if v == A else [240, 150, 40, 255], (len(box.faces), 1)))
            sc.add(pyrender.Mesh.from_trimesh(box, smooth=False))
        cx = 0.5 * (x0m + x1m) + 0.6; eye = [cx, 13.0, zm - 2.4]
        sc.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(52)), pose=look_at(eye, [cx, 0.0, zm - 2.2], up=(0, 0, -1))); lights(sc); return finish(aerial_r, sc)
    def verts_world(root_aa, body_aa, transl):
        with torch.no_grad():
            o = bm(global_orient=torch.tensor(root_aa, dtype=torch.float32)[None], body_pose=torch.tensor(body_aa.reshape(63), dtype=torch.float32)[None], betas=torch.zeros(1, 10), transl=torch.tensor(transl, dtype=torch.float32)[None])
        return o.vertices[0].numpy()
    # ---- layout
    PH = 320; vbox = {v: [int(x) for x in K[v]["view_box"]] for v in views}
    pw = {v: int(round(PH * (vbox[v][2] - vbox[v][0]) / (vbox[v][3] - vbox[v][1]))) for v in views}
    W = max(pw[A] + pw[B] + 30 + 20, 3 * RW + 4 * 20 + AW + 20); TLH = 330
    light_rect = exp.get("light_rect")                      # the sample's sync light (bright while the IMU records), labelled on the view that shows it
    def label(img, txt, xy, col=(255, 255, 255), scale=0.55, th=1, bg=(0, 0, 0)):
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, th); x, y = xy
        cv2.rectangle(img, (x - 3, y - tht - 5), (x + tw + 3, y + 4), bg, -1); cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)
    # ---- the static timeline image (matplotlib), cursor drawn per frame
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dpi = 100; fig = plt.figure(figsize=(W / dpi, TLH / dpi), dpi=dpi, facecolor=(0.11, 0.11, 0.11))
    L_, R_ = 0.06, 0.99
    axs = [fig.add_axes([L_, 0.80, R_ - L_, 0.13]), fig.add_axes([L_, 0.44, R_ - L_, 0.33]), fig.add_axes([L_, 0.08, R_ - L_, 0.33])]
    tt = t_all; win = (tt >= t0) & (tt <= t1)
    leg_ok = {v: usable[v][:, LEGS].all(1) for v in views}
    for y, v, c in ((1.0, A, "tab:blue"), (0.0, B, "tab:orange")):
        axs[0].fill_between(tt[win], y - 0.4, y + 0.4, where=leg_ok[v][win], color=c, step="mid", lw=0, alpha=0.9)
    axs[0].set_yticks([0, 1]); axs[0].set_yticklabels(["side view", "along view"], fontsize=7, color="w"); axs[0].set_ylim(-0.6, 1.6); axs[0].set_title("usable for the legs", fontsize=8, color="w", loc="left", pad=2)
    for ax, j, name in ((axs[1], 0, "left knee flexion (deg)"), (axs[2], 1, "right knee flexion (deg)")):
        ax.plot(tt[win], kn[A][win, j], color="tab:blue", lw=0.8, alpha=0.8, label="along view output")
        ax.plot(tt[win], kn[B][win, j], color="tab:orange", lw=0.8, alpha=0.8, label="side view output")
        ax.plot(tt[win], kn["truth"][win, j], color="w", lw=1.0, alpha=0.9, label="truth")
        ax.plot(tt[win], kn["merge"][win, j], color="#3fd13f", lw=2.0, label="merge (cross-faded segment rule)")
        ax.set_ylabel(name, fontsize=7, color="w"); ax.set_ylim(0, 90)
    axs[1].legend(fontsize=7, ncol=4, loc="upper left", facecolor=(0.2, 0.2, 0.2), labelcolor="w", framealpha=0.8)
    for ax in axs:
        ax.set_xlim(t0, t1); ax.set_facecolor((0.15, 0.15, 0.15)); ax.tick_params(colors="w", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("0.5")
    axs[0].set_xticklabels([]); axs[1].set_xticklabels([]); axs[2].set_xlabel("IMU time (s)", fontsize=8, color="w")
    tmpdir = Path(tempfile.mkdtemp()); fig.savefig(tmpdir / "timeline.png", dpi=dpi, facecolor=fig.get_facecolor()); plt.close(fig)
    timeline = cv2.imread(str(tmpdir / "timeline.png")); timeline = cv2.resize(timeline, (W, TLH))
    def cursor_x(t): return int(L_ * W + (t - t0) / max(t1 - t0, 1e-6) * (R_ - L_) * W)
    # ---- the filmstrip: at every --strip-step seconds a small render of what each view outputs, the merge, the truth
    ticks = np.arange(t0 + 0.5 * a.strip_step, t1, a.strip_step); n_tick = len(ticks); TW = min(56, int((R_ - L_) * W / max(n_tick, 1))); TH = int(TW * 5 / 3); RP = TH + 11; SH = 4 * RP + 20
    thumb_r = pyrender.OffscreenRenderer(2 * TW, 2 * TH)
    def thumb(V, colours):
        sc = pyrender.Scene(bg_color=(0.16, 0.16, 0.16, 1.0), ambient_light=(0.5, 0.5, 0.5)); sc.add(body_mesh(V, colours))
        c = np.array([0.0, V[:, 1].mean(), 0.0]); az = np.deg2rad(40); eye = c + 3.0 * np.array([np.sin(az), 0.2, np.cos(az)])
        sc.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(42)), pose=look_at(eye, c)); lights(sc)
        col, _ = thumb_r.render(sc, flags=pyrender.RenderFlags.SKIP_CULL_FACES); return cv2.resize(col[:, :, ::-1], (TW, TH), interpolation=cv2.INTER_AREA)
    strip = np.full((SH, W, 3), 28, np.uint8); rows = [("along view", A), ("side view", B), ("merge", None), ("truth", "truth")]
    # the well-tracked blocks per row, at frame resolution: a view's legs usable; for the merge, the frames the export keeps
    good = {A: leg_ok[A] & win, B: leg_ok[B] & win, None: export_ok & win}; row_bgr = {A: ROW_BGR["along"], B: ROW_BGR["side"], None: ROW_BGR["merge"]}
    xL = int(L_ * W); x_of_t = lambda t_: xL + (t_ - t0) / a.strip_step * TW          # thumbnail k spans IMU [t0 + k step, t0 + (k + 1) step)
    for r_i, (name, key) in enumerate(rows):
        y = 18 + r_i * RP; cv2.putText(strip, name, (4, y + TH // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        for k_i, tk in enumerate(ticks):
            f = int(np.argmin(np.abs(t_all - tk))); x = xL + k_i * TW; img = None
            if key == "truth":
                if f in gi:
                    img = thumb(verts(gt["pose_aa"][gi[f], :3], gt["pose_aa"][gi[f], 3:66]), np.tile(COL["truth"], (10475, 1)))
            elif key is None:
                if (lifted[A][f] or lifted[B][f]) and not none[f].all():
                    V_ = verts(root_fused[f], merged_draw[f]); cl = np.tile(COL["ok"], (len(V_), 1)); cl[np.isin(dom, np.flatnonzero(none[f]) + 1)] = COL["missing"]; img = thumb(V_, cl)
            elif lifted[key][f] and vis[key][f]:
                V_ = verts(root_al[key][f], np.nan_to_num(body[key][f])); cl = np.tile(COL["ok"], (len(V_), 1)); cl[np.isin(dom, np.flatnonzero(~usable[key][f]) + 1)] = COL["missing"]; img = thumb(V_, cl)
            if img is not None:
                if key in good and not good[key][f]:
                    img = (img * 0.45).astype(np.uint8)                                   # outside a well-tracked block: dimmed
                strip[y:y + TH, x:x + TW] = img
            else:
                cv2.rectangle(strip, (x, y), (x + TW - 1, y + TH - 1), (40, 40, 40), -1); cv2.putText(strip, "-", (x + TW // 2 - 4, y + TH // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
        if key in good:                                       # frame-resolution bar under the row; a box around the consecutive thumbnails inside one block
            edges = np.flatnonzero(np.diff(np.r_[0, good[key].astype(int), 0]))
            for a_, b_ in zip(edges[::2], edges[1::2]):
                ta, tb = t_all[a_], t_all[b_ - 1]
                cv2.rectangle(strip, (int(x_of_t(ta)), y + TH + 2), (min(int(x_of_t(tb)) + 1, xL + n_tick * TW - 1), y + TH + 5), row_bgr[key], -1)
                ks = [k_ for k_, tk in enumerate(ticks) if ta <= tk <= tb]
                if ks:
                    cv2.rectangle(strip, (xL + ks[0] * TW - 1, y - 2), (xL + (ks[-1] + 1) * TW, y + TH + 7), row_bgr[key], 1)
    cv2.putText(strip, f"every {a.strip_step:g} s: what each view outputs (red = joints it cannot see), the merge, the truth.   Bars and boxes mark the well-tracked blocks "
                       "(a view: its legs usable; the merge: the frames the export keeps); thumbnails outside a block are dimmed", (xL, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    thumb_r.delete()
    show_charts, show_strip = a.readout in ("both", "charts"), a.readout in ("both", "strips")
    H = 20 + PH + 30 + RH + 20 + (TLH if show_charts else 0) + (SH + 10 if show_strip else 0)
    # ---- frames
    import av
    container = av.open(a.out, mode="w"); stream = container.add_stream("libx264", rate=int(round(fps / a.step))); stream.width, stream.height = W, H; stream.pix_fmt = "yuv420p"; stream.options = {"crf": "22"}
    gif_frames = []; g0, g1 = [float(x) for x in a.gif_span.split(",")]; err_hist = []
    for n, f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        if not ok:
            break
        canvas = np.full((H, W, 3), 28, np.uint8); t = t_all[f]
        x = 10
        for v in views:                                                          # top: the views
            x0, y0, x1, y1 = vbox[v]; crop = img[y0:y1, x0:x1].copy()
            kp = kp_draw[v][f]
            for a_, b_ in SKELETON:
                if a_ in FACE_KP or b_ in FACE_KP:
                    continue
                if np.isfinite(kp[[a_, b_], :2]).all() and kp[a_, 2] >= 0.3 and kp[b_, 2] >= 0.3:
                    cv2.line(crop, (int(kp[a_, 0] - x0), int(kp[a_, 1] - y0)), (int(kp[b_, 0] - x0), int(kp[b_, 1] - y0)), (40, 230, 40), 3, cv2.LINE_AA)
            crop = cv2.resize(crop, (pw[v], PH), interpolation=cv2.INTER_AREA)
            if light_rect and x0 <= light_rect[0] < x1 and y0 <= light_rect[1] < y1:
                s_ = PH / (y1 - y0); lx, ly = int((light_rect[0] - x0) * s_), int((light_rect[1] - y0) * s_); lw_ = max(int(light_rect[2] * s_), 6)
                cv2.rectangle(crop, (lx - 3, ly - 3), (lx + lw_ + 3, ly + lw_ + 3), (255, 255, 255), 1); label(crop, "sync light", (lx - 78, ly + lw_ + 2), scale=0.4)
            legs_ok = leg_ok[v][f]; n_us = int(usable[v][f].sum())
            col = (40, 200, 40) if legs_ok else ((30, 170, 240) if lifted[v][f] else (60, 60, 220))
            cv2.rectangle(crop, (0, 0), (pw[v] - 1, PH - 1), col, 4); canvas[20:20 + PH, x:x + pw[v]] = crop
            name = "along view" if v == A else "side view"
            status = ("subject not tracked here" if not lifted[v][f] else ("bridged frame (not used)" if not vis[v][f] else f"{n_us}/21 usable" + ("" if legs_ok else ", legs at edge")))
            label(canvas, f"{name}: {status}", (x + 8, 20 + PH - 10), col=col); x += pw[v] + 30
        yb = 20 + PH + 30; gi_f = gi.get(f)                                       # middle: truth and reconstruction
        if gi_f is not None:
            aa = gt["pose_aa"][gi_f]; Vt = verts(aa[:3], aa[3:66]); canvas[yb:yb + RH, 20:20 + RW] = render(Vt, np.tile(COL["truth"], (len(Vt), 1)))
        label(canvas, "rendered truth", (28, yb + RH - 12))
        xr = 20 + RW + 20
        if not (lifted[A][f] or lifted[B][f]):
            label(canvas, "no data: the subject is tracked in neither view", (xr + 8, yb + RH // 2), col=(60, 60, 220), scale=0.5)
        elif none[f].all():
            label(canvas, "no usable joint in either view", (xr + 8, yb + RH // 2), col=(60, 60, 220))
        else:
            rt = root_fused[f]
            Vr = verts(rt, merged_draw[f]); colours = np.tile(COL["ok"], (len(Vr), 1)); colours[np.isin(dom, np.flatnonzero(none[f]) + 1)] = COL["missing"]
            canvas[yb:yb + RH, xr:xr + RW] = render(Vr, colours)
            if gi_f is not None:
                e = geodesic(merged_out[f], gt["pose_aa"][gi_f, 3:66].reshape(21, 3)); le = np.nanmedian(e[LEGS]) if np.isfinite(e[LEGS]).any() else np.nan
                if np.isfinite(le):
                    err_hist.append(le)
                label(canvas, f"leg-joint error vs truth: {le:5.1f} deg" if np.isfinite(le) else "leg joints: no usable view", (xr + 8, yb + 24))
            src = []
            for grp, nm in ((LEGS, "legs"), ([12, 13, 15, 16, 17, 18, 19, 20], "arms"), ([2, 5, 8, 11, 14], "spine")):
                a_ok, b_ok = usable[A][f][grp].any(), usable[B][f][grp].any()
                src.append(f"{nm}: " + ("both" if a_ok and b_ok else ("along" if a_ok else ("side" if b_ok else "none"))))
            label(canvas, "reconstruction  " + " | ".join(src), (xr + 8, yb + RH - 12), scale=0.5)
        label(canvas, "red = no view could see this part (not output)", (xr + 8, yb + RH - 34), col=(90, 90, 230), scale=0.5)
        label(canvas, "2D skeletons: ViTPose keypoints (the lifter's input)", (10, 20 + PH + 8), scale=0.4, col=(200, 200, 200), bg=(28, 28, 28))
        xe = xr + RW + 20                                                           # the export: the frame only if kept, only the parts some view saw
        if export_ok[f]:
            Ve = verts(root_fused[f], np.nan_to_num(merged_out[f])); keep = ~np.isin(dom[faces], np.flatnonzero(none[f]) + 1).any(1)
            canvas[yb:yb + RH, xe:xe + RW] = render(Ve, np.tile(COL["ok"], (len(Ve), 1)), keep)
        else:
            canvas[yb:yb + RH, xe:xe + RW] = render(None, None)
            why = ("subject tracked in neither view" if not (lifted[A][f] or lifted[B][f]) else ("a leg joint seen by neither view" if not np.isfinite(merged_out[f][LEGS]).all() else f"run shorter than {MIN_RUN / fps:.1f} s"))
            label(canvas, "frame not exported:", (xe + 8, yb + RH // 2 - 22), col=(60, 60, 220), scale=0.5); label(canvas, why, (xe + 8, yb + RH // 2), col=(60, 60, 220), scale=0.5)
        label(canvas, f"frames kept so far: {export_ok[win & (tt <= t)].mean():.0%}", (xe + 8, yb + RH - 56), scale=0.5)
        label(canvas, "parts no view saw are absent", (xe + 8, yb + RH - 34), scale=0.5)
        label(canvas, "export  (what a user takes away)", (xe + 8, yb + RH - 12))
        xt = xe + RW + 20
        Vw = verts_world(gt["pose_aa"][gi_f, :3], gt["pose_aa"][gi_f, 3:66], gt["transl"][gi_f]) if gi_f is not None else None
        canvas[yb:yb + AH, xt:xt + AW] = render_aerial(Vw)
        label(canvas, "aerial truth  (blue = along camera, orange = side camera)", (xt + 6, yb + AH - 8), scale=0.42)
        lines = [f"IMU t = {t:6.2f} s   (video frame {f})", "3D reconstruction from two partial views:", "a joint is output only from the view(s) that",
                 "see it, cross-faded over 0.5 s at segment edges;", "nothing where neither view sees it.",
                 f"legs usable so far: along {leg_ok[A][win & (tt <= t)].mean():.0%}  side {leg_ok[B][win & (tt <= t)].mean():.0%}  either {(leg_ok[A] | leg_ok[B])[win & (tt <= t)].mean():.0%}"]
        if err_hist:
            lines += [f"median leg error so far: {np.nanmedian(err_hist):.1f} deg"]
        for k_, ln in enumerate(lines):
            cv2.putText(canvas, ln, (xt, yb + AH + 26 + 22 * k_), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)
        yt = yb + RH + 20; cx = cursor_x(t)                                       # bottom: the readout(s) + cursor
        if show_strip:
            canvas[yt:yt + SH] = strip
            k_cur = int(np.clip((t - t0) // a.strip_step, 0, max(n_tick - 1, 0))); x_c = int(L_ * W) + k_cur * TW
            cv2.rectangle(canvas, (x_c - 2, yt + 14), (x_c + TW + 1, yt + SH - 2), (255, 255, 255), 1); yt += SH + 10
        if show_charts:
            canvas[yt:yt + TLH] = timeline; cv2.line(canvas, (cx, yt + 8), (cx, yt + TLH - 24), (255, 255, 255), 2)
        for pkt in stream.encode(av.VideoFrame.from_ndarray(canvas[:, :, ::-1].copy(), format="rgb24")):
            container.mux(pkt)
        if a.gif and g0 <= t <= g1 and n % 3 == 0:
            gif_frames.append(cv2.resize(canvas, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
        if n % 300 == 0:
            print(f"  {n}/{len(frames)} frames", flush=True)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close(); renderer.delete(); aerial_r.delete()
    print(f"video -> {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB, {len(frames)} frames, {W}x{H})")
    if a.gif and gif_frames:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        for i, fr_ in enumerate(gif_frames):
            cv2.imwrite(str(tmpdir / f"g{i:04d}.png"), fr_)
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "10", "-i", str(tmpdir / "g%04d.png"), "-vf", "palettegen=max_colors=128", str(tmpdir / "pal.png")], check=True)
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "10", "-i", str(tmpdir / "g%04d.png"), "-i", str(tmpdir / "pal.png"), "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", a.gif], check=True)
        print(f"gif -> {a.gif} ({Path(a.gif).stat().st_size / 1e6:.1f} MB, {len(gif_frames)} frames, IMU {g0}-{g1} s)")


if __name__ == "__main__":
    main()
