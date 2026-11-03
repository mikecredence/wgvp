"""Demo video: 3D reconstruction from two partial views (HC901).

    python tools/make_demo.py <data root with wgvp_output/HC901> --out demo.mp4 [--gif docs/figures/two_view_demo.gif]

Layout per frame (IMU time t):
  top      the along view and the side view at t with the chosen subject's 2D skeleton; the border says whether the
           view is usable for the legs at t (green), lifted but with the leg joints at a frame edge (amber), or
           without the subject (red)
  middle   left: the rendered truth. Right: the reconstruction by the cross-faded segment rule: every joint comes
           from the view(s) whose ViTPose keypoints for it are inside the frame, with the weights ramped over the
           first and last half-second of each usable segment so hand-overs between views are continuous, averaged
           where both qualify. Body parts driven by joints that no view could see are drawn red and are NOT part of
           the output (filled only so the mesh can be drawn). When the body is in neither view nothing is drawn.
  bottom   the readout as timelines on the shared IMU axis: what each view outputs (left and right knee flexion,
           thin blue / orange), what the merge outputs (green, gaps where no view saw the joint) and the truth
           (black), under strips of each view's usable segments; a cursor marks t.
The reconstruction's root comes from the view with more usable joints when the two views' roots disagree by more
than 60 deg, else their visibility-weighted mean, after each view's constant heading offset to the truth has been
removed (the two cameras' world frames differ by that heading). Root position is not part of the experiment; both
bodies are drawn at the origin.
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
from tools.two_view_fusion import LEGS, chordal_mean, fit_yaw, geodesic, ramp_inside, usable_joints_rt   # noqa: E402
from wgvp.tracking import FACE_KP, SKELETON                                                             # noqa: E402

COL = {"ok": [150, 160, 200], "missing": [220, 70, 70], "truth": [170, 170, 170]}
RAMP_FRAMES = 15


def knee_deg(body):   # (T, 21, 3) -> (T, 2) left, right knee flexion magnitude
    return np.degrees(np.linalg.norm(body[:, [3, 4]], axis=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("--session", default="HC901"); ap.add_argument("--recording", default="hc901_selfpace")
    ap.add_argument("--video", default=str(PKG / "sample_v2/data/HC901/SP/HC901_SP_DeID.mp4")); ap.add_argument("--gt", default=str(PKG / "sample_v2/HC901_gt.npz"))
    ap.add_argument("--body-models", default=str(Path.home() / "data/WearGait/body_models")); ap.add_argument("--out", default=str(PKG / "docs/figures/two_view_demo.mp4"))
    ap.add_argument("--gif", default=None); ap.add_argument("--gif-span", default="28,42"); ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--t0", type=float, default=None); ap.add_argument("--t1", type=float, default=None)
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
    # ---- the merge, dense: cross-faded segment rule
    w = {v: ramp_inside(usable[v], RAMP_FRAMES) for v in views}
    est = {v: np.nan_to_num(body[v]) for v in views}
    merged = chordal_mean([est[A], est[B]], np.stack([w[A], w[B]]))
    none = ~(usable[A] | usable[B]); fill = chordal_mean([est[A], est[B]], np.stack([lifted[A][:, None].repeat(21, 1).astype(float), lifted[B][:, None].repeat(21, 1).astype(float)]))
    merged_out = merged.copy(); merged_out[none] = np.nan                      # the output: nothing where no view saw the joint
    merged_draw = np.where(none[:, :, None], fill, merged)                       # for drawing only: unseen parts filled, shown red
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
    RW, RH = 400, 520; AW, AH = 360, 250
    renderer = pyrender.OffscreenRenderer(RW, RH); aerial_r = pyrender.OffscreenRenderer(AW, AH)
    # the scene's mat: the walk's extent along x (from the sample's expected answers) and the mat's lateral position
    exp = json.loads((Path(a.gt).parent / "expected.json").read_text()) if (Path(a.gt).parent / "expected.json").exists() else {}
    x0m, x1m = exp.get("scene", {}).get("walk_extent_m", [-2.45, 2.45]); zm = float(np.median(gt["transl"][:, 2])) if "transl" in gt.files else 0.0
    cams = {v: gt[f"{v}_cam_pose"] for v in views if f"{v}_cam_pose" in gt.files}
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
    def body_mesh(V, colours):
        tm = trimesh.Trimesh(V, faces, process=False); tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=np.c_[colours, np.full(len(V), 255)].astype(np.uint8))
        return pyrender.Mesh.from_trimesh(tm, smooth=True)
    def render(V, colours):
        """Close-up: the body at the origin on a floor with the mat running along x under it."""
        sc = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.45, 0.45, 0.45)); sc.add(body_mesh(V, colours))
        add_floor_and_mat(sc, float(V[:, 1].min()) - 0.01, 0.0, 2.45, 0.0)
        c = np.array([0.0, V[:, 1].mean(), 0.0]); az = np.deg2rad(40); eye = c + 3.3 * np.array([np.sin(az), 0.25, np.cos(az)])
        sc.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(40)), pose=look_at(eye, c)); lights(sc); return finish(renderer, sc)
    def render_aerial(V_world):
        """The whole scene from above: floor, mat, the truth body at its real position, the two cameras."""
        sc = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.5, 0.5, 0.5))
        add_floor_and_mat(sc, 0.0, 0.5 * (x0m + x1m), 0.5 * (x1m - x0m), zm)
        if V_world is not None:
            sc.add(body_mesh(V_world, np.tile(COL["truth"], (len(V_world), 1))))
        for v, cm in cams.items():
            box = trimesh.creation.box(extents=(0.25, 0.25, 0.25)); box.apply_translation(cm[:3, 3]); box.visual = trimesh.visual.ColorVisuals(mesh=box, face_colors=np.tile([60, 110, 230, 255] if v == A else [240, 150, 40, 255], (len(box.faces), 1)))
            sc.add(pyrender.Mesh.from_trimesh(box, smooth=False))
        cx = 0.5 * (x0m + x1m); eye = [cx, 12.0, zm - 2.6]
        sc.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(50)), pose=look_at(eye, [cx, 0.0, zm - 2.4], up=(0, 0, -1))); lights(sc); return finish(aerial_r, sc)
    def verts_world(root_aa, body_aa, transl):
        with torch.no_grad():
            o = bm(global_orient=torch.tensor(root_aa, dtype=torch.float32)[None], body_pose=torch.tensor(body_aa.reshape(63), dtype=torch.float32)[None], betas=torch.zeros(1, 10), transl=torch.tensor(transl, dtype=torch.float32)[None])
        return o.vertices[0].numpy()
    # ---- layout
    PH = 320; vbox = {v: [int(x) for x in K[v]["view_box"]] for v in views}
    pw = {v: int(round(PH * (vbox[v][2] - vbox[v][0]) / (vbox[v][3] - vbox[v][1]))) for v in views}
    W = max(pw[A] + pw[B] + 30 + 20, 2 * RW + 80 + AW); TLH = 330; H = 20 + PH + 30 + RH + 20 + TLH
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
            sk = S[v]; hit = np.flatnonzero(sk["frame_idx"] == f)
            if len(hit):
                kp = sk["kp2d"][hit[0]]
                for a_, b_ in SKELETON:
                    if a_ in FACE_KP or b_ in FACE_KP:
                        continue
                    if np.isfinite(kp[[a_, b_], :2]).all() and kp[a_, 2] >= 0.3 and kp[b_, 2] >= 0.3:
                        cv2.line(crop, (int(kp[a_, 0] - x0), int(kp[a_, 1] - y0)), (int(kp[b_, 0] - x0), int(kp[b_, 1] - y0)), (40, 230, 40), 3, cv2.LINE_AA)
            crop = cv2.resize(crop, (pw[v], PH), interpolation=cv2.INTER_AREA)
            legs_ok = leg_ok[v][f]; n_us = int(usable[v][f].sum())
            col = (40, 200, 40) if legs_ok else ((30, 170, 240) if lifted[v][f] else (60, 60, 220))
            cv2.rectangle(crop, (0, 0), (pw[v] - 1, PH - 1), col, 4); canvas[20:20 + PH, x:x + pw[v]] = crop
            name = "along view" if v == A else "side view"
            status = ("subject not in view" if not lifted[v][f] else ("not detected (bridged, not used)" if not vis[v][f] else f"{n_us}/21 joints usable" + ("" if legs_ok else "; legs at edge")))
            label(canvas, f"{name}: {status}", (x + 8, 20 + PH - 10), col=col); x += pw[v] + 30
        yb = 20 + PH + 30; gi_f = gi.get(f)                                       # middle: truth and reconstruction
        if gi_f is not None:
            aa = gt["pose_aa"][gi_f]; Vt = verts(aa[:3], aa[3:66]); canvas[yb:yb + RH, 20:20 + RW] = render(Vt, np.tile(COL["truth"], (len(Vt), 1)))
        label(canvas, "rendered truth", (28, yb + RH - 12))
        xr = 20 + RW + 20
        if not (lifted[A][f] or lifted[B][f]):
            label(canvas, "no data: the subject is in neither view", (xr + 8, yb + RH // 2), col=(60, 60, 220))
        elif none[f].all():
            label(canvas, "no usable joint in either view", (xr + 8, yb + RH // 2), col=(60, 60, 220))
        else:
            roots, rws = [], []
            for v in views:
                if lifted[v][f]:
                    roots.append((R.from_euler("y", yaw[v], degrees=True) * R.from_rotvec(root[v][f])).as_rotvec()); rws.append(float(usable[v][f].sum()) + 1e-3)
            if len(roots) == 1:
                rt = roots[0]
            elif np.degrees((R.from_rotvec(roots[0]).inv() * R.from_rotvec(roots[1])).magnitude()) > 60:
                rt = roots[int(np.argmax(rws))]
            else:
                q = R.from_rotvec(np.stack(roots)).as_quat(); q[1] *= np.sign(q[0] @ q[1]); rt = R.from_quat((rws[0] * q[0] + rws[1] * q[1]) / (rws[0] + rws[1])).as_rotvec()
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
        xt = xr + RW + 20
        Vw = verts_world(gt["pose_aa"][gi_f, :3], gt["pose_aa"][gi_f, 3:66], gt["transl"][gi_f]) if gi_f is not None else None
        canvas[yb:yb + AH, xt:xt + AW] = render_aerial(Vw)
        label(canvas, "aerial truth: mat, walker, along cam (blue), side cam (orange)", (xt + 6, yb + AH - 8), scale=0.42)
        lines = [f"IMU t = {t:6.2f} s   (video frame {f})", "3D reconstruction from two partial views:", "a joint is output only from the view(s) that",
                 "see it, cross-faded over 0.5 s at segment edges;", "nothing where neither view sees it.",
                 f"legs usable so far: along {leg_ok[A][win & (tt <= t)].mean():.0%}  side {leg_ok[B][win & (tt <= t)].mean():.0%}  either {(leg_ok[A] | leg_ok[B])[win & (tt <= t)].mean():.0%}"]
        if err_hist:
            lines += [f"median leg error so far: {np.nanmedian(err_hist):.1f} deg"]
        for k_, ln in enumerate(lines):
            cv2.putText(canvas, ln, (xt, yb + AH + 26 + 22 * k_), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)
        yt = yb + RH + 20; canvas[yt:yt + TLH] = timeline                        # bottom: the readout timelines + cursor
        cx = cursor_x(t); cv2.line(canvas, (cx, yt + 8), (cx, yt + TLH - 24), (255, 255, 255), 2)
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
