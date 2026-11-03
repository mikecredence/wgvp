"""Demo video of the two-view reconstruction on HC901: what each camera saw, what each view could use, and the body
rebuilt on the shared IMU timeline next to the rendered truth.

    python tools/make_demo.py <data root with wgvp_output/HC901> --out docs/figures/two_view_demo.mp4 [--gif docs/figures/two_view_demo.gif]

Layout per frame (IMU time t):
  top     the along view and the side view at t, with the chosen subject's 2D skeleton; the border says whether the
          view is usable for the legs at t (green), present but with the leg joints at a frame edge (amber), or
          without the subject (red)
  bottom  left: the rendered truth. Right: the reconstruction by the segment rule (run-time visibility): every joint
          comes from the view(s) whose ViTPose keypoints for it are inside the frame, averaged when both see it;
          body parts driven by joints that no view could see are drawn red and are NOT part of the output (they are
          filled with the plain two-view mean only so the mesh can be drawn). When neither view has the body at all
          the panel says so and nothing is drawn.
  strip   the usable segments of each view for the legs, the fused coverage, and the time cursor
The root orientation of the reconstruction is the visibility-weighted mean of the two views' roots after each
view's constant heading offset to the truth has been removed (the two cameras' world frames differ by that
heading); the root position is not part of the experiment and both bodies are drawn at the origin.
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
from tools.two_view_fusion import JOINT_KP, LEGS, chordal_mean, fit_yaw, geodesic, usable_joints_rt   # noqa: E402
from wgvp.tracking import FACE_KP, SKELETON                                                            # noqa: E402

# SMPL-X body joint index (0-20, i.e. SMPL joints 1-21) that drives each SMPL-X skinning joint 1-21; parts of the
# mesh are coloured by the joint with the largest skinning weight
PART_COLOURS = {"ok": [150, 160, 200], "missing": [220, 70, 70], "truth": [170, 170, 170]}


def load_views(deliver: Path):
    views = sorted(p.name for p in deliver.iterdir() if (p / "smpl_global.npz").exists())
    Z = {v: np.load(deliver / v / "smpl_global.npz") for v in views}
    K = {v: np.load(deliver / v / "kp2d_vitpose.npz") for v in views}
    S = {v: np.load(deliver / v / "subject_track.npz") for v in views}
    return views, Z, K, S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("--session", default="HC901"); ap.add_argument("--recording", default="hc901_selfpace")
    ap.add_argument("--video", default=str(PKG / "sample_v2/data/HC901/SP/HC901_SP_DeID.mp4")); ap.add_argument("--gt", default=str(PKG / "sample_v2/HC901_gt.npz"))
    ap.add_argument("--body-models", default=str(Path.home() / "data/WearGait/body_models")); ap.add_argument("--out", default=str(PKG / "docs/figures/two_view_demo.mp4"))
    ap.add_argument("--gif", default=None, help="also write a GIF excerpt"); ap.add_argument("--gif-span", default="31,43", help="IMU seconds for the GIF excerpt")
    ap.add_argument("--step", type=int, default=1, help="render every n-th frame"); ap.add_argument("--t0", type=float, default=None); ap.add_argument("--t1", type=float, default=None)
    a = ap.parse_args()
    deliver = Path(a.root) / "wgvp_output" / a.session / "deliver" / a.recording
    views, Z, K, S = load_views(deliver); A, B = views
    gt = np.load(a.gt); gi = {int(f): i for i, f in enumerate(gt["frame_idx"])}
    sync = json.loads((deliver / "sync.json").read_text()); offset = float(sync["offset_s"])
    # ---- per view, per combined-video frame: estimate, usable joints (run-time), presence
    per = {}
    for v in views:
        z, k = Z[v], K[v]; fr = z["frame_idx"].astype(int)
        us = usable_joints_rt(k["kp2d"], k["view_box"] if "view_box" in k.files else None, z["visible"])
        per[v] = {"idx": {int(f): i for i, f in enumerate(fr)}, "body": z["body_pose"].reshape(-1, 21, 3), "root": z["global_orient"], "usable": us, "visible": z["visible"].astype(bool),
                  "vbox": [int(x) for x in k["view_box"]], "sk": {int(f): kp for f, kp in zip(S[v]["frame_idx"], S[v]["kp2d"])}}
    # ---- heading offsets per view (constant yaw to the truth), on frames both have
    common = sorted(set(gi) & set(per[A]["idx"]) & set(per[B]["idx"]))
    gt_root = gt["pose_aa"][[gi[f] for f in common], :3]
    yaw = {v: fit_yaw(per[v]["root"][[per[v]["idx"][f] for f in common]], gt_root, per[v]["visible"][[per[v]["idx"][f] for f in common]]) for v in views}
    print("heading offsets (deg):", yaw)
    # ---- the frames to draw: the recording window
    cap = cv2.VideoCapture(a.video); n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = 30.0
    t_all = np.arange(n_video) / fps - offset
    t0 = a.t0 if a.t0 is not None else 0.0; t1 = a.t1 if a.t1 is not None else float(gt["imu_t_s"][-1])
    frames = [f for f in range(n_video) if t0 <= t_all[f] <= t1][::a.step]
    # ---- SMPL-X + renderer
    os.environ.pop("PYOPENGL_PLATFORM", None)
    import torch, smplx, trimesh, pyrender
    bm = smplx.create(a.body_models, model_type="smplx", gender="neutral", use_pca=False, batch_size=1); faces = bm.faces.astype(np.int32)
    W_ = bm.lbs_weights.detach().cpu().numpy()                                  # (V, J) skinning weights
    dom = W_[:, :22].argmax(1)                                                   # dominant skinning joint per vertex (0 = pelvis, 1..21 = body joints)
    def verts(root_aa, body_aa):
        with torch.no_grad():
            o = bm(global_orient=torch.tensor(root_aa, dtype=torch.float32)[None], body_pose=torch.tensor(body_aa.reshape(63), dtype=torch.float32)[None], betas=torch.zeros(1, 10), transl=torch.zeros(1, 3))
        return o.vertices[0].numpy()
    RW, RH = 420, 560
    renderer = pyrender.OffscreenRenderer(RW, RH)
    def render(V, colours):
        sc = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.45, 0.45, 0.45))
        tm = trimesh.Trimesh(V, faces, process=False); tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=np.c_[colours, np.full(len(V), 255)].astype(np.uint8))
        sc.add(pyrender.Mesh.from_trimesh(tm, smooth=True))
        c = np.array([0.0, V[:, 1].mean(), 0.0]); az = np.deg2rad(40); eye = c + 3.3 * np.array([np.sin(az), 0.12, np.cos(az)])
        fwd = eye - c; fwd /= np.linalg.norm(fwd); right = np.cross([0, 1, 0], fwd); right /= np.linalg.norm(right); up = np.cross(fwd, right)
        M = np.eye(4); M[:3, 0] = right; M[:3, 1] = up; M[:3, 2] = fwd; M[:3, 3] = eye
        sc.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(40)), pose=M)
        for p_ in ((2, 4, 3), (-2, 3, -2)):
            Lm = np.eye(4); Lm[:3, 3] = p_; sc.add(pyrender.DirectionalLight(intensity=2.5), pose=Lm)
        col, _ = renderer.render(sc, flags=pyrender.RenderFlags.RGBA); col = col.astype(np.float32); al = col[..., 3:4] / 255.0
        return (col[..., :3] * al + 255 * (1 - al)).astype(np.uint8)[:, :, ::-1]      # BGR for cv2
    # ---- layout
    PH = 360; pw = {v: int(round(PH * (per[v]["vbox"][2] - per[v]["vbox"][0]) / (per[v]["vbox"][3] - per[v]["vbox"][1]))) for v in views}
    top_w = pw[A] + pw[B] + 30; W = max(top_w, 2 * RW + 3 * 20 + 360); H = PH + 40 + RH + 40 + 90
    def label(img, txt, xy, col=(255, 255, 255), scale=0.6, th=2, bg=(0, 0, 0)):
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, th); x, y = xy
        cv2.rectangle(img, (x - 3, y - tht - 5), (x + tw + 3, y + 4), bg, -1); cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)
    # usable-for-legs timeline per view, fused coverage, over all frames in the window
    leg_ok = {v: np.array([per[v]["usable"][per[v]["idx"][f]][LEGS].all() if f in per[v]["idx"] else False for f in range(n_video)]) for v in views}
    any_leg = leg_ok[A] | leg_ok[B]
    w_frames = np.array([t0 <= t_all[f] <= t1 for f in range(n_video)])
    writer = None; tmpdir = Path(tempfile.mkdtemp()); gif_frames = []; g0, g1 = [float(x) for x in a.gif_span.split(",")]
    err_hist = []
    for n, f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        if not ok:
            break
        canvas = np.full((H, W, 3), 28, np.uint8); t = t_all[f]
        # --- top: the two views with the subject skeleton
        x = 10
        for v in views:
            x0, y0, x1, y1 = per[v]["vbox"]; crop = img[y0:y1, x0:x1].copy()
            kp = per[v]["sk"].get(f)
            if kp is not None:
                for a_, b_ in SKELETON:
                    if a_ in FACE_KP or b_ in FACE_KP:
                        continue
                    if np.isfinite(kp[[a_, b_], :2]).all() and kp[a_, 2] >= 0.3 and kp[b_, 2] >= 0.3:
                        cv2.line(crop, (int(kp[a_, 0] - x0), int(kp[a_, 1] - y0)), (int(kp[b_, 0] - x0), int(kp[b_, 1] - y0)), (40, 230, 40), 3, cv2.LINE_AA)
            crop = cv2.resize(crop, (pw[v], PH), interpolation=cv2.INTER_AREA)
            i = per[v]["idx"].get(f); detected = i is not None and per[v]["visible"][i]
            legs_ok = i is not None and per[v]["usable"][i][LEGS].all()
            n_us = int(per[v]["usable"][i].sum()) if i is not None else 0
            col = (40, 200, 40) if legs_ok else ((30, 170, 240) if i is not None else (60, 60, 220))
            cv2.rectangle(crop, (0, 0), (pw[v] - 1, PH - 1), col, 4)
            canvas[20:20 + PH, x:x + pw[v]] = crop
            name = "along view" if v == A else "side view"
            status = ("subject not in view" if i is None else ("subject not detected here (bridged frame, not used)" if not detected else
                      f"{n_us}/21 joints usable" + ("" if legs_ok else "; legs at a frame edge")))
            label(canvas, f"{name}: {status}", (x + 8, 20 + PH - 10), col=col); x += pw[v] + 30
        # --- bottom: truth and reconstruction
        gi_f = gi.get(f); yb = 20 + PH + 40
        if gi_f is not None:
            aa = gt["pose_aa"][gi_f]; Vt = verts(aa[:3], aa[3:66]); canvas[yb:yb + RH, 20:20 + RW] = render(Vt, np.tile(PART_COLOURS["truth"], (len(Vt), 1)))
        label(canvas, "rendered truth", (28, yb + RH - 12))
        iA, iB = per[A]["idx"].get(f), per[B]["idx"].get(f)
        xr = 20 + RW + 20
        if iA is None and iB is None:
            label(canvas, "no data: the subject is in neither view", (xr + 8, yb + RH // 2), col=(60, 60, 220))
        else:
            ests, ws, roots, rws = [], [], [], []
            for v, i in ((A, iA), (B, iB)):
                if i is None:
                    continue
                ests.append(per[v]["body"][i][None]); ws.append(per[v]["usable"][i].astype(float)[None])
                roots.append((R.from_euler("y", yaw[v], degrees=True) * R.from_rotvec(per[v]["root"][i])).as_rotvec()); rws.append(float(per[v]["usable"][i].sum()) + 1e-3)
            if len(ests) == 1:
                body = ests[0][0]; wsum = ws[0][0]
            else:
                wsum = ws[0][0] + ws[1][0]
                body = chordal_mean([ests[0], ests[1]], np.stack([ws[0], ws[1]]))[0]
                body_mean = chordal_mean([ests[0], ests[1]], np.ones((2, 1, 21)))[0]
                body = np.where((wsum > 0)[:, None], body, body_mean)                     # the mean fills what no view saw, drawn red
            missing = wsum <= 0
            if len(roots) == 1:
                root = roots[0]
            else:
                # the root from the view with more usable joints; averaged with the other only when the two agree
                # (a view that sees only the legs flips its root front-to-back in a few percent of frames)
                disagree = np.degrees((R.from_rotvec(roots[0]).inv() * R.from_rotvec(roots[1])).magnitude())
                if disagree > 60:
                    root = roots[int(np.argmax(rws))]
                else:
                    q = R.from_rotvec(np.stack(roots)).as_quat(); q[1] *= np.sign(q[0] @ q[1]); root = R.from_quat((rws[0] * q[0] + rws[1] * q[1]) / (rws[0] + rws[1])).as_rotvec()
            if missing.all():
                label(canvas, "no usable joint in either view", (xr + 8, yb + RH // 2), col=(60, 60, 220))
            else:
                Vr = verts(root, body)
                colours = np.tile(PART_COLOURS["ok"], (len(Vr), 1)); miss_v = np.isin(dom, np.flatnonzero(missing) + 1); colours[miss_v] = PART_COLOURS["missing"]
                canvas[yb:yb + RH, xr:xr + RW] = render(Vr, colours)
                if gi_f is not None:
                    e = geodesic(np.where(missing[:, None], np.nan, body), gt["pose_aa"][gi_f, 3:66].reshape(21, 3)); le = np.nanmedian(e[LEGS]) if np.isfinite(e[LEGS]).any() else np.nan
                    err_hist.append(le)
                    label(canvas, f"leg-joint error vs truth: {le:5.1f} deg" if np.isfinite(le) else "leg joints: no usable view", (xr + 8, yb + 26), col=(255, 255, 255), scale=0.55, th=1)
            src = []
            for grp, name in ((LEGS, "legs"), ([12, 13, 15, 16, 17, 18, 19, 20], "arms"), ([2, 5, 8, 11, 14], "spine")):
                a_ok = iA is not None and per[A]["usable"][iA][grp].any(); b_ok = iB is not None and per[B]["usable"][iB][grp].any()
                src.append(f"{name}: " + ("both views" if a_ok and b_ok else ("along view" if a_ok else ("side view" if b_ok else "none"))))
            label(canvas, "reconstruction, segment rule   " + " | ".join(src), (xr + 8, yb + RH - 12), scale=0.5, th=1)
        label(canvas, "red = no view could see this part (not output)", (xr + 8, yb + RH - 34), col=(90, 90, 230), scale=0.5, th=1)
        # --- text panel on the right
        xt = xr + RW + 20
        lines = [f"IMU t = {t:6.2f} s", f"video frame {f}", "", "segment rule:", "a joint is output only from", "views that see it; both ->", "average; neither -> no output", "",
                 f"usable for the legs so far:", f"  along view  {leg_ok[A][w_frames & (t_all <= t)].mean():.0%}", f"  side view   {leg_ok[B][w_frames & (t_all <= t)].mean():.0%}",
                 f"  either      {any_leg[w_frames & (t_all <= t)].mean():.0%}"]
        if err_hist:
            lines += ["", f"median leg error so far: {np.nanmedian(err_hist):.1f} deg"]
        for k_, ln in enumerate(lines):
            cv2.putText(canvas, ln, (xt, yb + 26 + 24 * k_), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
        # --- timeline strip
        ys = yb + RH + 30; xs0, xs1 = 20, W - 20; span = (t1 - t0) or 1.0
        def band(y, mask, col, name):
            cv2.putText(canvas, name, (xs0, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            idx = np.flatnonzero(mask & w_frames)
            for i0 in idx:
                px = int(xs0 + (t_all[i0] - t0) / span * (xs1 - xs0)); cv2.line(canvas, (px, y), (px, y + 10), col, 1)
        band(ys, leg_ok[A], (40, 200, 40), "along view usable (legs)"); band(ys + 26, leg_ok[B], (40, 200, 40), "side view usable (legs)"); band(ys + 52, any_leg, (255, 200, 60), "output (either view)")
        px = int(xs0 + (t - t0) / span * (xs1 - xs0)); cv2.line(canvas, (px, ys - 14), (px, ys + 64), (255, 255, 255), 2)
        if writer is None:
            import av
            container = av.open(a.out, mode="w"); stream = container.add_stream("libx264", rate=int(round(fps / a.step))); stream.width, stream.height = W, H; stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "22"}; writer = (container, stream)
        container, stream = writer
        for pkt in stream.encode(av.VideoFrame.from_ndarray(canvas[:, :, ::-1].copy(), format="rgb24")):
            container.mux(pkt)
        if a.gif and g0 <= t <= g1 and (n % 3 == 0):
            gif_frames.append(cv2.resize(canvas, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
        if n % 200 == 0:
            print(f"  {n}/{len(frames)} frames", flush=True)
    container, stream = writer
    for pkt in stream.encode():
        container.mux(pkt)
    container.close(); renderer.delete()
    print(f"video -> {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
    if a.gif and gif_frames:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        for i, fr in enumerate(gif_frames):
            cv2.imwrite(str(tmpdir / f"g{i:04d}.png"), fr)
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "10", "-i", str(tmpdir / "g%04d.png"), "-vf", "palettegen=max_colors=128", str(tmpdir / "pal.png")], check=True)
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "10", "-i", str(tmpdir / "g%04d.png"), "-i", str(tmpdir / "pal.png"), "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", a.gif], check=True)
        print(f"gif -> {a.gif} ({Path(a.gif).stat().st_size / 1e6:.1f} MB, {len(gif_frames)} frames, IMU {g0}-{g1} s)")


if __name__ == "__main__":
    main()
