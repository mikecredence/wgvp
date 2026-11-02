"""(Maintainers.) Regenerate the README figures from a dry-run output folder.

    python tools/make_readme_figures.py <dry_run_out> [--gpu-deliver <.../deliver/hc900_selfpace>] [--body-models <dir>]

<dry_run_out> is the folder given to `python -m wgvp dry-run --out ...` (it holds HC900/ and wgvp_output/). The two
GVHMR figures need a dry run that included the GPU stage (or the deliver folder of one, --gpu-deliver); the mesh
strip additionally needs smplx, trimesh, pyrender and the SMPL-X body model folder (--body-models, containing
smplx/SMPLX_NEUTRAL.npz). Everything else needs only the CPU environment plus matplotlib.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from wgvp.imu_ref import load_imu_ref                        # noqa: E402
from wgvp.selection import DEFAULTS, score_fragments         # noqa: E402
from wgvp.signals import bandpass, resample_to_grid          # noqa: E402
from wgvp.tracking import FACE_KP, SKELETON, TrackSet, hip_vertical_signal   # noqa: E402

OUT = PKG / "docs" / "figures"
GREEN, GREY = (40, 230, 40), (200, 200, 200)
PANEL_COL = {"left": (90, 200, 90), "right": (60, 160, 255)}


def load_run(d: Path):
    w = d / "wgvp_output/HC900/work"; dl = d / "wgvp_output/HC900/deliver/hc900_selfpace"
    ts = {v: TrackSet.load(w / f"HC900_SP_DeID/tracks_{v}.npz") for v in ("left", "right")}
    imu = load_imu_ref(w / "imu_ref/hc900_selfpace_imu_ref.npz")
    sel = json.loads((dl / "selection_joint.json").read_text())
    pan = json.loads((w / "HC900_SP_DeID/panels.json").read_text())
    exp = json.loads((PKG / "sample/expected.json").read_text())
    return ts, imu, sel, pan, exp, dl


def fragment_roles(ts, sel, exp) -> dict:
    """subject = chosen by the pipeline; co-walker = fragment whose hips follow the rendered decoy's ground truth."""
    roles = {}
    for v, t in ts.items():
        chosen = set(sel["views"][v]["track_ids"])
        dlut = {int(f): (x, y) for f, x, y in exp["views"][v]["decoy_hip_px"]}
        for tid in t.track_ids():
            tr = t.track(int(tid)); d = []
            for f, kp in zip(tr["frame_idx"], tr["kp"]):
                if int(f) in dlut and (kp[[11, 12], 2] >= 0.3).all():
                    d.append(np.hypot(*(kp[[11, 12], :2].mean(0) - dlut[int(f)])))
            roles[(v, int(tid))] = "subject" if int(tid) in chosen else ("co-walker" if len(d) >= 5 and np.median(d) < 40 else "other person")
    return roles


def draw_frame(cap, frame_idx, ts, sel, pan, roles, L, labels=True):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx); ok, img = cap.read(); assert ok
    for p in pan["panels"]:
        x0, y0, x1, y1 = p["box"]; cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), PANEL_COL[p["id"]], 3)
        cv2.putText(img, f"{p['id']} view {x1 - x0}x{y1 - y0}", (x0 + 8, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, PANEL_COL[p["id"]], 2, cv2.LINE_AA)
    todo = []
    for v, t in ts.items():
        pos = int(np.where(t.frame_idx == frame_idx)[0][0]); m = t.det_frame == pos
        for tid, box, kp in zip(t.det_tid[m], t.det_box[m], t.det_kp[m]):
            role = roles[(v, int(tid))]; subj = role == "subject"; col = GREEN if subj else GREY
            x0, y0, x1, y1 = box.astype(int); cv2.rectangle(img, (x0, y0), (x1, y1), col, 2 if subj else 1)
            for a, b in SKELETON:
                if a in FACE_KP or b in FACE_KP:
                    continue
                if kp[a, 2] >= .3 and kp[b, 2] >= .3:
                    cv2.line(img, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), col, 3 if subj else 2, cv2.LINE_AA)
            fr = sel["views"][v]["fragments"].get(str(int(tid)), {}); r = fr.get("r_at_lag")
            txt = f"{role}  r={r:.2f}" if r is not None else f"{role}  ({fr.get('reason') or 'unscored'})"
            todo.append((subj, txt, col, (x0, y0, x1, y1)))
    if labels:   # others first, the subject last and below its box, so no label hides another
        taken = []
        for subj, txt, col, (x0, y0, x1, y1) in sorted(todo, key=lambda z: z[0]):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = min(img.shape[0] - 6, y1 + th + 10) if subj else max(th + 6, y0 - 8)
            for _ in range(4):
                rect = (x0, ty - th - 6, x0 + tw + 6, ty + 4)
                if not any(rect[0] < q[2] and rect[2] > q[0] and rect[1] < q[3] and rect[3] > q[1] for q in taken):
                    break
                ty += th + 12
            taken.append(rect)
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), -1)
            cv2.putText(img, txt, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    t_s = frame_idx / 30.0
    bar = np.zeros((34, img.shape[1], 3), np.uint8)
    cv2.putText(bar, f"frame {frame_idx}   video t = {t_s:.1f} s   IMU t = {t_s - L:+.1f} s     green = chosen subject, grey = other people; r = pelvis correlation with the IMU at the shared lag",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return np.concatenate([img, bar], 0)


def fig_two_views(d, ts, sel, pan, roles):
    cap = cv2.VideoCapture(str(d / "HC900/SP/HC900_SP_DeID.mp4")); L = sel["offset_s"]
    fig = np.concatenate([draw_frame(cap, f, ts, sel, pan, roles, L) for f in (560, 1250)], 0)
    cv2.imwrite(str(OUT / "two_views_tracking.png"), fig, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return cap


def fig_gif(d, cap, ts, sel, pan, roles):
    import imageio_ffmpeg
    L = sel["offset_s"]; ff = imageio_ffmpeg.get_ffmpeg_exe(); tmp = Path(tempfile.mkdtemp())
    for k, fi in enumerate(range(430, 790, 3)):
        im = draw_frame(cap, fi, ts, sel, pan, roles, L, labels=False)[:600]
        im = cv2.resize(im, (900, int(900 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f"IMU t = {fi / 30 - L:+.1f} s   green = subject, grey = other people", (8, im.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(tmp / f"f{k:04d}.png"), im)
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "10", "-i", str(tmp / "f%04d.png"), "-vf", "palettegen=max_colors=96", str(tmp / "pal.png")], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "10", "-i", str(tmp / "f%04d.png"), "-i", str(tmp / "pal.png"), "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", str(OUT / "tracking.gif")], check=True)


def running_max(x, k):
    x = np.nan_to_num(np.asarray(x, float), nan=-1.0); out = np.full_like(x, -1.0)
    for i in range(len(x)):
        out[i] = x[max(0, i - k):i + k + 1].max()
    return out


def fig_selection(ts, imu, sel, roles, dl):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    L = sel["offset_s"]; fs = 100.0
    scored = {v: score_fragments(t, imu, DEFAULTS) for v, t in ts.items()}
    walk = [(a, b) for lab, a, b in imu["events"] if lab == "Walk"]
    t0, t1 = walk[1][0] + 0.5, walk[1][0] + 5.5            # inside the second walk bout: both views see the subject
    fig, ax = plt.subplots(3, 1, figsize=(11, 10.5), gridspec_kw={"height_ratios": [1.0, 1.2, 0.75]})
    ref = bandpass(imu["disp_up"].astype(float), fs); g = imu["t"]; w = (g >= t0) & (g <= t1)
    z = lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    ax[0].plot(g[w], z(ref[w]), color="k", lw=1.8, label="IMU: pelvis vertical displacement (LowerBack_FreeAcc_U integrated twice)")
    styles = {"subject": ("tab:green", 1.6, "-"), "co-walker": ("tab:red", 1.3, "--")}; done = set()
    for v in ("left", "right"):
        for tid, r in scored[v].items():
            role = roles[(v, tid)]
            if role not in styles or not r["match"].get("ok") or (v, role) in done:
                continue
            tt, yy = hip_vertical_signal(r["tr"]); ti = tt - L; okm = np.isfinite(yy) & (ti >= t0 - 1) & (ti <= t1 + 1)
            if okm.sum() < 50 or ti[okm].min() > t0 + 1 or ti[okm].max() < t1 - 1:
                continue
            grid, yv = resample_to_grid(ti[okm], yy[okm], fs, t0, t1); yv = bandpass(yv, fs); c, lw, ls = styles[role]; done.add((v, role))
            ax[0].plot(grid, z(yv), color=c, lw=lw, ls=ls, label=f"video, {v} view: {role} (fragment {tid}), placed at the shared lag")
    ax[0].set_xlim(t0, t1); ax[0].set_xlabel("IMU time (s)"); ax[0].set_ylabel("z-scored"); ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title("A. Inside a walk bout, the chosen person's video pelvis signal follows the accelerometer; the co-walker 0.8 s behind does not", loc="left", fontsize=10.5)
    k = int(0.5 * 100 / 100 * 50)   # running max over +-0.5 s on the 100 Hz lag axis
    for v in ("left", "right"):
        for tid, r in scored[v].items():
            m = r["match"]
            if not m.get("ok"):
                continue
            role = roles[(v, tid)]; c = {"subject": "tab:green", "co-walker": "tab:red"}.get(role, "0.55")
            ax[1].plot(m["lags_s"], running_max(m["r_fine_fn"], k), color=c, lw=1.6 if role == "subject" else 1.1, alpha=0.95 if role != "other person" else 0.8, label=f"{v}: {role} ({tid})")
    ax[1].axvline(L, color="k", lw=1, ls=":"); ax[1].text(L + 0.6, 0.86, f"shared lag {L:.2f} s (true 3.00 s)", fontsize=8.5)
    ax[1].axhline(DEFAULTS["accept_r"], color="0.3", lw=0.8, ls="--"); ax[1].text(-14.5, DEFAULTS["accept_r"] + 0.02, "accept_r", fontsize=8, color="0.3")
    ax[1].set_xlim(-15, 50); ax[1].set_ylim(0, 0.95); ax[1].set_xlabel("candidate offset: video time of IMU t = 0 (s)"); ax[1].set_ylabel("correlation (local max over +-0.5 s)")
    ax[1].set_title("B. Correlation versus lag for every scored person in both views: the offset is the lag with the largest joint support", loc="left", fontsize=10.5)
    ax[1].legend(fontsize=7.5, ncol=3, loc="upper right")
    for i, v in enumerate(("left", "right")):
        t = ts[v]; chosen = set(sel["views"][v]["track_ids"]); ti = t.pts_ms / 1000 - L
        pres = np.zeros(len(t.frame_idx), bool)
        for tid in chosen:
            pres[t.track(int(tid))["pos"]] = True
        ax[2].fill_between(ti, 1 - i - 0.3, 1 - i + 0.3, where=pres, color="tab:green", step="mid", alpha=0.85, lw=0)
        ax[2].text(-0.8, 1 - i, f"{v} view", ha="right", va="center", fontsize=9)
    for a, b in walk:
        ax[2].axvspan(a, b, color="0.85", zorder=0)
    ax[2].set_yticks([]); ax[2].set_ylim(-0.6, 1.6); ax[2].set_xlim(-0.8, imu["duration_s"] + 0.5); ax[2].set_xlabel("IMU time (s)")
    for a, b in sel["views"]["right"]["coverage"]["gaps_imu_time_s"]:
        ax[2].annotate("exit", xy=((a + b) / 2, 0), ha="center", va="center", fontsize=8, color="tab:red")
    ri = json.loads((dl.parent / "session_summary.json").read_text())["items"]["hc900_selfpace"]
    ax[2].set_title(f"C. Coverage per view on the IMU timeline (grey = annotated Walk bouts); exits are gaps, not another person; bouts covered {ri.get('n_walk_bouts_ok')}/{ri.get('n_walk_bouts')}", loc="left", fontsize=10.5)
    plt.tight_layout(); plt.savefig(OUT / "selection_and_sync.png", dpi=130); plt.close(fig)


def fig_gvhmr(gpu: Path, imu, body_models: str | None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Z = {v: np.load(gpu / v / "smpl_global.npz") for v in ("left", "right")}
    pose = next(iter(json.loads((gpu / "reconcile.json").read_text())["pairs"].values()))["pose"]
    walk = [(a, b) for lab, a, b in imu["events"] if lab == "Walk"]; t0, t1 = walk[1][0], walk[1][1]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.4), gridspec_kw={"height_ratios": [1.1, 1]})
    for v, c in (("left", "tab:blue"), ("right", "tab:orange")):
        z = Z[v]; t = z["imu_t_s"]; bp = z["body_pose"].reshape(-1, 21, 3); w = (t >= t0) & (t <= t1)
        ax[0].plot(t[w], np.degrees(np.linalg.norm(bp[w, 3], axis=1)), color=c, lw=1.5, label=f"{v} view: left knee")
        ax[0].plot(t[w], np.degrees(np.linalg.norm(bp[w, 4], axis=1)), color=c, lw=1.0, ls="--", alpha=0.85, label=f"{v} view: right knee")
    kl = pose["knee_residual_lag"]
    ax[0].set_xlabel("IMU time (s)"); ax[0].set_ylabel("knee flexion (deg)"); ax[0].legend(fontsize=8, ncol=2)
    ax[0].set_title(f"A. Knee flexion lifted from each view independently, each placed on the IMU timeline by its own sync (one walk bout; residual lag {kl['residual_lag_s'] * 1000:+.0f} ms)", loc="left", fontsize=10)
    pj = pose["per_joint_geodesic"]; names = list(pj); med = np.array([pj[n]["median_deg"] for n in names]); p90 = np.array([pj[n]["p90_deg"] for n in names])
    x = np.arange(len(names)); legs = ("L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle")
    ax[1].bar(x, med, color=["tab:green" if n in legs else "0.6" for n in names], label="median")
    ax[1].errorbar(x, med, yerr=[np.zeros(len(med)), p90 - med], fmt="none", ecolor="0.3", lw=0.8, capsize=2, label="to the 90th percentile")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=60, ha="right", fontsize=8); ax[1].set_ylabel("geodesic disagreement (deg)"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"B. Per-joint disagreement between the two views' SMPL-X poses over {pose['overlap_s']:.0f} s: legs median {pose['legs_median_deg']:.1f} deg (green), all joints {pose['all_median_deg']:.1f} deg", loc="left", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT / "gvhmr_two_views_agreement.png", dpi=130); plt.close(fig)
    if not body_models:
        return
    import os, torch, smplx, trimesh, pyrender
    os.environ.pop("PYOPENGL_PLATFORM", None)
    bm = smplx.create(body_models, model_type="smplx", gender="neutral", use_pca=False, batch_size=1); faces = bm.faces.astype(np.int32)

    def body_at(v, t_imu):
        z = Z[v]; i = int(np.argmin(np.abs(z["imu_t_s"] - t_imu)))
        with torch.no_grad():
            o = bm(global_orient=torch.tensor(z["global_orient"][i:i + 1]), body_pose=torch.tensor(z["body_pose"][i:i + 1]), betas=torch.zeros(1, 10), transl=torch.zeros(1, 3))
        return o.vertices[0].numpy(), float(z["imu_t_s"][i])

    def render(V, color, az_deg):
        sc = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.45, 0.45, 0.45))
        tm = trimesh.Trimesh(V, faces, process=False); tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=np.tile(color + [255], (len(V), 1)))
        sc.add(pyrender.Mesh.from_trimesh(tm, smooth=True))
        c = V.mean(0); az = np.deg2rad(az_deg); eye = c + 3.2 * np.array([np.sin(az), 0.15, np.cos(az)])
        f = eye - c; f /= np.linalg.norm(f); r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r); u = np.cross(f, r)
        M = np.eye(4); M[:3, 0] = r; M[:3, 1] = u; M[:3, 2] = f; M[:3, 3] = eye
        sc.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(40)), pose=M)
        for p in ((2, 4, 3), (-2, 3, -2)):
            Lm = np.eye(4); Lm[:3, 3] = p; sc.add(pyrender.DirectionalLight(intensity=2.5), pose=Lm)
        rd = pyrender.OffscreenRenderer(420, 560); col, _ = rd.render(sc, flags=pyrender.RenderFlags.RGBA); rd.delete()
        col = col.astype(np.float32); a = col[..., 3:4] / 255.0
        return (col[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    times = t0 + 1.0 + np.arange(5) * 0.3
    fig, ax = plt.subplots(2, len(times), figsize=(2.4 * len(times), 6.9))
    for j, t in enumerate(times):
        for i, (v, col) in enumerate((("left", [140, 150, 180]), ("right", [180, 150, 140]))):
            V, ti = body_at(v, t); ax[i, j].imshow(render(V, col, 35)); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(f"IMU t = {ti:.2f} s", fontsize=10)
            if j == 0:
                ax[i, j].set_ylabel(f"lifted from the {v} view", fontsize=10)
    plt.suptitle("The SMPL-X body GVHMR recovers from each camera view, at the same IMU times during a walk bout (world frame; body shape zeroed)", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT / "gvhmr_two_views_mesh.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dry_run_out"); ap.add_argument("--gpu-deliver"); ap.add_argument("--body-models"); ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args(); d = Path(a.dry_run_out); OUT.mkdir(parents=True, exist_ok=True)
    ts, imu, sel, pan, exp, dl = load_run(d); roles = fragment_roles(ts, sel, exp)
    cap = fig_two_views(d, ts, sel, pan, roles); print("two_views_tracking.png")
    if not a.no_gif:
        fig_gif(d, cap, ts, sel, pan, roles); print("tracking.gif")
    fig_selection(ts, imu, sel, roles, dl); print("selection_and_sync.png")
    gpu = Path(a.gpu_deliver) if a.gpu_deliver else dl
    if (gpu / "left" / "smpl_global.npz").exists():
        fig_gvhmr(gpu, imu, a.body_models); print("gvhmr_two_views_*.png")
    else:
        print("no GPU-stage output: GVHMR figures skipped")


if __name__ == "__main__":
    main()
