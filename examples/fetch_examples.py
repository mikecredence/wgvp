"""Build the example corpus (examples/data/) from public YouTube footage and a fabricated IMU recording per clip.

    python examples/fetch_examples.py                      # yt-dlp downloads the three sources (pip install yt-dlp)
    python examples/fetch_examples.py --source-dir DIR     # use <video_id>.mp4 files you already have
    python examples/fetch_examples.py --cookies-from-browser chrome   # when YouTube asks yt-dlp to prove it is not a bot
    python -m wgvp run examples/data --cpu-only            # then run the pipeline on the corpus ...
    python examples/verify_examples.py                     # ... and check it found the right person at the right offset

Per example (examples/sources.json): the segment is cut and re-encoded (constant 30 fps, <= 1280 px wide, H.264) into
the WearGait folder layout `<EXnn>/SP/<EXnn>_SP.mp4`; the pipeline's own 2D tracker runs once on the cut; the
fragment(s) under `subject_hints` become the subject; the subject's pelvis vertical signal (hip row relative to the
fitted horizon, in torso units) gets band-limited noise added at a displacement SNR of 2, is differentiated twice
and written as `LowerBack_FreeAcc_U` of `<EXnn>/<EXnn>_selfpace.csv` at 100 Hz, with IMU t = 0 placed `offset_s`
seconds into the cut and GeneralEvent = Walk while the subject is visible. (Noise is added in the displacement
domain on purpose: white noise added to the acceleration would, after the pipeline's double integration, come back
as 1/f^2 noise that buries the gait band whenever keypoint jitter dominates the second derivative.) The ground truth (offset, subject hip track) is written to
`<EXnn>/expected.json` for verify_examples.py. The reference is derived from the same tracker the pipeline uses, so
these examples show that the machinery runs and chooses the intended person on real footage; they are not an
independent accuracy measurement (the bundled sample and the unit tests cover that).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
from wgvp.config import DEFAULT_CONFIG                                      # noqa: E402
from wgvp.signals import bandpass                                           # noqa: E402
from wgvp.tracking import analyze_video, hip_vertical_signal, split_fragments, track_summary   # noqa: E402
from wgvp.video_io import ffmpeg_exe, probe_video                            # noqa: E402

WEIGHTS = PKG / "weights" / "yolov8n-pose.pt"
FS = 100.0


def obtain(ex: dict, cache: Path, source_dir: Path | None, cookies: str | None) -> Path:
    vid = ex["video_id"]
    if source_dir:
        hits = sorted(p for p in source_dir.glob(f"{vid}.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".m4v"))
        if not hits:
            raise FileNotFoundError(f"{vid}: no {vid}.mp4 (or .mkv/.webm/.mov) in {source_dir}")
        return hits[0]
    cache.mkdir(parents=True, exist_ok=True)
    hits = sorted(p for p in cache.glob(f"{vid}.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".m4v") and not p.name.endswith(".part"))
    if hits:
        return hits[0]
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "-f", "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]/b",
           "-o", str(cache / f"{vid}.%(ext)s"), ex["url"]]
    if cookies:
        cmd += ["--cookies-from-browser", cookies]
    print(f"[{ex['name']}] downloading {ex['url']} ...")
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-6:])
        hint = ("\nYouTube asked for a sign-in / bot check: re-run with --cookies-from-browser <chrome|firefox|safari|edge>, or download the video with any tool "
                f"and pass --source-dir with the file named {vid}.mp4") if "bot" in tail.lower() or "sign in" in tail.lower() else ""
        raise RuntimeError(f"yt-dlp failed for {vid} (is it installed? pip install yt-dlp):\n{tail}{hint}")
    hits = sorted(p for p in cache.glob(f"{vid}.*") if not p.name.endswith(".part"))
    if not hits:
        raise RuntimeError(f"yt-dlp reported success but no file {vid}.* in {cache}")
    return hits[0]


def cut(src: Path, dst: Path, t0: float, t1: float, fps: int = 30, max_w: int = 1280):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(src),
           "-vf", f"fps={fps},scale='min({max_w},iw)':-2", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def track(video: Path, log):
    """Exactly the pipeline's tracker settings, so the fragments found here are the ones the run will see."""
    info = probe_video(video, with_hash=False); prm = DEFAULT_CONFIG["params"]
    ts = analyze_video(video, WEIGHTS, [{"id": "full", "box": [0, 0, info["width"], info["height"]]}], max_width=prm["track_max_width"], conf=prm["pose_conf"], log=log)["full"]
    return split_fragments(ts)


def fragments_at_hints(ts, hints: list) -> list[int]:
    """The fragment whose box contains the hint point at (or within 5 frames of) the hint frame; nearest box centre on ties."""
    out = []
    for frame, x, y in hints:
        best = None
        for df, tid, box in zip(ts.det_frame, ts.det_tid, ts.det_box):
            f = int(ts.frame_idx[df])
            if abs(f - int(frame)) > 5:
                continue
            inside = box[0] - 10 <= x <= box[2] + 10 and box[1] - 10 <= y <= box[3] + 10
            if not inside:
                continue
            d = abs(f - int(frame)) * 50 + np.hypot((box[0] + box[2]) / 2 - x, (box[1] + box[3]) / 2 - y)
            if best is None or d < best[0]:
                best = (d, int(tid))
        if best is None:
            raise RuntimeError(f"no tracked person contains the hint point {(frame, x, y)}; the tracker found other people here than when the manifest was made")
        if best[1] not in out:
            out.append(best[1])
    return out


def acceleration_from_displacement(disp: np.ndarray, inside: np.ndarray, noise_sd: float, rng) -> np.ndarray:
    """Pelvis 'accelerometer' from a displacement signal on the 100 Hz grid (NaN where the walker is not visible):
    band-limited noise is added to the displacement (noise_sd x the displacement's own standard deviation inside the
    visible span, so noise_sd = 0.5 is an SNR of 2 in the quantity the pipeline matches), the sum is differentiated
    twice, and the result is scaled to a plausible amplitude (m/s^2). Outside the visible span the accelerometer
    carries noise only. The band-pass runs with the NaNs still in place (bridged, then restored) so that the
    non-zero mean of the displacement does not become a step at the span edges, whose ringing at the band edge
    would otherwise dominate the reference."""
    d = bandpass(np.where(inside, disp, np.nan), FS)
    d = np.where(inside & np.isfinite(d), d, 0.0)
    sd = d[inside].std() + 1e-9
    noise = np.nan_to_num(bandpass(rng.standard_normal(len(d)), FS)); noise = noise / (noise.std() + 1e-9) * noise_sd * sd
    acc = np.gradient(np.gradient(np.where(inside, d, 0.0) + noise, 1 / FS), 1 / FS)
    return 1.5 * acc / (acc[inside].std() + 1e-9)


def fabricate(ts, tids: list[int], offset_s: float, duration_s: float, noise_sd: float, seed: int) -> dict:
    """IMU-like recording from the subject's own pelvis signal: acc = d2/dt2 of the band-passed displacement, unit
    variance where the subject is visible, plus white noise everywhere; Walk where visible, Standing elsewhere."""
    rng = np.random.default_rng(seed)
    grid = np.arange(int(round(duration_s * FS)) + 1) / FS
    disp = np.full(len(grid), np.nan); inside = np.zeros(len(grid), bool); gt_frames, gt_hip = [], []
    for tid in tids:
        tr = ts.track(int(tid)); t, y = hip_vertical_signal(tr); ok = np.isfinite(y)
        if ok.sum() < 20:
            continue
        ti = t[ok] - offset_s; w = (grid >= ti.min()) & (grid <= ti.max())
        disp[w] = np.interp(grid[w], ti, y[ok]); inside |= w
        for f, kp, v in zip(tr["frame_idx"][ok], tr["kp"][ok], tr["vis"][ok]):
            gt_frames.append(int(f)); gt_hip.append([float(kp[[11, 12], 0].mean()), float(kp[[11, 12], 1].mean())])
    if inside.sum() < 3 * FS:
        raise RuntimeError("subject visible for under 3 s inside the recording window")
    acc_ms2 = acceleration_from_displacement(disp, inside, noise_sd, rng)
    # events: Walk over each visible stretch (merged across gaps < 0.5 s), Standing elsewhere
    lab = np.where(inside, "Walk", "Standing").astype(object)
    edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))      # starts at even positions, ends at odd ones
    for g0, g1 in zip(edges[1::2][:-1], edges[2::2]):                        # gaps between consecutive visible stretches
        if (g1 - g0) / FS < 0.5:
            lab[g0:g1] = "Walk"
    return {"t": grid, "acc_up": acc_ms2, "acc_en": 0.3 * rng.standard_normal((len(grid), 2)), "label": lab,
            "walk_s": float(inside.sum() / FS), "gt_frames": gt_frames, "gt_hip_px": gt_hip}


def write_csv(path: Path, rec: dict):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Time", "GeneralEvent", "ClinicalEvent", "LowerBack_FreeAcc_E", "LowerBack_FreeAcc_N", "LowerBack_FreeAcc_U"])
        for t, l, en, u in zip(rec["t"], rec["label"], rec["acc_en"], rec["acc_up"]):
            w.writerow([f"{t:.2f} sec", l, "", f"{en[0]:.4f}", f"{en[1]:.4f}", f"{u:.4f}"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "data"), help="corpus folder (default examples/data)")
    ap.add_argument("--source-dir", help="folder with already-downloaded <video_id>.mp4 files (skips yt-dlp)")
    ap.add_argument("--cookies-from-browser", help="passed to yt-dlp when YouTube requires a signed-in session")
    ap.add_argument("--only", nargs="*", help="example names, e.g. EX01")
    ap.add_argument("--force", action="store_true", help="rebuild examples that already exist")
    a = ap.parse_args()
    manifest = json.loads((HERE / "sources.json").read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = HERE / "_cache"; src_dir = Path(a.source_dir) if a.source_dir else None
    cfg_path = out / "wgvp_config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps({"_doc": "example corpus: single-view clips (no second panel to detect)", "views": {"layout": "single"}}, indent=1))
    built = []
    for ex in manifest["examples"]:
        name = ex["name"]
        if a.only and name not in a.only:
            continue
        sess = out / name; video = sess / "SP" / f"{name}_SP.mp4"; csv_path = sess / f"{name}_selfpace.csv"
        if video.exists() and csv_path.exists() and (sess / "expected.json").exists() and not a.force:
            print(f"[{name}] exists, skipping (--force to rebuild)"); built.append(name); continue
        src = obtain(ex, cache, src_dir, a.cookies_from_browser)
        t0, t1 = ex["cut_s"]; print(f"[{name}] cutting {src.name} {t0:.1f}-{t1:.1f} s -> {video.relative_to(out)}")
        cut(src, video, t0, t1)
        print(f"[{name}] tracking the cut (CPU) ...")
        ts = track(video, log=lambda m: print("   ", m))
        tids = fragments_at_hints(ts, ex["subject_hints"])
        info = [track_summary(ts, t) for t in tids]
        print(f"[{name}] subject fragments {tids}: " + ", ".join(f"{s['duration_s']:.1f} s" for s in info))
        clip_s = float(ts.pts_ms[-1] / 1000.0)
        rec = fabricate(ts, tids, ex["offset_s"], clip_s - ex["offset_s"] - 0.3, ex["noise_sd"], ex["seed"])
        write_csv(csv_path, rec)
        expected = {"session_id": name, "recording_id": f"{name}_selfpace", "video": f"SP/{name}_SP.mp4", "fps": 30, "offset_s": ex["offset_s"],
                    "imu_duration_s": float(rec["t"][-1]), "walk_s": rec["walk_s"], "subject_fragments_at_build": tids,
                    "gt_frames": rec["gt_frames"], "gt_hip_px": rec["gt_hip_px"], "source": {k: ex[k] for k in ("video_id", "url", "cut_s", "about")}}
        (sess / "expected.json").write_text(json.dumps(expected))
        (sess / "README.txt").write_text(f"{name}: {ex['about']}\nSource {ex['url']} cut {t0}-{t1} s (fetched locally, not redistributed).\n"
                                         f"{name}_selfpace.csv is a FABRICATED recording: LowerBack_FreeAcc_U = the walker's own pelvis motion from the video, differentiated twice, plus noise; IMU t=0 at video {ex['offset_s']} s.\n")
        print(f"[{name}] done: walk {rec['walk_s']:.1f} s of {rec['t'][-1]:.1f} s, IMU t=0 at video {ex['offset_s']} s")
        built.append(name)
    print(f"\nbuilt {built} in {out}\nnext:  python -m wgvp run {out} --cpu-only   then   python examples/verify_examples.py --data {out}")


if __name__ == "__main__":
    main()
