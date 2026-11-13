"""(Maintainers.) Build examples/bundled/ from our own recording: the offline example that ships in the repository.

    python examples/make_bundled.py /path/to/PXL_20260625_145313035.mp4

OWN01  the clip as recorded (phone, portrait, 13 s): one person walks away down an office corridor, turns at the far
       end and walks back. Re-encoded to 720 px wide, 30 fps. A fabricated recording CSV is built from the walker's
       own pelvis motion exactly as fetch_examples.py does (noise at a displacement SNR of 2, IMU t = 0 planted
       0.8 s into the clip).
OWN02  a synthetic TWO-VIEW file made from OWN01, shaped like a tiled two-camera session video: the left panel is
       the clip zoomed in a little (the middle 84 % of its width and 86 % of its height, scaled up), the right
       panel is the clip MIRRORED and zoomed in hard (the middle 50 % by 55 %, scaled to the same panel height, so
       the walker is 1.6x larger and head and feet leave the frame when close), letterboxed differently and separated by
       a black band. A mirrored second panel is the test the merge needs: its left leg is the walker's right leg,
       and the merge must notice (`view_mirrored` in merged_smpl.npz) and reflect that view back before fusing;
       the two zooms test the per-view scale and the usable-joint logic. It is still one camera, not two: the two
       panels carry the same information. The same recording CSV as OWN01 applies (same frames, same timing).

The data root's wgvp_config.json sets detector_imgsz = 960: at the default 640 the walker at the far end of the
corridor (under a tenth of the frame height) drops out of detection for 2-3 s and the track splits into two 5-s
fragments, which is below what the selector accepts. It also gives the lifter each view's field of view
(gvhmr.focal_35mm), read from the corridor's own lines with tools/estimate_focal.py: 17.3 mm for the clip (wider than a
phone's main camera), 20.2 and 32.2 for the two zoomed panels of OWN02 (17.3 times each panel's zoom predicts 20.2
and 32.1). The person in the clip consented to its publication.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from examples.fetch_examples import fabricate, fragments_at_hints, track, write_csv   # noqa: E402
from wgvp.tracking import track_summary                                            # noqa: E402
from wgvp.video_io import ffmpeg_exe, probe_video                                  # noqa: E402

IMGSZ = 960
FOCAL_35MM = {"full": 17.3, "left": 20.2, "right": 32.2}     # tools/estimate_focal.py on OWN01's clip and on each panel of OWN02's
OFFSET_S = 0.8
ABOUT1 = ("OWN01: one person (the author) walks away down an office corridor, turns at the far end and walks back; phone video, portrait, 13 s. "
          "OWN01_selfpace.csv is a FABRICATED recording: LowerBack_FreeAcc_U = the walker's own pelvis motion from the video plus noise, differentiated twice; "
          f"IMU t=0 at video {OFFSET_S} s.")
ABOUT2 = ("OWN02: a synthetic two-view file made from OWN01 (left panel = the clip zoomed in a little, the middle 84 % x 86 %; right panel = the clip MIRRORED and "
          "zoomed in hard, the middle 50 % x 55 %, scaled to the same height, so head and feet are cut when the walker is close; letterboxed differently; black separator), "
          "to exercise the two-view code path, the mirror detection of the merge, two zoom levels and truncation offline. Not a second camera. Same fabricated recording as OWN01.")
ZOOM_L = (0.84, 0.86)      # left panel: fraction of the clip's width and height kept (centred)
ZOOM_R = (0.50, 0.55)      # right panel: the mirrored panel, zoomed in hard enough that the walker is cut when close


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="the original recording (PXL_20260625_145313035.mp4)")
    ap.add_argument("--out", default=str(HERE / "bundled")); ap.add_argument("--width", type=int, default=720)
    a = ap.parse_args(); out = Path(a.out); ff = ffmpeg_exe()
    for d in ("OWN01", "OWN02", "wgvp_output"):
        shutil.rmtree(out / d, ignore_errors=True)
    # ---- OWN01: the clip
    s1 = out / "OWN01"; (s1 / "SP").mkdir(parents=True); v1 = s1 / "SP" / "OWN01_SP.mp4"
    run([ff, "-y", "-loglevel", "error", "-i", a.source, "-vf", f"fps=30,scale={a.width}:-2", "-c:v", "libx264", "-preset", "slow", "-crf", "23",
         "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(v1)])
    print(f"OWN01 video {v1.stat().st_size / 1e6:.1f} MB; tracking ...")
    ts = track(v1, log=lambda *x: None, imgsz=IMGSZ)
    longest = max(ts.track_ids(), key=lambda t: track_summary(ts, int(t))["duration_s"]); tr = ts.track(int(longest)); k = len(tr["pos"]) // 2
    hint = [int(tr["frame_idx"][k]), int(tr["kp"][k][[11, 12], 0].mean()), int(tr["kp"][k][[11, 12], 1].mean())]
    tids = fragments_at_hints(ts, [hint]); clip_s = float(ts.pts_ms[-1] / 1000.0)
    print(f"OWN01 subject fragment {tids}: {track_summary(ts, tids[0])['duration_s']:.1f} s of {clip_s:.1f} s; hint {hint}")
    rec = fabricate(ts, tids, OFFSET_S, clip_s - OFFSET_S - 0.3, 0.5, 7); write_csv(s1 / "OWN01_selfpace.csv", rec)
    exp1 = {"session_id": "OWN01", "recording_id": "OWN01_selfpace", "video": "SP/OWN01_SP.mp4", "fps": 30, "offset_s": OFFSET_S, "gt_view": "full",
            "imu_duration_s": float(rec["t"][-1]), "walk_s": rec["walk_s"], "subject_hint": hint, "subject_fragments_at_build": tids,
            "gt_frames": rec["gt_frames"], "gt_hip_px": rec["gt_hip_px"], "detector_imgsz": IMGSZ, "about": ABOUT1}
    (s1 / "expected.json").write_text(json.dumps(exp1), encoding="utf-8"); (s1 / "README.txt").write_text(ABOUT1 + "\n", encoding="utf-8")
    # ---- OWN02: the mirrored, differently zoomed composite. Both panels end up PH pixels tall: the left crop scaled
    # by PH / (0.86 H), the mirrored right crop by PH / (0.68 H); the right panel is letterboxed top and bottom.
    info = probe_video(v1, with_hash=False); W, H = info["width"], info["height"]; PH = H; sep = 6
    def crop_box(fx, fy):
        cw = int(round(W * fx)) // 2 * 2; ch = int(round(H * fy)) // 2 * 2; return cw, ch, (W - cw) // 2, (H - ch) // 2
    cwl, chl, xl, yl = crop_box(*ZOOM_L); cwr, chr_, xr, yr = crop_box(*ZOOM_R)
    sl = PH / chl; pwl = int(round(cwl * sl)) // 2 * 2                                  # left panel: full height
    pad = int(round(H * 0.05)) // 2 * 2; phr = PH - 2 * pad; sr = phr / chr_; pwr = int(round(cwr * sr)) // 2 * 2   # right panel: letterboxed
    s2 = out / "OWN02"; (s2 / "SP").mkdir(parents=True); v2 = s2 / "SP" / "OWN02_SP.mp4"
    fc = (f"[0:v]crop={cwl}:{chl}:{xl}:{yl},scale={pwl}:{PH}[l];"
          f"[0:v]crop={cwr}:{chr_}:{xr}:{yr},hflip,scale={pwr}:{phr},pad={pwr}:{PH}:0:{pad}:black[r];"
          f"color=black:s={sep}x{PH}:r=30[s];[l][s][r]hstack=inputs=3:shortest=1[v]")      # shortest=1: the colour source is endless
    run([ff, "-y", "-loglevel", "error", "-i", str(v1), "-filter_complex", fc, "-map", "[v]", "-shortest", "-c:v", "libx264", "-preset", "slow", "-crf", "24",
         "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(v2)])
    shutil.copy(s1 / "OWN01_selfpace.csv", s2 / "OWN02_selfpace.csv")
    gt_hip = [[(x - xl) * sl, (y - yl) * sl] for x, y in rec["gt_hip_px"]]           # the truth hips in the left panel's pixels
    exp2 = {"session_id": "OWN02", "recording_id": "OWN02_selfpace", "video": "SP/OWN02_SP.mp4", "fps": 30, "offset_s": OFFSET_S, "gt_view": "left",
            "imu_duration_s": float(rec["t"][-1]), "walk_s": rec["walk_s"], "panels_true": {"left": [0, 0, pwl, PH], "right": [pwl + sep, pad, pwl + sep + pwr, pad + phr]},
            "zoom": {"left": {"crop_frac": list(ZOOM_L), "scale": sl}, "right": {"crop_frac": list(ZOOM_R), "scale": sr, "mirrored": True}},
            "gt_frames": rec["gt_frames"], "gt_hip_px": gt_hip, "detector_imgsz": IMGSZ, "about": ABOUT2}
    (s2 / "expected.json").write_text(json.dumps(exp2), encoding="utf-8"); (s2 / "README.txt").write_text(ABOUT2 + "\n", encoding="utf-8")
    print(f"OWN02 video {v2.stat().st_size / 1e6:.1f} MB ({pwl + sep + pwr}x{PH}; panels left {exp2['panels_true']['left']} x{sl:.2f}, right {exp2['panels_true']['right']} x{sr:.2f} mirrored)")
    (out / "wgvp_config.json").write_text(json.dumps({"_doc": "bundled offline examples: a portrait phone clip (OWN01) and a two-view composite of it (OWN02: zoomed | mirrored and zoomed more); "
                                                              "detector_imgsz 960 keeps the walker detected at the far end of the corridor; "
                                                              "gvhmr.focal_35mm is each view's field of view as measured from the corridor's lines by tools/estimate_focal.py",
                                                      "params": {"detector_imgsz": IMGSZ}, "gvhmr": {"focal_35mm": FOCAL_35MM}}, indent=1), encoding="utf-8")
    print(f"done -> {out}\nnext:  python -m wgvp run {out} --cpu-only   then   python examples/verify_examples.py --data {out}")


if __name__ == "__main__":
    main()
