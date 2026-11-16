"""A short lift: the first seconds of one lifter chunk of the dry-run sample through GVHMR, checked against the truth.

The whole sample takes a GPU-less machine hours (two 49-s views). This cuts the first `--frames` frames of one
view's chunk into a session of its own, lifts it with the stage's own `process_chunk` (so the boxes, the CPU wrapper
and the result conversion are the pipeline's), and compares the lifted leg joints with the rendered truth. Three
steps, since the lift runs in the GVHMR environment's interpreter and the rest in the pipeline's:

    python tools/lift_short_chunk.py prepare <dry-run output> <short root> [--frames 60] [--view left]
    <GVHMR python> tools/lift_short_chunk.py lift <short root> --gvhmr-root <GVHMR> [--device cpu] [--timeout 7200]
    python tools/lift_short_chunk.py check <short root> --gt sample/HC900_gt.npz [--max-legs-deg 20]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SESSION, REC = "HC900", "hc900_selfpace"


def chunk_dir(root: Path, view: str) -> Path:
    return Path(root) / "wgvp_output" / SESSION / "work" / view / REC / "chunks"


def prepare(src_root: Path, out_root: Path, frames: int, view: str) -> Path:
    """The first `frames` frames of the view's first chunk, as a chunk of a session under `out_root`."""
    import imageio_ffmpeg
    src = chunk_dir(src_root, view); meta = json.loads((src / "chunk_0.json").read_text(encoding="utf-8"))
    n = min(frames, int(meta["n_frames"]))
    dst = chunk_dir(out_root, view)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src / meta["video"]), "-frames:v", str(n), "-c:v", "libx264", "-preset", "fast",
                    "-crf", "18", "-pix_fmt", "yuv420p", str(dst / meta["video"])], check=True)
    np.save(dst / meta["bbox"], np.load(src / meta["bbox"])[:n])
    short = {k: (v[:n] if isinstance(v, list) and len(v) == int(meta["n_frames"]) else v)          # every per-frame list, cut alike
             for k, v in meta.items() if k not in ("status", "gvhmr_run", "error", "pose_npz")}
    short["n_frames"] = n; short["status"] = "pending"; short["short_of"] = str(src)
    (dst / "chunk_0.json").write_text(json.dumps(short, indent=1), encoding="utf-8")
    print(f"prepared {dst / 'chunk_0.json'}: {n} of {meta['n_frames']} frames of {view}")
    return dst


def lift(out_root: Path, gvhmr_root: Path, device: str, timeout: int) -> int:
    """Every chunk under the short session, through the stage's process_chunk (run this in the GVHMR interpreter)."""
    from wgvp.gvhmr_stage import process_chunk
    rc = 0
    for j in sorted((Path(out_root) / "wgvp_output" / SESSION / "work").glob("*/*/chunks/chunk_*.json")):
        t0 = time.time(); m = process_chunk(Path(gvhmr_root), j, timeout, device=device)
        print(f"{j.parts[-4]}/{j.stem}: {m['status']} in {time.time() - t0:.0f} s ({m['n_frames']} frames) {m.get('error', '')}")
        rc |= m["status"] != "done"
    return rc


def check(out_root: Path, gt: Path, max_legs_deg: float, view: str) -> int:
    """The lifted leg joints against the rendered truth, frame by frame."""
    from wgvp.fusion import LEGS, geodesic
    d = chunk_dir(out_root, view); meta = json.loads((d / "chunk_0.json").read_text(encoding="utf-8"))
    if meta.get("status") != "done":
        print(f"FAIL: the chunk is {meta.get('status')}: {meta.get('error')}"); return 1
    z = np.load(d / meta["pose_npz"]); G = np.load(gt)
    n = int(meta["n_frames"]); gi = {int(f): i for i, f in enumerate(G["frame_idx"])}
    sel = [gi[int(f)] for f in meta["frame_idx"] if int(f) in gi]
    ok = True
    for key in ("incam_body_pose", "global_body_pose", "incam_global_orient", "global_transl", "incam_betas"):
        a = z[key]; fin = bool(np.isfinite(a).all()); ok &= fin and len(a) == n
        print(f"  {key:22s} {a.shape} finite={fin}")
    body = z["incam_body_pose"].reshape(-1, 21, 3)[: len(sel)]; gt_body = G["pose_aa"][sel, 3:66].reshape(-1, 21, 3)
    legs = geodesic(body[:, LEGS], gt_body[:, LEGS]); med = float(np.nanmedian(legs)); p90 = float(np.nanpercentile(legs, 90))
    el = (meta.get("gvhmr_run") or {}).get("elapsed_s")
    print(f"  legs vs truth: median {med:.1f} deg, 90th percentile {p90:.1f} deg over {len(sel)} frames; lift took {el:.0f} s = {el / n:.1f} s per frame" if el else
          f"  legs vs truth: median {med:.1f} deg, 90th percentile {p90:.1f} deg over {len(sel)} frames")
    ok &= med < max_legs_deg and len(sel) >= 0.9 * n
    print("PASS" if ok else f"FAIL: legs median {med:.1f} deg (limit {max_legs_deg}), {len(sel)} of {n} frames matched to the truth")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("src_root"); p.add_argument("out_root"); p.add_argument("--frames", type=int, default=60); p.add_argument("--view", default="left")
    p = sub.add_parser("lift"); p.add_argument("out_root"); p.add_argument("--gvhmr-root", required=True); p.add_argument("--device", choices=["cuda", "cpu"], default="cpu"); p.add_argument("--timeout", type=int, default=7200)
    p = sub.add_parser("check"); p.add_argument("out_root"); p.add_argument("--gt", required=True); p.add_argument("--max-legs-deg", type=float, default=20.0); p.add_argument("--view", default="left")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(Path(a.src_root), Path(a.out_root), a.frames, a.view); return 0
    if a.cmd == "lift":
        return lift(Path(a.out_root), Path(a.gvhmr_root), a.device, a.timeout)
    return check(Path(a.out_root), Path(a.gt), a.max_legs_deg, a.view)


if __name__ == "__main__":
    sys.exit(main())
