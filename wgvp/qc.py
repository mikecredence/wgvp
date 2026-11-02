"""QC records, skeleton-only thumbnails (no pixels from the video), the validation bundle and the privacy check."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .tracking import TrackSet, SKELETON, FACE_KP

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".webm", ".pt"}   # .pt: GVHMR dumps are kept in work/, never shipped
ALLOWED_DELIVER_EXT = {".npz", ".json", ".txt", ".log", ".md", ".csv", ".png"}


class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def dump_json(obj, path: str | Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if not (isinstance(v, np.ndarray) and v.size > 64)}
        if isinstance(o, (list, tuple)):
            return [strip(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    with open(path, "w") as f:
        json.dump(strip(obj), f, indent=1, cls=JsonEncoder)


def skeleton_frame(ts: TrackSet, pos: int, subject_tids: list[int], width: int, height: int, walkway: dict | None = None,
                   light: dict | None = None, label: str = "", scale: float = 0.5, min_conf: float = 0.3) -> np.ndarray:
    """Render all tracked skeletons at decoded-frame position `pos` on a black canvas: subject green, others grey.
    Face keypoints are not drawn. Nothing from the video pixels is used."""
    w, h = int(width * scale), int(height * scale)
    img = np.zeros((h, w, 3), np.uint8)
    if walkway:
        pts = (np.array(walkway["points"], np.float32) * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (160, 90, 0), 1)
    if light:
        if light.get("rect"):
            x, y, rw, rh = [int(v * scale) for v in light["rect"]]
            cv2.rectangle(img, (x, y), (x + rw, y + rh), (0, 160, 230), 1)
        elif light.get("cell") is not None:
            from .tracking import GRID_NX, GRID_NY
            cx, cy = light["cell"]
            cv2.rectangle(img, (int(cx * w / GRID_NX), int(cy * h / GRID_NY)), (int((cx + 1) * w / GRID_NX), int((cy + 1) * h / GRID_NY)), (0, 160, 230), 1)
    m = ts.det_frame == pos
    for tid, box, kp in zip(ts.det_tid[m], ts.det_box[m], ts.det_kp[m]):
        col = (60, 220, 60) if int(tid) in subject_tids else (150, 150, 150)
        x1, y1, x2, y2 = (box * scale).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
        for a, b in SKELETON:
            if a in FACE_KP or b in FACE_KP:
                continue
            if kp[a, 2] >= min_conf and kp[b, 2] >= min_conf:
                cv2.line(img, tuple((kp[a, :2] * scale).astype(int)), tuple((kp[b, :2] * scale).astype(int)), col, 2)
        for j in range(5, 17):
            if kp[j, 2] >= min_conf:
                cv2.circle(img, tuple((kp[j, :2] * scale).astype(int)), 3, col, -1)
        cv2.putText(img, f"id{int(tid)}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(img, label, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img


def write_thumbnails(ts: TrackSet, subject_tids: list[int], out_dir: Path, n: int = 6, walkway=None, light=None,
                     tag: str = "", offset_frame: int | None = None, imu_duration_s: float | None = None) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    F = len(ts.frame_idx)
    if F == 0:
        return []
    positions = sorted(set(int(p) for p in np.linspace(0, F - 1, n)))
    if offset_frame is not None:
        p0 = int(np.argmin(np.abs(ts.frame_idx - offset_frame)))
        positions = sorted(set(positions + [p0]))
    paths = []
    for p in positions:
        t_s = ts.pts_ms[p] / 1000.0
        rel = "" if offset_frame is None else f"  imu_t={(t_s - ts.pts_ms[np.argmin(np.abs(ts.frame_idx - offset_frame))] / 1000.0):+.2f}s"
        img = skeleton_frame(ts, p, subject_tids, ts.width, ts.height, walkway, light,
                             label=f"{tag} frame {int(ts.frame_idx[p])}  t={t_s:.2f}s{rel}")
        f = out_dir / f"{tag}_f{int(ts.frame_idx[p]):07d}.png"
        cv2.imwrite(str(f), img); paths.append(str(f))
    return paths


def is_skeleton_only(png_path: Path, max_nonblack_frac: float = 0.08, max_colours: int = 16, max_nonblack_palette: float = 0.5) -> bool:
    """A delivered PNG must be a render, not a photograph: either sparse (at most max_nonblack_frac non-black pixels,
    a few skeletons on black) or palette-limited (at most max_colours distinct colours after coarse quantisation and
    at most max_nonblack_palette non-black, which covers a crowd of skeletons or an edge map but no photographic
    content, whose quantised colour count runs into the hundreds)."""
    img = cv2.imread(str(png_path))
    if img is None:
        return False
    nonblack = (img.max(axis=2) > 20).mean()
    if nonblack <= max_nonblack_frac:
        return True
    q = (img // 32).reshape(-1, 3)
    n_col = len(np.unique(q[q.max(1) > 0], axis=0))
    return bool(nonblack <= max_nonblack_palette and n_col <= max_colours)


def privacy_check(deliver_dir: Path) -> dict:
    """Assert the deliverable holds no video frames: only allowed extensions, PNGs skeleton-only, NPZ keys declared."""
    problems = []; files = []
    for f in sorted(deliver_dir.rglob("*")):
        if f.is_dir():
            continue
        files.append(str(f.relative_to(deliver_dir)))
        ext = f.suffix.lower()
        if ext in VIDEO_EXT or (ext in IMAGE_EXT and ext != ".png"):
            problems.append(f"{f.relative_to(deliver_dir)}: video/image file in deliverable")
        elif ext == ".png":
            if "validation_bundle" not in f.parts or not is_skeleton_only(f):
                problems.append(f"{f.relative_to(deliver_dir)}: PNG is not a sparse skeleton render")
        elif ext not in ALLOWED_DELIVER_EXT:
            problems.append(f"{f.relative_to(deliver_dir)}: unexpected file type {ext}")
        elif ext == ".npz":
            try:
                z = np.load(f, allow_pickle=False)
                for k in z.files:
                    a = z[k]
                    if a.dtype == np.uint8 and a.ndim >= 2 and a.size > 64 * 64:
                        problems.append(f"{f.relative_to(deliver_dir)}:{k}: uint8 array that could be an image")
            except Exception as e:
                problems.append(f"{f.relative_to(deliver_dir)}: unreadable npz ({e})")
    return {"ok": not problems, "n_files": len(files), "problems": problems, "files": files}


def tee_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a")

    def log(msg: str):
        import datetime
        line = f"{datetime.datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); fh.write(line + "\n"); fh.flush()
    log.close = fh.close
    return log
