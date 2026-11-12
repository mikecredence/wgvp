"""Video probing, decoding with per-frame PTS, and frame-accurate cutting (CPU)."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mts", ".MP4", ".MOV")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe_video(path: str | Path, with_hash: bool = True) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path} (codec not supported by this OpenCV build? convert with ffmpeg)")
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
    info = {"path": str(path), "name": path.name, "size_bytes": int(path.stat().st_size),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_nominal": float(cap.get(cv2.CAP_PROP_FPS)), "n_frames_nominal": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec_fourcc": codec}
    cap.release()
    info["duration_nominal_s"] = info["n_frames_nominal"] / info["fps_nominal"] if info["fps_nominal"] > 0 else None
    if with_hash:
        info["sha256"] = sha256_file(path)
    return info


def iter_frames(path: str | Path, start: int = 0, end: int | None = None, step: int = 1):
    """Yield (frame_index, pts_ms, frame_bgr). frame_index counts decoded frames from 0 (container order).
    pts_ms is OpenCV's position timestamp of the decoded frame (container PTS); if the backend reports nothing,
    pts_ms falls back to index / fps_nominal."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    last_pts = -1.0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i >= start and (end is None or i < end) and (i - start) % step == 0:
                pts = cap.get(cv2.CAP_PROP_POS_MSEC)
                if not np.isfinite(pts) or pts < 0 or (pts == 0 and i > 0 and last_pts <= 0):
                    pts = 1000.0 * i / fps
                last_pts = pts
                ok2, frame = cap.retrieve()
                if not ok2:
                    break
                yield i, float(pts), frame
            elif end is not None and i >= end:
                break
            i += 1
    finally:
        cap.release()


def timing_qc(frame_idx: np.ndarray, pts_ms: np.ndarray, fps_nominal: float) -> dict:
    """Variable frame rate / dropped frame diagnostics from the PTS series of consecutive decoded frames."""
    pts = np.asarray(pts_ms, dtype=np.float64); idx = np.asarray(frame_idx)
    if len(pts) < 3:
        return {"n_frames": int(len(pts)), "fps_measured": None, "vfr": None, "n_gaps": 0, "gap_frames_est": 0}
    d = np.diff(pts) / np.maximum(np.diff(idx), 1)
    med = float(np.median(d))
    fps_meas = 1000.0 / med if med > 0 else None
    span_fps = (len(pts) - 1) / ((pts[-1] - pts[0]) / 1000.0) if pts[-1] > pts[0] else None
    gaps = np.where(d > 1.5 * med)[0]
    gap_frames = int(np.round((d[gaps] / med) - 1).sum()) if len(gaps) else 0
    nonmono = int((np.diff(pts) <= 0).sum())
    return {"n_frames": int(len(pts)), "fps_measured": fps_meas, "fps_span": span_fps,
            "interval_ms_median": med, "interval_ms_std": float(np.std(d)),
            "vfr": bool(np.std(d) > 1.0), "n_gaps": int(len(gaps)), "gap_frames_est": gap_frames,
            "gap_frame_indices": [int(idx[g]) for g in gaps[:50]], "n_nonmonotonic_pts": nonmono,
            "pts_start_ms": float(pts[0]), "pts_end_ms": float(pts[-1])}


def write_video(frames, out_path: str | Path, fps: float, width: int, height: int, crf: int = 18) -> str:
    """Write BGR frames (iterable of HxWx3 uint8) to H.264 MP4 via ffmpeg (stdin pipe); falls back to OpenCV mp4v."""
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    exe = ffmpeg_exe()
    if exe:
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        n = 0
        try:
            for fr in frames:
                if fr.shape[1] != width or fr.shape[0] != height:
                    fr = cv2.resize(fr, (width, height))
                p.stdin.write(np.ascontiguousarray(fr).tobytes()); n += 1
        finally:
            p.stdin.close(); err = p.stderr.read().decode(errors="ignore"); p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed writing {out_path}: {err[-500:]}")
        return "ffmpeg-libx264"
    w = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for fr in frames:
        if fr.shape[1] != width or fr.shape[0] != height:
            fr = cv2.resize(fr, (width, height))
        w.write(fr)
    w.release()
    return "opencv-mp4v"


def cut_video(src: str | Path, out_path: str | Path, frame_indices: np.ndarray, out_fps: float, scale: float = 1.0,
              crop: list[int] | None = None) -> dict:
    """Write the frames listed in frame_indices (sorted, container frame indices) to a new MP4 at out_fps, optionally
    cropped to the panel box crop=[x0, y0, x1, y1] and downscaled by scale."""
    frame_indices = np.asarray(sorted(set(int(i) for i in frame_indices)))
    want = set(frame_indices.tolist())
    info = probe_video(src, with_hash=False)
    if crop is not None:
        x0, y0, x1, y1 = [int(v) for v in crop]; w0, h0 = x1 - x0, y1 - y0
    else:
        x0, y0, w0, h0 = 0, 0, info["width"], info["height"]
    w, h = w0, h0
    if scale != 1.0:
        w, h = int(round(w0 * scale / 2) * 2), int(round(h0 * scale / 2) * 2)
    w, h = w - w % 2, h - h % 2

    def gen():
        for i, pts, fr in iter_frames(src, start=int(frame_indices[0]), end=int(frame_indices[-1]) + 1):
            if i in want:
                fr = fr[y0:y0 + h0, x0:x0 + w0]
                yield fr if (fr.shape[1] == w and fr.shape[0] == h) else cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
    codec = write_video(gen(), out_path, out_fps, w, h)
    n = probe_video(out_path, with_hash=False)["n_frames_nominal"]
    return {"out": str(out_path), "n_frames": int(n), "expected": int(len(frame_indices)), "width": w, "height": h,
            "fps": out_fps, "writer": codec, "scale": scale, "crop": crop}


# ----------------------------------------------------------------------------------------------------------------
# Combined two-view frames: panel detection (black borders + content seam), previews, cropping
# ----------------------------------------------------------------------------------------------------------------
def sample_frames(path: str | Path, n: int = 40, max_width: int = 640) -> tuple[np.ndarray, list[int]]:
    """n grey frames spread over the video, downscaled to max_width (keeps aspect). Returns (F, h, w) uint8 + indices."""
    cap = cv2.VideoCapture(str(path))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idx = sorted(set(int(i) for i in np.linspace(0, max(N - 2, 0), n)))
    s = min(1.0, max_width / max(W, 1)); w, h = int(round(W * s)), int(round(H * s))
    out, got = [], []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA) if s < 1 else g); got.append(i)
    cap.release()
    if not out:
        raise RuntimeError(f"no frames readable from {path}")
    return np.stack(out), got


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """[start, end) runs of True at least min_len long."""
    m = np.r_[False, mask, False]; d = np.diff(m.astype(int))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_len]


def _content_box(nonblack: np.ndarray, x0: int, x1: int, y0: int, y1: int, black_line: float = 0.9) -> list[int]:
    """Content box inside [x0,x1)x[y0,y1): trim border lines that are (almost) entirely black, from the outside in.
    Dark content inside a view (a black walkway mat, dark clothes) is not a border because it does not span a whole
    edge-adjacent line."""
    sub = nonblack[y0:y1, x0:x1]
    h, w = sub.shape
    top, bot, left, right = 0, h, 0, w
    while top < bot - 8 and (1 - sub[top, left:right].mean()) >= black_line:
        top += 1
    while bot > top + 8 and (1 - sub[bot - 1, left:right].mean()) >= black_line:
        bot -= 1
    while left < right - 8 and (1 - sub[top:bot, left].mean()) >= black_line:
        left += 1
    while right > left + 8 and (1 - sub[top:bot, right - 1].mean()) >= black_line:
        right -= 1
    return [x0 + left, y0 + top, x0 + right, y0 + bot]


def detect_panels(path: str | Path, n_samples: int = 40, black_level: int = 28, min_panel_frac: float = 0.2,
                  seam_min_score: float = 1.6) -> dict:
    """Find the two tiled views in a combined frame.

    1. black borders: pixels darker than black_level in (nearly) every sampled frame;
    2. content seam: the column (or row) where the temporal correlation between adjacent columns (rows) collapses -
       two independent views do not co-vary across the seam - combined with a jump in the non-black row (column)
       extent (letterboxing differs between panels);
    3. panel boxes = the tight non-black box on each side of the seam. Falls back to a single panel.
    4. veto: a seam that moving content crosses is a scene edge inside one view (a wall corner, a post), not a seam -
       nothing moves from one tiled view into the other. Black separator bands are exempt.
    Boxes are in full-resolution pixels [x0, y0, x1, y1] (x1/y1 exclusive).
    """
    F, idx = sample_frames(path, n_samples)
    info = probe_video(path, with_hash=False); W, H = info["width"], info["height"]
    f, h, w = F.shape; sx, sy = W / w, H / h
    nonblack = (F > black_level).mean(0) >= 0.5           # (h, w) content pixels
    X = F.astype(np.float32)

    def seam_profile(axis: int):
        """Per line boundary i|i+1 along `axis`: (a) persistent edge = median over frames of the mean |difference| across
        the boundary where both sides are content, relative to the typical boundary; (b) decorrelation of the two
        sides over (frames x positions). Letterbox edges fail (a)/(b) because they are evaluated only where both
        sides are content. Black separator bands between two content regions are explicit candidates."""
        A = X if axis == 1 else np.transpose(X, (0, 2, 1))          # (f, other, lines)
        M = nonblack if axis == 1 else nonblack.T                     # (other, lines)
        other = M.shape[0]
        a = A[:, :, :-1]; b = A[:, :, 1:]; m = (M[:, :-1] & M[:, 1:])[None].astype(np.float32)
        cnt = m[0].sum(0)                                             # content positions per boundary
        valid = cnt >= 0.3 * other
        edge = (np.abs(a - b) * m).sum(1) / np.maximum(m.sum(1), 1)   # (f, lines-1)
        edge_med = np.median(edge, 0)
        typical = np.median(edge_med[valid]) + 1e-6 if valid.any() else 1.0
        edge_ratio = np.where(valid, edge_med / typical, 0.0)
        # temporal co-fluctuation of the two sides (motion crossing a boundary inside one view co-varies; across the
        # seam the two views fluctuate independently). Per-pixel temporal means removed.
        am = a - a.mean(axis=0, keepdims=True); bm = b - b.mean(axis=0, keepdims=True)
        num = (am * bm * m).sum(axis=(0, 1)); den = np.sqrt((am ** 2 * m).sum(axis=(0, 1)) * (bm ** 2 * m).sum(axis=(0, 1))) + 1e-6
        corr = np.where(valid, num / den, np.nan)
        ext = M.mean(0)                                               # content fraction per line
        jump = np.abs(np.diff(ext))                                   # letterboxing differs between panels
        score = np.where(valid, np.log1p(edge_ratio) * (1 - 0.5 * np.clip(np.nan_to_num(corr, nan=0.0), 0, 1)) + 0.5 * jump, 0.0)
        # black separator bands strictly inside the content extent with content on both sides
        content_lines = np.where(ext >= 0.5)[0]; seps = set()
        if len(content_lines):
            lo_c, hi_c = content_lines[0], content_lines[-1]
            for s0, s1 in _runs(ext < 0.2, 1):
                if s0 > lo_c and s1 - 1 < hi_c and ext[max(lo_c, s0 - 4):s0].max() >= 0.5 and ext[s1:min(hi_c + 1, s1 + 4)].max() >= 0.5:
                    score[max(0, (s0 + s1) // 2 - 1)] = max(score[max(0, (s0 + s1) // 2 - 1)], 3.0); seps.add(max(0, (s0 + s1) // 2 - 1))
        return score, corr, ext, edge_ratio, seps

    def crossings(axis: int, i: int, level: float = 20.0, min_len: float = 0.04) -> tuple[int, int]:
        """Frames in which moving content (pixels off their temporal median) sits on both sides of boundary i|i+1 at
        the same positions, and frames in which it sits on either side. -> (both, either)."""
        A = X if axis == 1 else np.transpose(X, (0, 2, 1))
        fg = np.abs(A - np.median(A, 0)) > level
        a = fg[:, :, max(0, i - 2):i].any(2); b = fg[:, :, i + 3:i + 5].any(2); n = min_len * A.shape[1]
        return int(((a & b).sum(1) >= n).sum()), int(((a | b).sum(1) >= n).sum())

    best = None
    for axis, name in ((1, "horizontal"), (0, "vertical")):
        score, corr, ext, edge_ratio, seps = seam_profile(axis)
        L = len(score) + 1
        lo, hi = int(min_panel_frac * L), int((1 - min_panel_frac) * L)
        if hi <= lo + 2:
            continue
        s = score.copy(); s[:lo] = -1; s[hi:] = -1
        for i in range(lo, hi):   # content must exist on both sides (looking past a black separator band)
            if ext[max(0, i - 24):i + 1].max() < 0.3 or ext[i + 1:i + 25].max() < 0.3:
                s[i] = -1
        i = int(np.argmax(s))
        cand = {"layout": name, "seam_small": i + 1, "score": float(s[i]), "corr_at_seam": float(corr[i]) if np.isfinite(corr[i]) else None,
                "edge_ratio_at_seam": float(edge_ratio[i]), "median_corr": float(np.nanmedian(corr)) if np.isfinite(corr).any() else None,
                "separator": i in seps, "axis": axis}
        if best is None or cand["score"] > best["score"]:
            best = cand
    out = {"width": W, "height": H, "n_samples": len(idx), "panels": [], "layout": "single", "seam": None, "detail": best}
    full = _content_box(nonblack, 0, w, 0, h)
    if best is not None and best["score"] >= seam_min_score and not best["separator"]:
        both, either = crossings(best.pop("axis"), best["seam_small"] - 1)
        best["frames_crossing_seam"] = [both, either]
        if both >= 2 and both >= 0.2 * either:
            best["vetoed"] = "moving content crosses the candidate seam: a scene edge inside one view"
    if best is not None:
        best.pop("axis", None)
    if best is not None and best["score"] >= seam_min_score and not best.get("vetoed"):
        c = best["seam_small"]
        if best["layout"] == "horizontal":
            boxes = [("left", _content_box(nonblack, 0, c, 0, h)), ("right", _content_box(nonblack, c, w, 0, h))]
            out["seam"] = int(round(c * sx))
        else:
            boxes = [("top", _content_box(nonblack, 0, w, 0, c)), ("bottom", _content_box(nonblack, 0, w, c, h))]
            out["seam"] = int(round(c * sy))
        out["layout"] = best["layout"]
        for pid, (x0, y0, x1, y1) in boxes:
            out["panels"].append({"id": pid, "box": [int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy))]})
    else:
        x0, y0, x1, y1 = full
        out["panels"].append({"id": "full", "box": [int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy))]})
    for p in out["panels"]:
        b = p["box"]; p["width"] = b[2] - b[0]; p["height"] = b[3] - b[1]; p["frac_of_frame"] = round(p["width"] * p["height"] / (W * H), 3)
    return out


def panel_preview(path: str | Path, panels: list[dict], out_png: str | Path, mode: str = "edges", frame: int = 0, max_width: int = 1400) -> str:
    """PNG of one frame with the panel boxes drawn; mode 'edges' (two-level Canny map, no photographic pixels) or 'pixels'."""
    img = None
    for i, pts, fr in iter_frames(path, start=frame, end=frame + 1):
        img = fr
    if img is None:
        raise RuntimeError("frame not readable")
    s = min(1.0, max_width / img.shape[1])
    if mode == "edges":
        img = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), 60, 140)
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if mode == "edges":   # two levels again after the resize: a textured scene (grass, gravel) must stay a line drawing
        img = cv2.cvtColor(np.where(img >= 96, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cols = [(60, 220, 60), (60, 160, 255), (220, 60, 220)]
    for k, p in enumerate(panels):
        x0, y0, x1, y1 = (int(round(v * s)) for v in p["box"]); col = cols[k % 3]
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), col, 3)
        cv2.putText(img, f"{p['id']} {p['box'][2] - p['box'][0]}x{p['box'][3] - p['box'][1]}", (x0 + 8, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    if mode == "edges":   # exactly black, white and the box colours: OpenCV 5 draws text with blended edges whatever the line type
        pal = np.array([(0, 0, 0), (255, 255, 255)] + cols, np.int32)
        img = pal[np.argmin(((img[:, :, None, :].astype(np.int32) - pal) ** 2).sum(-1), axis=2)].astype(np.uint8)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(out_png), img)
    return str(out_png)
