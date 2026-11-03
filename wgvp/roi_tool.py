"""Operator tools with no extra installs (tkinter): draw the walkway polygon / light rectangle once per camera, or
pick the subject on a skeleton-only frame. Nothing here is required for an automatic run."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .qc import dump_json, skeleton_frame
from .tracking import TrackSet
from .video_io import iter_frames


def _frame_image(video: Path, frame: int, show: str, ts: TrackSet | None, subject=()) -> np.ndarray:
    img = None
    for i, pts, fr in iter_frames(video, start=frame, end=frame + 1):
        img = fr
    if img is None:
        raise RuntimeError(f"frame {frame} not readable")
    if show == "pixels":
        base = img
    elif show == "edges":
        e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), 60, 140)
        base = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
    else:
        base = np.zeros_like(img)
    if ts is not None:
        pos = int(np.argmin(np.abs(ts.frame_idx - frame)))
        sk = skeleton_frame(ts, pos, list(subject), ts.width, ts.height, scale=1.0, label=f"frame {frame}")
        base = cv2.addWeighted(base, 1.0, sk, 1.0, 0)
    return base


def tk_pick_points(image_bgr: np.ndarray, title: str, mode: str = "polygon", max_points: int | None = None) -> list[tuple[float, float]]:
    """Show an image in a tkinter window; left-click adds points, right-click removes the last, Enter/Done finishes.
    Returns points in ORIGINAL pixel coordinates."""
    import tkinter as tk
    h, w = image_bgr.shape[:2]
    scale = min(1.0, 1400 / w, 850 / h)
    disp = cv2.resize(image_bgr, (int(w * scale), int(h * scale))) if scale < 1 else image_bgr
    ok, png = cv2.imencode(".png", disp)
    root = tk.Tk(); root.title(title)
    photo = tk.PhotoImage(data=png.tobytes())
    canvas = tk.Canvas(root, width=disp.shape[1], height=disp.shape[0]); canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=photo)
    pts, items = [], []
    info = tk.Label(root, text="left-click: add point   right-click: undo   Enter or Done: finish"); info.pack()

    def redraw():
        for it in items:
            canvas.delete(it)
        items.clear()
        for i, (x, y) in enumerate(pts):
            items.append(canvas.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#00ff00", width=2))
            if i:
                items.append(canvas.create_line(pts[i - 1][0], pts[i - 1][1], x, y, fill="#00ff00", width=2))
        if mode == "polygon" and len(pts) > 2:
            items.append(canvas.create_line(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1], fill="#00ff00", dash=(4, 2)))
        if mode == "rect" and len(pts) == 2:
            items.append(canvas.create_rectangle(pts[0][0], pts[0][1], pts[1][0], pts[1][1], outline="#00aaff", width=2))

    def add(e):
        if max_points and len(pts) >= max_points:
            return
        pts.append((e.x, e.y)); redraw()
        if max_points and len(pts) >= max_points and mode in ("rect", "point"):
            root.after(300, root.destroy)

    def undo(e=None):
        if pts:
            pts.pop(); redraw()
    canvas.bind("<Button-1>", add); canvas.bind("<Button-3>", undo); canvas.bind("<Button-2>", undo)
    root.bind("<Return>", lambda e: root.destroy())
    tk.Button(root, text="Done", command=root.destroy).pack()
    root.mainloop()
    return [(x / scale, y / scale) for x, y in pts]


def set_roi(video: Path, config_dir: Path, camera_id: str, kind: str, frame: int = 0, show: str = "edges", points=None,
            tracks: TrackSet | None = None) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    out = config_dir / f"{kind}_{camera_id}.json"
    if points is None:
        img = _frame_image(video, frame, show, tracks)
        if kind == "walkway":
            points = tk_pick_points(img, f"{camera_id}: click the walkway corners (polygon), then Enter", "polygon")
        else:
            points = tk_pick_points(img, f"{camera_id}: click two opposite corners of the sync light", "rect", max_points=2)
    if kind == "walkway":
        if len(points) < 3:
            raise SystemExit("need at least 3 points for the walkway polygon")
        roi = {"type": "polygon", "points": [[float(x), float(y)] for x, y in points], "frame_used": frame, "camera_id": camera_id}
    else:
        (x0, y0), (x1, y1) = points[:2]
        roi = {"type": "rect", "rect": [float(min(x0, x1)), float(min(y0, y1)), float(abs(x1 - x0)), float(abs(y1 - y0))], "frame_used": frame, "camera_id": camera_id}
    dump_json(roi, out)
    return out


def pick_subject(tracks: TrackSet, config_dir: Path, camera_id: str, recording_id: str, frame: int | None = None, track_id: int | None = None,
                 png_out: Path | None = None, points: list | None = None) -> Path:
    """Skeleton-only pick. With --track-id or --pick frame:x,y no window is opened (headless servers); points are
    resolved against the tracking at run time, so they stay valid when the video is tracked again elsewhere.
    Otherwise a tkinter window shows every skeleton with its id; the click nearest a skeleton selects it, and the
    click itself is stored beside the id for the same reason."""
    config_dir.mkdir(parents=True, exist_ok=True)
    out = config_dir / f"pick_{camera_id}_{recording_id}.json"
    if points:
        from .selection import track_at_point
        resolved = [track_at_point(tracks, f, x, y) for f, x, y in points]
        dump_json({"points": [{"frame": int(f), "x": float(x), "y": float(y)} for f, x, y in points], "track_ids_at_pick": [t for t in resolved if t is not None],
                   "camera_id": camera_id, "recording_id": recording_id}, out)
        return out
    if frame is None:   # frame with the most tracks visible
        counts = np.bincount(tracks.det_frame, minlength=len(tracks.frame_idx))
        frame = int(tracks.frame_idx[int(np.argmax(counts))])
    pos = int(np.argmin(np.abs(tracks.frame_idx - frame)))
    img = skeleton_frame(tracks, pos, [], tracks.width, tracks.height, scale=1.0, label=f"frame {frame}: click the subject")
    if png_out:
        cv2.imwrite(str(png_out), img)
    if track_id is None:
        pts = tk_pick_points(img, f"{camera_id}/{recording_id}: click the subject's skeleton", "point", max_points=1)
        if not pts:
            raise SystemExit("no click")
        x, y = pts[0]
        m = tracks.det_frame == pos
        best, bd = None, 1e9
        for tid, kp in zip(tracks.det_tid[m], tracks.det_kp[m]):
            d = np.nanmin(np.hypot(kp[:, 0] - x, kp[:, 1] - y))
            if d < bd:
                best, bd = int(tid), d
        if best is None:
            raise SystemExit("no skeleton at that frame")
        track_id = best
    rec = {"track_ids": [int(track_id)], "frame": int(frame), "camera_id": camera_id, "recording_id": recording_id}
    if "x" in dir() and "y" in dir():
        rec["points"] = [{"frame": int(frame), "x": float(x), "y": float(y)}]
    dump_json(rec, out)
    return out
