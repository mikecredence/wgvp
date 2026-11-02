"""One decode pass per video: for each detected panel (view), multi-person 2D pose tracking (YOLO-pose + ByteTrack,
one tracker per view), a coarse appearance descriptor per detection (torso colour histogram, used only for
re-identification on the partner's machine), a visibility class per detection, and a coarse brightness grid (for
the optional light cross-check). CPU is enough; CUDA is used when available.

Per view the result is a TrackSet (coordinates in FULL-FRAME pixels; the view box is stored) saved in work/.
All people are kept there for selection; only the chosen subject's track is exported.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ultralytics must never try to pip-install or phone home on the partner's machine
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_VERBOSE", "false")

from .video_io import iter_frames, probe_video, timing_qc

COCO17 = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist",
          "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
FACE_KP = [0, 1, 2, 3, 4]
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 5), (0, 6)]
GRID_NX, GRID_NY = 32, 18
APP_BINS = (12, 4)      # H x S histogram bins of the torso region
VIS_FULL, VIS_PARTIAL, VIS_NONE = 2, 1, 0


@dataclass
class TrackSet:
    video: str
    view_id: str
    box: list                        # panel box [x0, y0, x1, y1] in full-frame pixels
    width: int                       # full frame size
    height: int
    fps_nominal: float
    frame_idx: np.ndarray            # (F,) decoded frame indices (container order)
    pts_ms: np.ndarray               # (F,)
    det_frame: np.ndarray            # (N,) position into frame_idx
    det_tid: np.ndarray              # (N,) tracker id (fragment id after splitting)
    det_box: np.ndarray              # (N, 4) xyxy full-frame pixels
    det_conf: np.ndarray             # (N,)
    det_kp: np.ndarray               # (N, 17, 3) x, y full-frame pixels, conf
    det_app: np.ndarray              # (N, 48) torso H-S histogram (L1-normalised), zeros if not computable
    det_vis: np.ndarray              # (N,) visibility class 2 full / 1 partial / 0 none
    grid: np.ndarray                 # (F, GRID_NY, GRID_NX) mean grey per cell inside the panel
    roi_series: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def track_ids(self) -> np.ndarray:
        return np.unique(self.det_tid)

    def track(self, tid: int) -> dict:
        m = self.det_tid == tid
        pos = self.det_frame[m]
        return {"tid": int(tid), "pos": pos, "frame_idx": self.frame_idx[pos], "pts_ms": self.pts_ms[pos], "box": self.det_box[m],
                "conf": self.det_conf[m], "kp": self.det_kp[m], "app": self.det_app[m], "vis": self.det_vis[m]}

    def save(self, path: str | Path):
        import json
        np.savez_compressed(path, video=self.video, view_id=self.view_id, box=np.array(self.box), width=self.width, height=self.height,
                            fps_nominal=self.fps_nominal, frame_idx=self.frame_idx, pts_ms=self.pts_ms, det_frame=self.det_frame, det_tid=self.det_tid,
                            det_box=self.det_box, det_conf=self.det_conf, det_kp=self.det_kp, det_app=self.det_app.astype(np.float16), det_vis=self.det_vis,
                            grid=self.grid, roi_names=np.array(list(self.roi_series.keys())),
                            roi_values=np.stack(list(self.roi_series.values())) if self.roi_series else np.zeros((0, len(self.frame_idx))),
                            meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path: str | Path) -> "TrackSet":
        import json
        z = np.load(path, allow_pickle=False)
        roi = {str(n): v for n, v in zip(z["roi_names"], z["roi_values"])}
        return cls(video=str(z["video"]), view_id=str(z["view_id"]), box=z["box"].tolist(), width=int(z["width"]), height=int(z["height"]),
                   fps_nominal=float(z["fps_nominal"]), frame_idx=z["frame_idx"], pts_ms=z["pts_ms"], det_frame=z["det_frame"], det_tid=z["det_tid"],
                   det_box=z["det_box"], det_conf=z["det_conf"], det_kp=z["det_kp"], det_app=z["det_app"].astype(np.float32), det_vis=z["det_vis"],
                   grid=z["grid"], roi_series=roi, meta=json.loads(str(z["meta"])))


def _load_pose_model(weights: str | Path):
    from ultralytics import YOLO
    return YOLO(str(weights))


def _roi_mask(roi: dict, w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    if roi.get("type") == "polygon":
        cv2.fillPoly(m, [np.array(roi["points"], np.int32).reshape(-1, 1, 2)], 255)
    else:
        x, y, rw, rh = [int(v) for v in roi["rect"]]
        m[max(0, y):y + rh, max(0, x):x + rw] = 255
    return m


def visibility_class(kp: np.ndarray, min_conf: float = 0.3) -> int:
    sh = (kp[[5, 6], 2] >= min_conf).any(); hip = (kp[[11, 12], 2] >= min_conf).any(); ank = (kp[[15, 16], 2] >= min_conf).any()
    kn = (kp[[13, 14], 2] >= min_conf).any()
    if sh and hip and ank:
        return VIS_FULL
    if hip or sh or ank or kn:
        return VIS_PARTIAL
    return VIS_NONE


def torso_histogram(img_bgr: np.ndarray, kp: np.ndarray, box: np.ndarray, min_conf: float = 0.3) -> np.ndarray:
    """H-S histogram of the torso region (shoulders..hips, else the middle of the box). L1-normalised, 48 floats."""
    h, w = img_bgr.shape[:2]
    pts = kp[[5, 6, 11, 12]]
    if (pts[:, 2] >= min_conf).sum() >= 3:
        p = pts[pts[:, 2] >= min_conf, :2]
        x0, y0 = p.min(0); x1, y1 = p.max(0)
    else:
        x0, y0, x1, y1 = box[0] + 0.25 * (box[2] - box[0]), box[1] + 0.25 * (box[3] - box[1]), box[2] - 0.25 * (box[2] - box[0]), box[3] - 0.45 * (box[3] - box[1])
    x0, x1 = int(np.clip(x0, 0, w - 1)), int(np.clip(x1, 0, w)); y0, y1 = int(np.clip(y0, 0, h - 1)), int(np.clip(y1, 0, h))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return np.zeros(APP_BINS[0] * APP_BINS[1], np.float32)
    hsv = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(APP_BINS), [0, 180, 0, 256]).ravel()
    s = hist.sum()
    return (hist / s).astype(np.float32) if s > 0 else np.zeros_like(hist, dtype=np.float32)


def analyze_video(video: str | Path, weights: str | Path, panels: list[dict], start: int = 0, end: int | None = None, step: int = 1,
                  max_width: int = 1280, conf: float = 0.3, device: str | None = None, imgsz: int = 640, rois: dict | None = None,
                  log=print, tracker: str = "bytetrack.yaml") -> dict:
    """Decode once; per panel: crop, downscale to max_width, YOLO-pose tracking with its own tracker, appearance,
    visibility, brightness grid. Returns {view_id: TrackSet}."""
    info = probe_video(video, with_hash=False)
    W, H = info["width"], info["height"]
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    views = []
    for p in panels:
        x0, y0, x1, y1 = [int(v) for v in p["box"]]
        pw, ph = x1 - x0, y1 - y0
        scale = min(1.0, max_width / max(pw, 1))
        views.append({"id": p["id"], "box": [x0, y0, x1, y1], "scale": scale, "tw": int(round(pw * scale)), "th": int(round(ph * scale)),
                      "model": _load_pose_model(weights), "frames": [], "pts": [], "grid": [], "det": [], "roi": {}, "roi_masks": {}})
        for name, r in (rois or {}).get(p["id"], {}).items():
            views[-1]["roi_masks"][name] = _roi_mask(r, W, H)[y0:y1, x0:x1]; views[-1]["roi"][name] = []
    t0 = time.time(); n = 0
    for i, pts, fr in iter_frames(video, start, end, step):
        for v in views:
            x0, y0, x1, y1 = v["box"]
            crop = fr[y0:y1, x0:x1]
            grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            v["grid"].append(cv2.resize(grey, (GRID_NX, GRID_NY), interpolation=cv2.INTER_AREA))
            for name, m in v["roi_masks"].items():
                v["roi"][name].append(float(cv2.mean(grey, mask=m)[0]))
            small = cv2.resize(crop, (v["tw"], v["th"]), interpolation=cv2.INTER_AREA) if v["scale"] < 1 else crop
            res = v["model"].track(small, persist=True, conf=conf, classes=[0], imgsz=imgsz, device=device, verbose=False, tracker=tracker)[0]
            pos = len(v["frames"]); v["frames"].append(i); v["pts"].append(pts)
            if res.boxes is not None and res.boxes.id is not None and len(res.boxes) > 0:
                ids = res.boxes.id.int().cpu().numpy(); boxes = res.boxes.xyxy.cpu().numpy(); confs = res.boxes.conf.cpu().numpy()
                kxy = res.keypoints.xy.cpu().numpy(); kc = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])
                for j in range(len(ids)):
                    kp_local = np.concatenate([kxy[j], kc[j][:, None]], axis=1)
                    app = torso_histogram(small, kp_local, boxes[j])
                    kp_full = kp_local.copy(); kp_full[:, :2] = kp_full[:, :2] / v["scale"] + [x0, y0]
                    box_full = boxes[j] / v["scale"] + [x0, y0, x0, y0]
                    v["det"].append((pos, int(ids[j]), box_full, float(confs[j]), kp_full, app, visibility_class(kp_local)))
        n += 1
        if n % 500 == 0:
            log(f"      tracked {n} frames ({n / (time.time() - t0):.1f} fps)")
    out = {}
    for v in views:
        det = v["det"]
        ts = TrackSet(video=str(video), view_id=v["id"], box=v["box"], width=W, height=H, fps_nominal=info["fps_nominal"],
                      frame_idx=np.array(v["frames"], np.int64), pts_ms=np.array(v["pts"], np.float64),
                      det_frame=np.array([d[0] for d in det], np.int64), det_tid=np.array([d[1] for d in det], np.int64),
                      det_box=np.array([d[2] for d in det], np.float32).reshape(-1, 4), det_conf=np.array([d[3] for d in det], np.float32),
                      det_kp=np.array([d[4] for d in det], np.float32).reshape(-1, 17, 3), det_app=np.array([d[5] for d in det], np.float32).reshape(-1, APP_BINS[0] * APP_BINS[1]),
                      det_vis=np.array([d[6] for d in det], np.int8), grid=np.array(v["grid"], np.uint8),
                      roi_series={k: np.array(x, np.float32) for k, x in v["roi"].items()},
                      meta={"weights": Path(weights).name, "imgsz": imgsz, "conf": conf, "device": device, "track_scale": v["scale"], "tracker": tracker,
                            "elapsed_s": time.time() - t0, "step": step, "start": start, "end": end, "timing": timing_qc(np.array(v["frames"]), np.array(v["pts"]), info["fps_nominal"])})
        out[v["id"]] = ts
        log(f"      view {v['id']}: {len(ts.track_ids())} raw track ids, {len(det)} detections")
    log(f"    tracking done: {n} frames x {len(views)} view(s), {time.time() - t0:.0f} s")
    return out


# ----------------------------------------------------------------------------------------------------------------
# Fragment splitting (identity switches) and per-track derived signals
# ----------------------------------------------------------------------------------------------------------------
def split_fragments(ts: TrackSet, max_jump_frac: float = 0.5, max_gap_s: float = 1.0, app_jump: float = 1.1) -> TrackSet:
    """Re-label tracker ids into continuous fragments: split where the box centre jumps by more than max_jump_frac
    box heights per frame between consecutive detections, where the time gap exceeds max_gap_s, or where the torso
    appearance changes abruptly and persistently (L1 distance between the median descriptor of the 8 detections
    before and the 8 after > app_jump; range 0..2). Fragment ids are new integers."""
    new_tid = np.zeros_like(ts.det_tid); nxt = 1
    for tid in ts.track_ids():
        m = np.where(ts.det_tid == tid)[0]
        m = m[np.argsort(ts.det_frame[m], kind="stable")]
        cur = nxt; nxt += 1
        app = ts.det_app[m]; has_app = app.sum(1) > 0
        for k, i in enumerate(m):
            if k > 0:
                j = m[k - 1]
                dt = (ts.pts_ms[ts.det_frame[i]] - ts.pts_ms[ts.det_frame[j]]) / 1000.0
                bh = max(ts.det_box[j, 3] - ts.det_box[j, 1], 1.0)
                c_i = ts.det_box[i, [0, 1]] + ts.det_box[i, [2, 3]]; c_j = ts.det_box[j, [0, 1]] + ts.det_box[j, [2, 3]]
                jump = np.linalg.norm(c_i - c_j) / 2 / bh
                frames_apart = max(dt * 30.0, 1.0)
                dapp = 0.0
                if 8 <= k <= len(m) - 8 and has_app[k - 8:k].sum() >= 4 and has_app[k:k + 8].sum() >= 4:
                    a0 = np.median(app[k - 8:k][has_app[k - 8:k]], 0); a1 = np.median(app[k:k + 8][has_app[k:k + 8]], 0)
                    dapp = float(np.abs(a0 - a1).sum())
                if dt > max_gap_s or jump > max_jump_frac * frames_apart or dapp > app_jump:
                    cur = nxt; nxt += 1
            new_tid[i] = cur
    ts.det_tid = new_tid
    ts.meta["n_fragments"] = int(nxt - 1)
    return ts


def hip_vertical_signal(tr: dict, min_kp_conf: float = 0.3, scale_smooth_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis vertical position (up positive, in torso-length units) on the track's PTS time; NaN without hips.

    Perspective-robust: for a camera at height hc, hip row = cy + f (hc - h) / Z and torso length l = f L / Z, so
    (hip_row - cy) / l = (hc - h) / L does not depend on the distance Z. The horizon row cy is fitted per track
    (hip_row ~ cy + a * l, least squares) whenever the apparent size varies along the track (walking toward or
    away from the camera, also in oblique views); when it does not vary, cy is immaterial. Dividing the raw row by
    l instead leaves a term cy / l that oscillates with every step toward or away from the camera and produces
    strong correlation peaks at wrong lags. The scale reference l is ONE quantity per track (shoulder-hip distance
    when the shoulders are confident in most frames, else box height / 3.2), smoothed over scale_smooth_s with
    edge padding (zero padding shrinks l at the fragment ends and inflates the signal there), so the per-step
    modulation of the torso length does not enter the signal.
    """
    kp = tr["kp"]
    hips = kp[:, [11, 12], :]; sh = kp[:, [5, 6], :]
    ok = (hips[:, :, 2] >= min_kp_conf).all(1)
    hip_y = hips[:, :, 1].mean(1)
    torso = np.linalg.norm(sh[:, :, :2].mean(1) - hips[:, :, :2].mean(1), axis=1)
    sh_ok = (sh[:, :, 2] >= min_kp_conf).all(1) & ok & (torso > 1)
    box_h = tr["box"][:, 3] - tr["box"][:, 1]
    n = len(hip_y)
    ref = np.where(sh_ok, torso, np.nan) if (n and sh_ok.mean() >= 0.7) else np.where(box_h > 1, box_h / 3.2, np.nan)
    t = tr["pts_ms"] / 1000.0
    okr = np.isfinite(ref)
    if n > 5 and okr.sum() > 2:
        rr = np.interp(np.arange(n), np.flatnonzero(okr), ref[okr])
        k = max(3, int(round(scale_smooth_s * n / max(t[-1] - t[0], 1e-3)))); k = k if k % 2 else k + 1
        k = min(k, n if n % 2 else n - 1)
        ref = np.convolve(np.pad(rr, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    cy = 0.0
    sel = ok & np.isfinite(ref)
    if sel.sum() > 30 and np.ptp(ref[sel]) > 0.10 * np.nanmedian(ref[sel]):
        A = np.c_[np.ones(sel.sum()), ref[sel]]
        coef, *_ = np.linalg.lstsq(A, hip_y[sel], rcond=None)
        cy = float(coef[0])
    y = -(hip_y - cy) / ref
    y[~ok] = np.nan
    return t, y


def locomotion_fraction(tr: dict, fps: float) -> float:
    """Share of the track's time with the box centre moving (> 0.15 box heights per second over 1 s windows)."""
    if len(tr["pos"]) < int(fps):
        return 0.0
    c = (tr["box"][:, [0, 1]] + tr["box"][:, [2, 3]]) / 2; bh = np.maximum(tr["box"][:, 3] - tr["box"][:, 1], 1)
    t = tr["pts_ms"] / 1000.0
    w = max(2, int(round(fps)))
    d = np.linalg.norm(c[w:] - c[:-w], axis=1) / bh[:-w]; dt = np.maximum(t[w:] - t[:-w], 1e-3)
    return float(np.mean(d / dt > 0.15))


def track_summary(ts: TrackSet, tid: int) -> dict:
    tr = ts.track(tid); box = tr["box"]
    fps = ts.meta.get("timing", {}).get("fps_measured") or ts.fps_nominal
    return {"tid": int(tid), "n_frames": int(len(tr["pos"])), "first_frame": int(tr["frame_idx"][0]), "last_frame": int(tr["frame_idx"][-1]),
            "duration_s": float((tr["pts_ms"][-1] - tr["pts_ms"][0]) / 1000.0), "mean_box_h_frac": float(np.mean(box[:, 3] - box[:, 1]) / ts.height),
            "mean_conf": float(np.mean(tr["conf"])), "hip_conf_frac": float(np.mean((tr["kp"][:, [11, 12], 2] >= 0.3).all(1))),
            "vis_full_frac": float(np.mean(tr["vis"] == VIS_FULL)), "locomotion_frac": locomotion_fraction(tr, fps)}
