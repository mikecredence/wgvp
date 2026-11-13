"""A camera's field of view from the room it films, for `gvhmr.focal_35mm`.

    python tools/estimate_focal.py <video> [--box x0 y0 x1 y1] [--times 2 5 8] [--out folder]

A room is built of three perpendicular families of straight lines (floor and ceiling edges along the room, the same
across it, door frames and wall corners up it). In a picture each family aims at one vanishing point, and the three
points fix the camera's focal length: two perpendicular directions with vanishing points v1 and v2 satisfy
(v1 - c) . (v2 - c) = -f^2, with c the picture centre. Nothing is measured in the room and no chart is held up.

Per frame: line segments (OpenCV's LSD), then the focal length and the room's orientation that put the most segment
length on one of the three vanishing points (RANSAC over pairs of points, then a least-squares refinement). The answer
is the median over the frames, in millimetres of 35 mm equivalent on the picture's diagonal: the number
`gvhmr.focal_35mm` takes for that view. `--box` crops one panel of a tiled video (pixels, as in panels.json).

What it needs: straight edges in two of the three directions at least, and a camera that looks at the room at an
angle. A camera square to a wall sees two of the three families parallel in the picture, and then the focal length
cannot be read (the tool says `weak`). A lens that bends straight lines (an action camera, a wide webcam) biases the
answer; a phone's video is already straightened. The picture centre is taken as the optical centre.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

DIAG_35 = float(np.hypot(36.0, 24.0))
MIN_SEGMENTS, MIN_SUPPORT, MIN_LEAN_DEG = 6, 0.05, 6.0     # a direction counts with this many segments and this share of the line length; an axis must lean this far out of the picture plane
F_RANGE_35 = (9.0, 110.0)                                   # focal lengths a room camera can have, mm of 35 mm equivalent


def line_segments(gray: np.ndarray, min_len_frac: float = 0.03, enhance: bool = False) -> np.ndarray:
    """n x 4 segments (x0, y0, x1, y1), longer than min_len_frac of the picture's diagonal. enhance: local contrast
    stretch first, for faint edges (a floor pattern in two near-equal greys)."""
    if enhance:
        gray = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(gray)
    try:
        l = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    except cv2.error:        # builds without LSD
        l = cv2.HoughLinesP(cv2.Canny(gray, 40, 120), 1, np.pi / 720, 60, minLineLength=40, maxLineGap=4)
    S = np.zeros((0, 4)) if l is None else np.asarray(l, float).reshape(-1, 4)
    return S[np.linalg.norm(S[:, 2:] - S[:, :2], axis=1) > min_len_frac * np.hypot(*gray.shape[:2])]


def _aim_error(v: np.ndarray, mid: np.ndarray, d: np.ndarray) -> np.ndarray:
    """|sin| of the angle between each segment and the ray from its midpoint to the (homogeneous) point v."""
    r = v[:2] - mid * v[2]
    return np.abs(r[:, 0] * d[:, 1] - r[:, 1] * d[:, 0]) / (np.linalg.norm(r, axis=1) + 1e-12)


def _points(f: float, R: np.ndarray) -> np.ndarray:
    """Homogeneous vanishing points (rows) of the three room axes (columns of R), centred coordinates."""
    return np.c_[f * R[0], f * R[1], R[2]]


def manhattan_focal(S: np.ndarray, size: tuple[int, int], tol_deg: float = 0.7, iters: int = 6000, seed: int = 0) -> dict | None:
    """Focal length (pixels) and room orientation from segments S of a (width, height) picture, or None."""
    w, h = size; c = np.array([w / 2.0, h / 2.0]); diag = float(np.hypot(w, h))
    if len(S) < 8:
        return None
    p0, p1 = S[:, :2] - c, S[:, 2:] - c
    L = np.cross(np.c_[p0, np.ones(len(S))], np.c_[p1, np.ones(len(S))]); L /= np.linalg.norm(L[:, :2], axis=1, keepdims=True)
    mid = (p0 + p1) / 2; d = p1 - p0; ln = np.linalg.norm(d, axis=1); d = d / ln[:, None]
    tol = np.sin(np.radians(tol_deg)); rng = np.random.default_rng(seed); pr = ln / ln.sum(); mm = DIAG_35 / diag

    def assign(f, R):
        e = np.stack([_aim_error(v, mid, d) for v in _points(f, R)])
        return e.argmin(0), e.min(0)

    best, best_score = None, -1.0
    for _ in range(iters):
        i, j, k, m = rng.choice(len(S), 4, replace=False, p=pr)
        v1, v2 = np.cross(L[i], L[j]), np.cross(L[k], L[m])
        ww = v1[2] * v2[2]
        if abs(ww) < 1e-12:
            continue
        f2 = -(v1[0] * v2[0] + v1[1] * v2[1]) / ww
        if not (F_RANGE_35[0] / mm) ** 2 < f2 < (F_RANGE_35[1] / mm) ** 2:
            continue
        f = np.sqrt(f2)
        d1 = np.array([v1[0], v1[1], f * v1[2]]); d1 /= np.linalg.norm(d1)
        d2 = np.array([v2[0], v2[1], f * v2[2]]); d2 -= d1 * (d1 @ d2); d2 /= np.linalg.norm(d2)
        R = np.c_[d1, d2, np.cross(d1, d2)]
        _, e = assign(f, R); score = ln[e < tol].sum()
        if score > best_score:
            best_score, best = score, (f, R)
    if best is None:
        return None
    f, R = best

    def refine(f, R, axes=(0, 1, 2)):
        """Least squares on the segments the chosen axes explain; the labels are re-drawn between rounds."""
        for _ in range(3):
            lab, e = assign(f, R); inl = (e < tol) & np.isin(lab, axes)
            if inl.sum() < 6:
                break
            R0, sw = R.copy(), np.sqrt(ln[inl])

            def res(x, R0=R0, lab=lab, inl=inl, sw=sw):
                V = _points(np.exp(x[0]), Rotation.from_rotvec(x[1:]).as_matrix() @ R0)
                return np.stack([_aim_error(v, mid[inl], d[inl]) for v in V])[lab[inl], np.arange(inl.sum())] * sw

            x = least_squares(res, np.r_[np.log(f), 0, 0, 0], loss="soft_l1", f_scale=tol * np.sqrt(np.median(ln))).x
            f, R = float(np.exp(x[0])), Rotation.from_rotvec(x[1:]).as_matrix() @ R0
        return f, R

    f, R = refine(f, R)
    lab, e = assign(f, R); inl = e < tol
    support = [float(ln[inl & (lab == a)].sum() / ln.sum()) for a in range(3)]
    count = [int((inl & (lab == a)).sum()) for a in range(3)]
    used = [a for a in range(3) if count[a] >= MIN_SEGMENTS and support[a] >= MIN_SUPPORT]
    lean = np.degrees(np.arcsin(np.clip(np.abs(R[2]), 0, 1)))    # how far each room axis leans out of the picture plane
    # the same focal length from each pair of directions on its own: they agree when the picture centre is the
    # optical centre and the lens keeps straight lines straight
    pairs = [refine(f, R, (a, b))[0] * mm for i, a in enumerate(used) for b in used[i + 1:] if min(lean[a], lean[b]) >= MIN_LEAN_DEG]
    weak = len(used) < 2 or np.sort(lean)[1] < MIN_LEAN_DEG or not pairs or not F_RANGE_35[0] < f * mm < F_RANGE_35[1]
    return {"f_px": f, "focal_35mm": f * mm, "fov_diag_deg": float(np.degrees(2 * np.arctan(diag / 2 / f))), "R": R,
            "labels": np.where(inl, lab, -1), "support": support, "count": count, "explained": float(ln[inl].sum() / ln.sum()),
            "pairs_35mm": [float(x) for x in sorted(pairs)], "weak": bool(weak), "second_axis_lean_deg": float(np.sort(lean)[1])}


def read_frame(video: Path, t: float, box=None) -> np.ndarray:
    """One frame at t seconds (display orientation), cropped to box = (x0, y0, x1, y1)."""
    import imageio_ffmpeg
    vf = [] if box is None else ["-vf", f"crop={box[2] - box[0]}:{box[3] - box[1]}:{box[0]}:{box[1]}"]
    raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1",
                          *vf, "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def estimate(video: Path, times, box=None, out: Path | None = None, max_side: int = 1280) -> dict:
    rows = []
    for t in times:
        img = read_frame(video, t, box)
        if img is None:
            continue
        s = min(1.0, max_side / max(img.shape[:2]))
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]; gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for enhance in (False, True):                           # faint edges get a second pass with the contrast stretched
            S = line_segments(gray, enhance=enhance); r = manhattan_focal(S, (w, h))
            if r is not None and not r["weak"]:
                break
        if r is None:
            rows.append({"t": t, "focal_35mm": None}); continue
        rows.append({"t": t, "focal_35mm": round(r["focal_35mm"], 2), "fov_diag_deg": round(r["fov_diag_deg"], 1), "weak": r["weak"], "contrast_stretched": enhance,
                     "explained": round(r["explained"], 2), "support": [round(x, 2) for x in r["support"]], "count": r["count"],
                     "pairs_35mm": [round(x, 2) for x in r["pairs_35mm"]], "second_axis_lean_deg": round(r["second_axis_lean_deg"], 1)})
        if out is not None:
            out.mkdir(parents=True, exist_ok=True); vis = img.copy()
            for seg, a in zip(S, r["labels"]):
                cv2.line(vis, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), [(160, 160, 160), (0, 190, 255), (255, 120, 0), (60, 200, 60)][a + 1], 2, cv2.LINE_AA)
            cv2.imwrite(str(out / f"focal_lines_{t:06.2f}.jpg"), vis)
    ok = [r["focal_35mm"] for r in rows if r["focal_35mm"] and not r["weak"]]
    lo = [r["pairs_35mm"][0] for r in rows if r["focal_35mm"] and not r["weak"]]; hi = [r["pairs_35mm"][-1] for r in rows if r["focal_35mm"] and not r["weak"]]
    return {"video": str(video), "box": box, "frames": rows, "n_used": len(ok),
            "focal_35mm": round(float(np.median(ok)), 1) if ok else None,
            "focal_35mm_iqr": [round(float(x), 1) for x in np.percentile(ok, [25, 75])] if ok else None,
            "pairs_35mm": [round(float(np.median(lo)), 1), round(float(np.median(hi)), 1)] if ok else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video"); ap.add_argument("--box", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--times", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--out", help="folder for the per-frame pictures of the segments each direction explains")
    a = ap.parse_args(argv)
    res = estimate(Path(a.video), a.times, a.box, Path(a.out) if a.out else None)
    for r in res["frames"]:
        print(f"  t={r['t']:6.2f} s  " + ("no estimate" if not r["focal_35mm"] else f"{r['focal_35mm']:6.2f} mm  fov {r['fov_diag_deg']:5.1f} deg  segments per direction {r['count']}"
                                       + (f"  pairs {r['pairs_35mm'][0]:.1f} to {r['pairs_35mm'][-1]:.1f}" if r["pairs_35mm"] else "") + ("  weak" if r["weak"] else "")))
    if res["focal_35mm"] is None or 2 * res["n_used"] < len(res["frames"]):
        print("too few usable frames: the room's lines do not fix the focal length from this view"); return 1
    print(f"focal_35mm {res['focal_35mm']} (quartiles {res['focal_35mm_iqr'][0]} to {res['focal_35mm_iqr'][1]} over {res['n_used']} frames; pairs of directions on their own: {res['pairs_35mm'][0]} to {res['pairs_35mm'][1]})")
    if a.out:
        (Path(a.out) / "focal.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
