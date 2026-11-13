"""A walk between two ground marks a known distance apart: the pelvis path read from one side view, and every
estimate of the recording against it.

    python tools/chalk_walk.py <root> --session S --recording R --distance-m 4.1148 \
        --marks "148,829 246,787 1783,758 1703,730" [--view left] [--span 3.0,28.5] [--label NAME] \
        [--subject-height-m 1.651] [--truth-root OTHER_ROOT] [--body-models DIR] [--out docs/chalk_walk.json]

The recording is a straight walk back and forth between two lines drawn on the ground across the walk (chalk, tape),
`--distance-m` apart along the ground, seen by a still camera that looks across the walk. `--marks` is two image
points on the first line and two on the second, in the pixels of that view's panel. Nothing else is measured: no
lens, no camera height, no subject height.

**The truth.** The two lines are parallel on the ground, so their images meet at a vanishing point on the ground's
horizon, and a ground point's position across the lines is linear in the slope of the image line that joins it to
that point: X = h (x - x_vp) / (y - y_vp), with h set by the known distance. The lens does not enter. The pelvis is
placed over the ground at the walker's feet (the bottom of the subject's box, smoothed over a step). The horizon is
level in the picture on level ground. On ground that rises along the walk it turns about the vanishing point, and
the angle taken is the one that keeps the walker's size (box height over the feet row's distance below the horizon)
constant along the walk. The stature that comes out is the height of the walker's box in mid-stride, shoes on, IF
the distance is right: it is printed as the check on the marks, and a walker of another height means another
distance.

**The scores.** Each estimate's horizontal pelvis path is put on its own main axis and compared with the truth's
position: the slope of estimate on truth (1 = the right distance), the spread around that line, the path's spread
off its axis (a straight walk has none) and the travel from turn to turn. Every body-model estimate takes its scale
from the size of its body, so the slope is also given in body heights: slope x the truth's stature / the estimate's
own standing height. Rows: the merge, the joint fit when present, and each view alone read two ways, the lifter's
own world path and `wgvp.jointfit.joint_fit` handed that view twice (as in `tools/one_camera_speed.py`).

**The other reading.** The picture fixes the distance between the lines in walker heights and nothing more. With
`--subject-height-m` (standing, barefoot) the same picture is read from the height instead: the distance the height
implies, and every slope under it. Each slope is also given with the estimate's body rescaled to that height, which
is what a pipeline told the subject's height could deliver. When the tape and the height disagree the picture cannot
say which one is off, and both readings are printed.

**A truth from another run.** `--truth-root` takes the side track from another root that holds the same recording
(same session, recording and frame numbers), for a run whose own side view cuts the body and cannot place the feet
or size the walker.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from wgvp.config import load_config                                                          # noqa: E402
from wgvp.jointfit import find_body_models, joint_fit, load_skeleton, load_views, to_world   # noqa: E402

MAX_SLOPE = 0.15                 # rad: the search range for the ground's slope along the walk
FOOT_WIN, SIZE_WIN = 15, 31      # frames: the feet row over a step, the box height over a stride
EDGE_PX = 6                      # a box this close to the panel's edge is cut by it
HIP_EDGE_PX = 40                 # a pelvis this close to the panel's side is leaving the picture: the turn there reads short
TURN_GAP_S, TURN_PROM_M, TURN_WIN_S = 2.0, 1.0, 0.8
MIN_FRAMES = 100
BOX_OVER_HEIGHT = 1.008          # the walker's box in mid-stride, shoes on, over the barefoot standing height: 3 cm of sole, 1 % of dip


def _line(p, q):
    return np.cross([*p, 1.0], [*q, 1.0])


def ground_path(marks, distance_m, hip_x, foot_y, box_h, whole) -> dict:
    """Position across two ground lines from one view. marks: four (x, y), two points on each line. hip_x, foot_y,
    box_h (T,): the pelvis's column, the feet row and the box height, in the marks' pixels. whole (T,) bool: frames
    where the body is uncut. Returns the level and the sloped reading: X (T,) in metres, zero under the vanishing
    point and growing from the first line to the second, with the camera's height over the ground, the two lines'
    positions and the walker's stature (median and SD over the uncut frames)."""
    marks = np.asarray(marks, float).reshape(4, 2); vp = np.cross(_line(marks[0], marks[1]), _line(marks[2], marks[3])); vp = vp[:2] / vp[2]
    k = [(p[0] - vp[0]) / (p[1] - vp[1]) for p in (marks[0], marks[2])]

    def turned(x, y, th):        # the picture turned about the vanishing point until the ground's horizon is level
        c, s = np.cos(th), np.sin(th); return vp[0] + c * (x - vp[0]) + s * (y - vp[1]), vp[1] - s * (x - vp[0]) + c * (y - vp[1])

    def size(th):
        return box_h / (turned(hip_x, foot_y, th)[1] - vp[1])

    th = float(minimize_scalar(lambda a: np.std(np.log(size(a)[whole])), bounds=(-MAX_SLOPE, MAX_SLOPE), method="bounded").x)
    out = {"vanishing_point_px": vp.tolist(), "ground_rise_deg": float(-np.degrees(th))}      # positive: rises to the picture's right
    for name, a in (("level", 0.0), ("sloped", th)):
        kk = [(q * np.cos(a) + np.sin(a)) / (np.cos(a) - q * np.sin(a)) for q in k]; h = distance_m / (kk[1] - kk[0])
        x2, y2 = turned(hip_x, foot_y, a); st = abs(h) * size(a)
        out[name] = {"X": h * (x2 - vp[0]) / (y2 - vp[1]), "camera_height_m": float(abs(h)), "lines_m": [float(h * kk[0]), float(h * kk[1])],
                     "stature_m": float(np.median(st[whole])), "stature_sd_m": float(np.std(st[whole]))}
    return out


def height_reading(truth: dict, distance_m: float, height_m: float) -> dict:
    """The picture read from the walker's standing height instead of the distance. truth: one reading of
    `ground_path` (its stature and camera height at `distance_m`). Returns the distance between the lines and the
    camera's height that the height implies, and the factor every slope takes under them."""
    box = BOX_OVER_HEIGHT * height_m; f = box / truth["stature_m"]
    return {"subject_height_m": float(height_m), "box_m": float(box), "distance_in_boxes": float(distance_m / truth["stature_m"]),
            "distance_m": float(f * distance_m), "camera_height_m": float(f * truth["camera_height_m"]), "slope_factor": float(1 / f)}


def side_track(rec_dir: Path, view: str) -> dict:
    """The subject's track in one view, in the panel's pixels: pelvis column, feet row, box height, the frames where
    the body is uncut and those where the pelvis is at the panel's side."""
    z = np.load(Path(rec_dir) / view / "subject_track.npz", allow_pickle=True); vb = z["view_box"].astype(float)
    b = z["bbox_xyxy"].astype(float) - np.tile(vb[:2], 2); W, H = vb[2] - vb[0], vb[3] - vb[1]; idx = np.arange(len(b))
    whole = (b[:, 0] > EDGE_PX) & (b[:, 1] > EDGE_PX) & (b[:, 2] < W - EDGE_PX) & (b[:, 3] < H - EDGE_PX)
    h = np.interp(idx, idx[whole], (b[:, 3] - b[:, 1])[whole]); hip = z["kp2d"][:, [11, 12], 0].astype(float).mean(1) - vb[0]
    return {"frame_idx": np.asarray(z["frame_idx"]), "hip_x": hip, "foot_y": median_filter(b[:, 3], FOOT_WIN, mode="nearest"),
            "box_h": median_filter(h, SIZE_WIN, mode="nearest"), "whole": whole, "at_edge": (hip < HIP_EDGE_PX) | (hip > W - HIP_EDGE_PX)}


def standing_height(body_models, betas) -> float:
    """Height of the shaped neutral SMPL-X body at rest: the scale a body-model estimate carries."""
    z = np.load(Path(body_models) / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True); b = np.asarray(betas, float); b = np.nanmedian(b, 0) if b.ndim == 2 else b
    return float(np.ptp((np.asarray(z["v_template"], float) + np.asarray(z["shapedirs"][:, :, :10], float) @ b[:10])[:, 1]))


def find_turns(Xt, fps) -> list[tuple[int, int]]:
    """(frame, +1 at the far line or -1 at the near one) for each end of the walk, in order."""
    kw = {"distance": max(1, int(TURN_GAP_S * fps)), "prominence": TURN_PROM_M}; X = np.where(np.isfinite(Xt), Xt, np.nanmedian(Xt))
    return sorted([(int(i), 1) for i in find_peaks(X, **kw)[0]] + [(int(i), -1) for i in find_peaks(-X, **kw)[0]])


def score(P, Xt, ok, turns, fps, at_edge) -> dict | None:
    """One estimate's horizontal path (T, 2) against the truth's position (T,) on the frames `ok`. A pass that ends at
    a turn where the pelvis reached the panel's side has a truth that is a lower bound."""
    fin = np.isfinite(P).all(1); ok = ok & fin & np.isfinite(Xt)
    if ok.sum() < MIN_FRAMES:
        return None
    c = P[ok].mean(0); Q = P[ok] - c; u = np.linalg.svd(Q, full_matrices=False)[2][0]; s = np.full(len(P), np.nan); s[fin] = (P[fin] - c) @ u
    s = -s if np.corrcoef(s[ok], Xt[ok])[0, 1] < 0 else s
    a, b = np.polyfit(Xt[ok], s[ok], 1); off = Q - np.outer(Q @ u, u); win = int(round(TURN_WIN_S * fps)); ext = []
    for i, sg in turns:
        w = slice(max(0, i - win), i + win + 1); e, x = s[w][np.isfinite(s[w])], Xt[w][np.isfinite(Xt[w])]
        ext.append((sg * np.max(sg * e) if len(e) > 10 else np.nan, sg * np.max(sg * x), bool(at_edge[w].any())))
    return {"n": int(ok.sum()), "slope": float(a), "rms_m": float(np.sqrt(np.mean((s[ok] - a * Xt[ok] - b) ** 2))),
            "off_axis_rms_m": float(np.sqrt(np.mean((off ** 2).sum(1)))),
            "passes": [{"truth_m": float(abs(q[1] - p[1])), "est_m": float(abs(q[0] - p[0])), "truth_is_lower_bound": p[2] or q[2]} for p, q in zip(ext[:-1], ext[1:])]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root"); ap.add_argument("--session", required=True); ap.add_argument("--recording", required=True)
    ap.add_argument("--marks", required=True, help='"x,y x,y x,y x,y": two points on the first line, two on the second, panel pixels')
    ap.add_argument("--distance-m", type=float, required=True); ap.add_argument("--view", default="left", help="the view that looks across the walk")
    ap.add_argument("--span", help="start,end in seconds of the recording: the walk, without what comes before and after")
    ap.add_argument("--subject-height-m", type=float, help="the walker's standing height, barefoot: adds the reading that trusts it over the distance")
    ap.add_argument("--truth-root", help="another root with the same recording, whose view gives the truth (for a run whose side view cuts the body)")
    ap.add_argument("--label"); ap.add_argument("--body-models"); ap.add_argument("--out")
    a = ap.parse_args(); root = Path(a.root); d = root / "wgvp_output" / a.session / "deliver" / a.recording
    dt = Path(a.truth_root) / "wgvp_output" / a.session / "deliver" / a.recording if a.truth_root else d
    marks = [[float(v) for v in p.split(",")] for p in a.marks.split()]
    m = np.load(d / "merged_smpl.npz", allow_pickle=True); frames = np.asarray(m["frame_idx"]); T = len(frames); fps = float(m["fps"]); t = frames / fps
    tk = side_track(dt, a.view); g = ground_path(marks, a.distance_m, tk["hip_x"], tk["foot_y"], tk["box_h"], tk["whole"])
    Xs = np.convolve(np.pad(g["sloped"]["X"], 2, mode="edge"), np.ones(5) / 5, "valid"); Xt = np.interp(frames, tk["frame_idx"], Xs, left=np.nan, right=np.nan)
    at_edge = np.interp(frames, tk["frame_idx"], tk["at_edge"].astype(float)) > 0
    lo, hi = (float(v) for v in a.span.split(",")) if a.span else (-np.inf, np.inf); ok = (t > lo) & (t < hi)
    turns = [(i, sg) for i, sg in find_turns(np.where(ok, Xt, np.nan), fps) if ok[i]]; st = g["sloped"]["stature_m"]
    hr = height_reading(g["sloped"], a.distance_m, a.subject_height_m) if a.subject_height_m else None
    res = {"truth": {"view": a.view, "from_another_run": bool(a.truth_root), "distance_m": a.distance_m, "marks_px": marks,
                     "turns_s": [round(float(t[i]), 2) for i, _ in turns], **({"by_height": hr} if hr else {}),
                     **{k: v for k, v in g.items() if k not in ("level", "sloped")},
                     **{n: {k: v for k, v in g[n].items() if k != "X"} for n in ("level", "sloped")}}, "estimates": {}, "body_m": {}}
    bm = find_body_models(load_config(root), a.body_models); skel = load_skeleton(bm)

    def add(name, P, betas):
        r = score(P, Xt, ok, turns, fps, at_edge)
        if r is not None:
            b = standing_height(bm, betas); res["body_m"][name] = b; r["slope_in_body_heights"] = r["slope"] * st / b; res["estimates"][name] = r
            if hr:
                k = a.subject_height_m / b; r["slope_body_rescaled"] = r["slope"] * k
                r["by_height"] = {"slope": r["slope"] * hr["slope_factor"], "slope_body_rescaled": r["slope"] * hr["slope_factor"] * k}

    add("merge", m["transl"][:, [0, 2]].astype(float), m["betas"])
    if (d / "jointfit_smpl.npz").exists():
        j = np.load(d / "jointfit_smpl.npz", allow_pickle=True); add("joint fit", j["transl"][:, [0, 2]].astype(float), j["betas"])
        Rb = j["cam_b_from_a_R"].astype(float); c = -Rb.T @ j["cam_b_from_a_t"].astype(float)
        res["camera_b"] = {"baseline_m": float(np.linalg.norm(c)), "axis_angle_deg": float(np.degrees(np.arccos(np.clip(Rb[2, 2], -1, 1)))),
                           "reproj_px_median": [float(np.nanmedian(x)) for x in j["reproj_px"]], "seen_both_frac": float(np.mean(j["seen_both"]))}
    fi = {int(f): i for i, f in enumerate(frames)}
    for v in load_views(d, m) or []:
        gl = np.load(d / v["name"] / "smpl_global.npz"); lp = np.full((T, 3), np.nan); src = [k for k, f in enumerate(gl["frame_idx"]) if int(f) in fi]
        lp[[fi[int(gl["frame_idx"][k])] for k in src]] = gl["transl"][src]; add(f"{v['name']} alone, lifter path", lp[:, [0, 2]], gl["betas"])
        sol = joint_fit([v, dict(v)], {"global_orient": v["glob_root"], "body_pose": v["cam_body"].reshape(T, 63)}, skel)
        if sol is not None:
            add(f"{v['name']} alone, keypoint fit", to_world(sol)[2][:, 0][:, [0, 2]], sol["betas"])
    s = res["truth"]["sloped"]
    print(f"ground rises {res['truth']['ground_rise_deg']:+.2f} deg to the right; camera {s['camera_height_m']:.3f} m over it; lines at {s['lines_m'][0]:+.3f} and "
          f"{s['lines_m'][1]:+.3f} m; stature {s['stature_m']:.3f} m (SD {s['stature_sd_m']:.3f}; level model {res['truth']['level']['stature_sd_m']:.3f}); {len(turns)} turns")
    for k, r in res["estimates"].items():
        print(f"{k:28s} n {r['n']:4d}  slope {r['slope']:.3f}  rms {r['rms_m']:.3f} m  off-axis {r['off_axis_rms_m']:.3f} m  body {res['body_m'][k]:.3f} m  "
              f"in body heights {r['slope_in_body_heights']:.3f}  passes " + " ".join(f"{p['est_m']:.2f}/{p['truth_m']:.2f}{'+' if p['truth_is_lower_bound'] else ''}" for p in r["passes"]))
    if hr:
        print(f"the lines are {hr['distance_in_boxes']:.3f} walker boxes apart. A walker of {hr['subject_height_m']:.3f} m (box {hr['box_m']:.3f} m) puts them "
              f"{hr['distance_m']:.3f} m apart, not {a.distance_m:.3f}, and the camera {hr['camera_height_m']:.3f} m up. Slopes, as delivered and with the body "
              f"rescaled to the walker:")
        for k, r in res["estimates"].items():
            print(f"{k:28s} distance trusted {r['slope']:.3f} {r['slope_body_rescaled']:.3f}   height trusted {r['by_height']['slope']:.3f} "
                  f"{r['by_height']['slope_body_rescaled']:.3f}")
    if a.out:
        out = Path(a.out); doc = json.loads(out.read_text()) if out.exists() else {}; doc[a.label or a.recording] = res; out.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
