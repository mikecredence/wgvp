"""One camera's walking speed on a rendered sample: the lifter's own path against a keypoint fit of the same view.

    python tools/one_camera_speed.py <root> [--pair "Sub 001/Sub001_selfpace=sample/HC900_gt.npz" ...] [--out docs/one_camera.json]

<root> is a data folder the pipeline has run on (GPU stage and merge included). Without --pair the six recordings of
the partner layout (`tools/make_partner_layout.py`) are scored against the bundled samples' truths.

The question is what one camera knows about the distance a walker covers along its line of sight. Each view of each
recording is taken by itself and its pelvis speed is read two ways:

  lifter   the view's own world path (smpl_global.npz), the path the merge is fused from
  fit      `wgvp.jointfit.joint_fit` handed that one view twice: a body of fixed size fitted to the view's 2D
           keypoints, so the distance to the walker comes from the body's size in the picture and the lens

Both are scored as mean speed over the truth's mean speed, on the frames where the truth moves faster than 0.25 m/s
and the view's leg joints are usable, split by the walker's true direction against the camera's line of sight:
"along" within 30 deg of it, "across" more than 60 deg off it. A two-panel recording cut from ONE camera (the
bundled OWN02) is this case: the joint fit of such a recording is the "fit" column here, and the merge is the
"lifter" column.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from wgvp.jointfit import find_body_models, joint_fit, load_skeleton, load_views, to_world   # noqa: E402
from tools.compare_jointfit import PAIRS                                                     # noqa: E402

MOVING_MPS = 0.25
ALONG, ACROSS = 0.87, 0.5                # |cos| of the walk against the sight line: within 30 deg, more than 60 deg off
LEG_JOINTS = [0, 1, 3, 4, 6, 7]          # hips, knees, ankles (body index)
MIN_FRAMES = 30


def _vel(p: np.ndarray, fps: float, k: int = 2) -> np.ndarray:
    out = np.full((len(p), 2), np.nan); out[k:-k] = (p[2 * k:, [0, 2]] - p[:-2 * k, [0, 2]]) * fps / (2 * k); return out


def one_camera(root: Path, sess: str, rec: str, gt: Path, skeleton) -> dict | None:
    d = root / "wgvp_output" / sess / "deliver" / rec
    if not (d / "merged_smpl.npz").exists() or not gt.exists():
        print(f"  {rec}: no merge or no truth, skipped"); return None
    m = np.load(d / "merged_smpl.npz", allow_pickle=True); G = dict(np.load(gt, allow_pickle=True)); fps = float(m["fps"]); frames = np.asarray(m["frame_idx"]); T = len(frames)
    views = load_views(d, m)
    if views is None:
        print(f"  {rec}: a view lacks its camera-frame file, skipped"); return None
    gi = {int(f): i for i, f in enumerate(G["frame_idx"])}; sel = np.array([gi.get(int(f), -1) for f in frames]); has = sel >= 0
    pt = np.full((T, 3), np.nan); pt[has] = G["transl"].astype(float)[sel[has]]; vt = _vel(pt, fps); st = np.linalg.norm(vt, axis=1)
    fi = {int(f): i for i, f in enumerate(frames)}; out = {}
    for v in views:
        name = v["name"]; g = np.load(d / name / "smpl_global.npz"); lp = np.full((T, 3), np.nan)
        for j, f in enumerate(g["frame_idx"]):
            if int(f) in fi:
                lp[fi[int(f)]] = g["transl"][j]
        sol = joint_fit([v, dict(v)], {"global_orient": v["glob_root"], "body_pose": v["cam_body"].reshape(T, 63)}, skeleton)
        if sol is None:
            continue
        fp = to_world(sol)[2][:, 0]; sl = np.linalg.norm(_vel(lp, fps), axis=1); sf = np.linalg.norm(_vel(fp, fps), axis=1)
        M = np.diag([1.0, -1.0, -1.0, 1.0]) @ np.linalg.inv(G[f"{name}_cam_pose"].astype(np.float64)); centre = -M[:3, :3].T @ M[:3, 3]
        with np.errstate(all="ignore"):
            sight = pt[:, [0, 2]] - centre[[0, 2]]; sight /= np.linalg.norm(sight, axis=1, keepdims=True); along = np.abs((vt * sight).sum(1)) / st
            ok = v["usable"][:, LEG_JOINTS].all(1) & v["detected"] & np.isfinite(st) & (st > MOVING_MPS) & np.isfinite(sl) & np.isfinite(sf); r = {}
            for tag, k in (("all", ok), ("along", ok & (along > ALONG)), ("across", ok & (along < ACROSS))):
                if k.sum() >= MIN_FRAMES:
                    r[tag] = {"n": int(k.sum()), "lifter_speed_ratio": float(sl[k].mean() / st[k].mean()), "fit_speed_ratio": float(sf[k].mean() / st[k].mean()),
                              "lifter_speed_rmse": float(np.sqrt(np.mean((sl[k] - st[k]) ** 2))), "fit_speed_rmse": float(np.sqrt(np.mean((sf[k] - st[k]) ** 2)))}
        out[name] = r
    return out


def table(res: dict) -> str:
    rows = ["| recording, view | walking along the sight line: frames, lifter, fit | across it: frames, lifter, fit |", "|---|---|---|"]
    cell = lambda x: f"{x['n']}, {x['lifter_speed_ratio']:.2f}, {x['fit_speed_ratio']:.2f}" if x else "-"
    for rec, r in res.items():
        rows += [f"| {rec}, {name} | {cell(v.get('along'))} | {cell(v.get('across'))} |" for name, v in r.items()]
    for tag in ("along", "across"):
        c = [v[tag] for r in res.values() for v in r.values() if tag in v]
        if c:
            rows.append(f"median of {len(c)} views {tag}: lifter {np.median([x['lifter_speed_ratio'] for x in c]):.2f}, fit {np.median([x['fit_speed_ratio'] for x in c]):.2f} of the true speed")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("--pair", action="append", help='"<session>/<recording>=<truth npz>" (repeatable; default: the partner layout\'s six)')
    ap.add_argument("--body-models"); ap.add_argument("--out", help="write the numbers as JSON")
    a = ap.parse_args(argv); root = Path(a.root)
    from wgvp.config import load_config
    skel = load_skeleton(find_body_models(load_config(root), a.body_models)); res = {}
    for pr in a.pair or PAIRS:
        key, gt = pr.split("="); sess, rec = key.split("/"); gt = Path(gt) if Path(gt).is_absolute() else PKG / gt
        r = one_camera(root, sess, rec, gt, skel)
        if r:
            res[rec] = r; print(f"  {rec}: " + " | ".join(f"{n} {t} {x['lifter_speed_ratio']:.2f}/{x['fit_speed_ratio']:.2f}" for n, v in r.items() for t, x in v.items() if t != "all"), flush=True)
    if not res:
        return 1
    print(); print(table(res))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
