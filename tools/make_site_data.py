"""The data behind docs/index.html's interactive demo, from the pipeline's own outputs.

    python tools/make_site_data.py --dryrun-out <a finished dry run> --two-camera-out <a finished run of examples/two_camera> [--out docs/site]

Writes site/find_the_walker.json (the rendered sample's left view: every tracked person's pelvis signal, the sensor's
vertical displacement, and each track's correlation against it over the lag), site/two_views.json (the two-camera
recording OWN03: both views' 2D keypoints, the training target's joint positions frame by frame, each joint's view
weights, and the QC numbers) and site/results.json (the errors against the truth: the six rendered sessions from
docs/jointfit.json and the real recording's walked distance from docs/chalk_walk.json). Numbers only; no pixels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def r3(a, nd=3):
    return [round(float(x), nd) if np.isfinite(x) else None for x in np.asarray(a).ravel()]


def find_the_walker(dry: Path) -> dict:
    from wgvp.imu_ref import load_imu_ref
    from wgvp.signals import bandpass, match_signals, resample_to_grid
    from wgvp.tracking import TrackSet, hip_vertical_signal
    w = dry / "wgvp_output" / "HC900" / "work"
    ts = TrackSet.load(w / "HC900_SP_DeID" / "tracks_left.npz"); imu = load_imu_ref(w / "imu_ref" / "hc900_selfpace_imu_ref.npz")
    sel = json.loads((dry / "wgvp_output/HC900/deliver/hc900_selfpace/left/selection.json").read_text(encoding="utf-8"))["primary"]
    exp = json.loads((REPO / "sample/expected.json").read_text(encoding="utf-8"))
    ref = imu["disp_up"].astype(float); fs = float(imu["fs"]); step = int(round(fs / 20))                      # the sensor at 20 Hz for the page
    ref_n = ref / (np.nanstd(ref) or 1.0)
    out = {"what": "the dry-run sample, left view", "true_offset_s": exp["offset_s"], "fps": exp["fps"], "video_s": 1594 / exp["fps"],
           "imu": {"t": r3(imu["t"][::step], 2), "x": r3(ref_n[::step], 2)}, "tracks": []}
    for tid in ts.track_ids():
        tr = ts.track(int(tid)); t, y = hip_vertical_signal(tr)
        grid, v = resample_to_grid(t, y, 20.0, float(np.nanmin(t)), float(np.nanmax(t)))
        ok = np.isfinite(v); vb = v.copy(); vb[~ok] = 0.0; vb = bandpass(vb, 20.0); vb = vb / (np.nanstd(vb[ok]) or 1.0); vb[~ok] = np.nan
        m = match_signals(t, y, ref); frag = sel["fragments"].get(str(int(tid)), {})
        lags = np.asarray(m["lags_s"]); r = np.asarray(m["r_fine_fn"]); keep = (lags >= -8) & (lags <= 12)
        out["tracks"].append({"tid": int(tid), "role": frag.get("role", ""), "duration_s": round(float(frag.get("duration_s", 0)), 1),
                              "t": r3(grid, 2), "x": r3(vb, 2), "lags": r3(lags[keep][::5], 2), "r": r3(r[keep][::5], 3),
                              "best_lag_s": round(float(m["offset_s"]), 3), "r_best": round(float(m["r_fine"]), 3), "chosen": int(tid) in sel["track_ids"]})
    return out


def two_views(run: Path) -> dict:
    """OWN03, the two-camera recording: the target (the joint fit) with the merge's per-joint view weights."""
    from wgvp.fusion import SMPLX_PARENTS
    from wgvp.jointfit import body_joints, mean_skeleton
    d = run / "wgvp_output/OWN03/deliver/OWN03_selfpace"
    tgt = json.loads((d / "target.json").read_text(encoding="utf-8")); M = np.load(d / tgt["target"], allow_pickle=True); G = np.load(d / "merged_smpl.npz", allow_pickle=True)
    J = body_joints(M["global_orient"], M["body_pose"], M["transl"], M["betas"], mean_skeleton())             # (T, 22, 3), y up
    T = len(M["frame_idx"]); views = [str(v) for v in G["views"]]
    kp = {}
    for v in views:
        S = np.load(d / v / "subject_track.npz", allow_pickle=True); K = np.full((T, 17, 2), np.nan); idx = {int(f): j for j, f in enumerate(M["frame_idx"])}
        for f, p in zip(S["frame_idx"], S["kp2d"]):
            if int(f) in idx:
                q = p[:, :2].copy(); q[p[:, 2] < 0.3] = np.nan; K[idx[int(f)]] = q
        kp[v] = {"view_box": [int(x) for x in S["view_box"]], "xy": r3(K, 0)}
    sync = json.loads((d / "sync.json").read_text(encoding="utf-8")); rc = json.loads((d / "reconcile.json").read_text(encoding="utf-8"))
    pair = next(iter(rc["pairs"].values())); mg = json.loads((d / "merge.json").read_text(encoding="utf-8"))
    return {"what": "OWN03, a real two-camera recording: side view left, front view right", "fps": float(M["fps"]), "n_frames": T, "frame_idx": [int(f) for f in M["frame_idx"]],
            "imu_t_s": r3(M["imu_t_s"], 3), "views": views, "reference_view": str(M["reference_view"]), "mirrored": [bool(x) for x in G["view_mirrored"]],
            "target": tgt["method"], "body_height_m": tgt.get("body_height_m"), "parents": SMPLX_PARENTS, "joint_names": ["Pelvis"] + [str(j) for j in M["joints"]],
            "joints3d": r3(J, 2), "weight": r3(M["weight"], 2), "export_ok": [bool(x) for x in M["export_ok"]],
            "video": {"file": "OWN03_SP.mp4", "width": 1516, "height": 960}, "kp2d": kp,
            "sync": {"offset_s": round(sync["offset_s"], 3), "source": sync["source"], "r_fine": round(sync["xcorr"]["r_fine"], 3), "status": sync["status"], "flags": sync["flags"],
                     "flash_minus_xcorr_s": round(sync["agreement"]["flash_minus_xcorr_s"], 3)},
            "reconcile": {"view_offsets_s": {k: round(v, 3) for k, v in pair["view_offsets_s"].items()}, "legs_median_deg": round(pair["pose"]["legs_median_deg"], 1),
                          "all_median_deg": round(pair["pose"]["all_median_deg"], 1), "knee_r": round(pair["pose"]["knee_residual_lag"]["r_best"], 3)},
            "merge": {"export_frac": mg["export"]["frac_of_lifted_frames"], "n_runs": mg["export"]["n_runs"]}}


SESSIONS = [("Sub001_selfpace", "HC900", "self-paced walk"), ("Sub002_selfpace", "HC901", "self-paced walk"), ("Sub003_selfpace", "HC902", "self-paced walk"),
            ("Sub001_tug", "HC903", "timed up and go"), ("Sub003_hurriedpace", "HC904", "hurried pace"), ("Sub004_selfpace_mat", "HC905", "walk on the mat")]


def results() -> dict:
    """The errors against the truth: the six rendered sessions (joint rotations, speed) and the real recording's walked distance."""
    jf = json.loads((REPO / "docs/jointfit.json").read_text(encoding="utf-8")); ck = json.loads((REPO / "docs/chalk_walk.json").read_text(encoding="utf-8"))
    real = ck["OWN03, published half-size clip"]
    rows = [{"id": hc, "label": f"{hc} {what}", "kind": "rendered",
             **{f"{m}_{k}": round(jf[key][n][k], 3) for m, n in (("merge", "merge"), ("fit", "joint fit")) for k in ("rot_legs", "rot_root", "knee_rmse", "speed_ratio", "path_med_m")}}
            for key, hc, what in SESSIONS]
    rows.append({"id": "OWN03", "label": "OWN03 real, chalk marks", "kind": "real", "truth_m": round(real["truth"]["distance_m"], 3), "subject_height_m": real["truth"]["by_height"]["subject_height_m"],
                 "merge_speed_ratio": round(real["estimates"]["merge"]["slope"], 3), "fit_speed_ratio": round(real["estimates"]["joint fit"]["slope"], 3),
                 "merge_rms_m": round(real["estimates"]["merge"]["rms_m"], 3), "fit_rms_m": round(real["estimates"]["joint fit"]["rms_m"], 3),
                 "merge_body_m": round(real["body_m"]["merge"], 3), "fit_body_m": round(real["body_m"]["joint fit"], 3), "passes": len(real["estimates"]["merge"]["passes"])})
    return {"what": "errors against the truth, per session", "sessions": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dryrun-out", required=True); ap.add_argument("--two-camera-out", required=True); ap.add_argument("--out", default=str(REPO / "docs/site"))
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, d in (("find_the_walker.json", find_the_walker(Path(a.dryrun_out))), ("two_views.json", two_views(Path(a.two_camera_out))), ("results.json", results())):
        (out / name).write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8"); print(f"{out / name}: {(out / name).stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
