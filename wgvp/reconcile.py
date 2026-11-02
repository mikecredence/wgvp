"""Two-angle reconciliation on the IMU timeline (per recording).

CPU level (always): the two cameras' pelvis-vertical signals, each mapped to IMU time by its own sync offset, must
agree (correlation, residual lag ~0). Pose level (after GVHMR): per-joint agreement of the SMPL body pose
(global frame) between cameras on the IMU timeline: geodesic angle per joint (median, deg), correlation of knee /
hip flexion magnitudes, residual lag of the knee signal.
"""
from __future__ import annotations

import numpy as np

from .signals import bandpass, masked_xcorr, parabolic_peak, resample_to_grid, FS_MATCH

SMPLX_BODY_JOINTS = ["L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot",
                     "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
                     "L_Wrist", "R_Wrist"]


def signal_agreement(tA: np.ndarray, xA: np.ndarray, tB: np.ndarray, xB: np.ndarray, fs: float = FS_MATCH,
                     max_lag_s: float = 1.0, label: str = "") -> dict:
    """Both signals already on IMU time. Returns correlation at zero lag, best lag and r there."""
    t0 = max(np.nanmin(tA), np.nanmin(tB)); t1 = min(np.nanmax(tA), np.nanmax(tB))
    if t1 - t0 < 3.0:
        return {"label": label, "ok": False, "reason": f"overlap on IMU time only {t1 - t0:.1f}s"}
    _, a = resample_to_grid(tA, xA, fs, t0, t1); _, b = resample_to_grid(tB, xB, fs, t0, t1)
    a = bandpass(a, fs); b = bandpass(b, fs)
    lags, r, n = masked_xcorr(a, b, int(max_lag_s * fs), min_overlap=int(2 * fs))
    if not np.isfinite(r).any():
        return {"label": label, "ok": False, "reason": "no valid overlap"}
    i = int(np.nanargmax(r)); i0 = int(np.argmin(np.abs(lags)))
    return {"label": label, "ok": True, "overlap_s": float(t1 - t0), "r_zero_lag": float(r[i0]) if np.isfinite(r[i0]) else None,
            "r_best": float(r[i]), "residual_lag_s": float((lags[0] + parabolic_peak(r, i)) / fs)}


def _geodesic_deg(aa1: np.ndarray, aa2: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    r1 = R.from_rotvec(aa1.reshape(-1, 3)); r2 = R.from_rotvec(aa2.reshape(-1, 3))
    return np.degrees((r1.inv() * r2).magnitude()).reshape(aa1.shape[:-1])


def pose_agreement(tA: np.ndarray, poseA: np.ndarray, tB: np.ndarray, poseB: np.ndarray, fs: float = 30.0) -> dict:
    """poseX: (T, 63) SMPL-X body_pose axis-angle (21 joints) on IMU time tX. Resample both to fs on the overlap."""
    t0 = max(tA.min(), tB.min()); t1 = min(tA.max(), tB.max())
    if t1 - t0 < 3.0:
        return {"ok": False, "reason": f"overlap on IMU time only {t1 - t0:.1f}s"}
    grid = np.arange(t0, t1, 1 / fs)
    A = np.stack([np.interp(grid, tA, poseA[:, j]) for j in range(63)], 1).reshape(-1, 21, 3)
    B = np.stack([np.interp(grid, tB, poseB[:, j]) for j in range(63)], 1).reshape(-1, 21, 3)
    geo = _geodesic_deg(A, B)  # (T, 21)
    per_joint = {SMPLX_BODY_JOINTS[j]: {"median_deg": float(np.median(geo[:, j])), "p90_deg": float(np.percentile(geo[:, j], 90))} for j in range(21)}
    magA = np.linalg.norm(A, axis=2); magB = np.linalg.norm(B, axis=2)
    corr = {}
    for name, j in (("L_Knee", 3), ("R_Knee", 4), ("L_Hip", 0), ("R_Hip", 1), ("L_Elbow", 17), ("R_Elbow", 18)):
        a, b = magA[:, j], magB[:, j]
        corr[name] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else None
    knee = signal_agreement(grid, magA[:, 3] + magA[:, 4], grid, magB[:, 3] + magB[:, 4], fs=fs, label="knee_flexion_sum")
    legs = [0, 1, 3, 4, 6, 7]
    return {"ok": True, "overlap_s": float(t1 - t0), "n_samples": int(len(grid)), "per_joint_geodesic": per_joint,
            "legs_median_deg": float(np.median(geo[:, legs])), "all_median_deg": float(np.median(geo)),
            "flexion_corr": corr, "knee_residual_lag": knee}


def reconcile_recording(cams: dict, tol_frames: float = 3.0) -> dict:
    """cams: camera_id -> {fps, sync: {offset_s, status, source, agreement}, hip: (t_imu, x) or None,
    pose: (t_imu, body_pose63) or None}. Produces per-pair agreement and an overall status."""
    ids = sorted(cams)
    out = {"cameras": ids, "pairs": {}, "status": "PASS", "flags": []}
    if len(ids) < 2:
        out.update(status="N/A", flags=["single_camera"]); return out
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]; A, B = cams[a], cams[b]
            pair = {}
            # flash latency consistency: (flash - xcorr) should be the same system latency on both cameras
            fa = A["sync"].get("agreement", {}).get("flash_minus_xcorr_s"); fb = B["sync"].get("agreement", {}).get("flash_minus_xcorr_s")
            if fa is not None and fb is not None:
                pair["flash_latency_diff_s"] = float(fa - fb)
            tol = tol_frames / min(A["fps"] or 30.0, B["fps"] or 30.0)
            # main inter-view sync check: each view's own placement of IMU t=0 (from its accepted fragments)
            oa, ob = A.get("view_offset_s"), B.get("view_offset_s")
            if oa is not None and ob is not None:
                pair["view_offset_diff_s"] = float(oa - ob); pair["view_offsets_s"] = {a: float(oa), b: float(ob)}
                if abs(oa - ob) > tol:
                    out["flags"].append(f"{a}-{b}: the views place IMU t=0 {oa - ob:+.3f}s apart (tolerance {tol:.3f}s)"); out["status"] = "WARN"
            if A.get("hip") and B.get("hip"):
                # local phase agreement of the 2D pelvis signals (search limited to +-0.3 s: less than a step period, so no stride alias)
                pair["hip_vertical"] = signal_agreement(*A["hip"], *B["hip"], max_lag_s=0.3, label="hip_vertical_2d")
                if pair["hip_vertical"].get("r_best") is not None and pair["hip_vertical"]["r_best"] < 0.2:
                    out["flags"].append(f"{a}-{b}: 2D pelvis signals agree poorly (r {pair['hip_vertical']['r_best']:.2f})"); out["status"] = "WARN"
            if A.get("pose") and B.get("pose"):
                pair["pose"] = pose_agreement(*A["pose"], *B["pose"])
                if pair["pose"].get("ok"):
                    rl = pair["pose"]["knee_residual_lag"].get("residual_lag_s")
                    if rl is not None and abs(rl) > tol_frames / 30.0:
                        out["flags"].append(f"{a}-{b}: knee-angle residual lag {rl:+.3f}s"); out["status"] = "WARN"
                    if pair["pose"]["legs_median_deg"] > 25:
                        out["flags"].append(f"{a}-{b}: leg joints differ by {pair['pose']['legs_median_deg']:.0f} deg (median)"); out["status"] = "WARN"
            out["pairs"][f"{a}|{b}"] = pair
    return out
