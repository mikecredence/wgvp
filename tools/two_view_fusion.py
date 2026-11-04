"""Two compromised views, one 3D pose: how much of the walker's true pose does each view recover, and does fusing
the two recover more? Runs on the synthetic session HC901 (sample_v2/, rendered with the walker's ground-truth SMPL
pose saved) after the pipeline has lifted both views with GVHMR.

    python tools/two_view_fusion.py <data root with wgvp_output/HC901> [--gt sample_v2/HC901_gt.npz] [--out docs/two_view_fusion]

Per view, per frame and per joint: the geodesic angle between GVHMR's joint rotation and the truth (SMPL-X body
joints 1-21 are the SMPL joints 1-21). Frames are classed by what the camera could see of the walker (from the
ground truth: head out of frame, shoulders out, hips out). Fusion rules, evaluated against the same truth:

  single view         each view alone (what a one-camera lab gets)
  mean                per-joint rotation average of the two views (chordal mean), equal weights
  confidence-weighted per-joint weights from what is available at run time: the view's ViTPose confidence for the
                      keypoints that joint moves, zero when the frame was bridged (visible = False)
  visibility-weighted per-joint weights 1 / 0 by whether the joint and the segment it moves are inside the view's
                      frame (here from the rendered truth's in-frame flags; in deployment the same test comes from
                      the 2D keypoints against the panel edges), averaged where both views see the joint
  segment rule        a joint only from the view(s) whose keypoints for it are confident and inside the panel,
                      averaged when both, nothing when neither; cross-faded over 15 frames at segment edges;
                      optionally each view's weights scaled by its share of the fused root (fuse_root) and the
                      result smoothed over 5 frames (smooth_rotations): this last variant is what the demo shows
  oracle pick         per joint, the view with the smaller error (an upper bound: needs the truth)

The fusion library lives in `wgvp.fusion` (shared with the merge stage, `wgvp.merge`); this tool only adds the
comparison with the truth. Root orientation is compared after removing the constant heading offset between each
GVHMR world frame and the render's (a single yaw per view, fitted on visible frames); body-joint rotations need no
alignment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from wgvp.reconcile import SMPLX_BODY_JOINTS      # noqa: E402
from wgvp.fusion import (ARMS, JOINT_KP, LEGS, SPINE, _moving_average, chordal_mean, export_mask, fit_yaw, fuse_root, geodesic,   # noqa: E402,F401
                         ramp_inside, smooth_rotations, step_stats, usable_joints_rt)
# SMPL joint ids used for the visibility classes of a frame (from the ground-truth in-frame flags)
SMPL_HEAD, SMPL_SHOULDERS, SMPL_HIPS, SMPL_ANKLES = 15, [16, 17], [1, 2], [7, 8]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("--session", default="HC901"); ap.add_argument("--recording", default="hc901_selfpace")
    ap.add_argument("--gt", default=str(PKG / "sample_v2" / "HC901_gt.npz")); ap.add_argument("--out", default=str(PKG / "docs" / "two_view_fusion"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    deliver = Path(a.root) / "wgvp_output" / a.session / "deliver" / a.recording
    gt = np.load(a.gt); views = sorted(p.name for p in deliver.iterdir() if (p / "smpl_global.npz").exists())
    assert len(views) == 2, f"need two lifted views under {deliver}, found {views}"
    Z = {v: np.load(deliver / v / "smpl_global.npz") for v in views}; K = {v: np.load(deliver / v / "kp2d_vitpose.npz") for v in views}
    # common frames (combined-video frame indices) between the truth and both views
    frames = np.array(sorted(set(gt["frame_idx"].tolist()) & set(Z[views[0]]["frame_idx"].tolist()) & set(Z[views[1]]["frame_idx"].tolist())))
    gi = {int(f): i for i, f in enumerate(gt["frame_idx"])}; sel_gt = [gi[int(f)] for f in frames]
    gt_body = gt["pose_aa"][sel_gt, 3:66].reshape(-1, 21, 3); gt_root = gt["pose_aa"][sel_gt, :3]
    est, root, w_conf, visible, in_frame, in_frame_rt = {}, {}, {}, {}, {}, {}
    for v in views:
        z, k = Z[v], K[v]; zi = {int(f): i for i, f in enumerate(z["frame_idx"])}; s = [zi[int(f)] for f in frames]
        est[v] = z["body_pose"][s].reshape(-1, 21, 3); root[v] = z["global_orient"][s]; visible[v] = z["visible"][s].astype(bool)
        conf = np.nan_to_num(k["kp2d"][s][:, :, 2], nan=0.0)                               # (T,17) ViTPose confidence (face points blanked -> 0)
        in_frame_rt[v] = usable_joints_rt(k["kp2d"][s], k["view_box"] if "view_box" in k.files else None, visible[v])
        jin = gt[f"{v}_joints_in"][sel_gt]                                                   # (T,24) truth: joint inside this view
        w = np.stack([conf[:, JOINT_KP[j]].mean(1) for j in range(21)], 1)                   # (T,21)
        w = w * visible[v][:, None]                                                          # bridged frames carry no weight
        w_conf[v] = w; in_frame[v] = jin
    # ---- visibility classes per view from the truth
    def vclass(jin):
        head = jin[:, SMPL_HEAD]; sh = jin[:, SMPL_SHOULDERS].all(1); hips = jin[:, SMPL_HIPS].all(1); ank = jin[:, SMPL_ANKLES].all(1)
        c = np.full(len(jin), "whole body", object); c[~head] = "head cut"; c[~sh] = "shoulders cut"; c[~hips] = "hips cut (legs only)"; c[~ank] = "feet cut"; c[~jin.any(1)] = "out of view"
        return c
    classes = {v: vclass(in_frame[v]) for v in views}
    # ---- errors
    err = {v: geodesic(est[v], gt_body) for v in views}                                       # (T,21)
    A, B = views
    # visibility weights: the SMPL-X body joint j is SMPL joint j+1; a joint's rotation is observed when the joint and
    # the child segment it moves are in frame (SMPL kinematic children)
    SMPL_CHILD = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 10: 10, 11: 11, 12: 15, 13: 16, 14: 17, 15: 15, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23}
    def vis_w(v):
        # 1 when the joint and the segment it moves are in frame, 0.1 when the person is in frame but that joint is
        # not (so a view that sees the joint outweighs one that does not by 10:1, and two views that both miss it
        # fall back to an even average), 0 when the frame was bridged
        jin = in_frame[v]
        w = np.stack([(jin[:, j + 1] & jin[:, SMPL_CHILD[j + 1]]).astype(float) for j in range(21)], 1)
        return np.maximum(w, 0.1) * visible[v][:, None]
    w_vis = {v: vis_w(v) for v in views}
    fused = {"mean": chordal_mean([est[A], est[B]], np.ones((2, len(frames), 21))),
             "confidence-weighted": chordal_mean([est[A], est[B]], np.stack([w_conf[A], w_conf[B]])),
             "visibility-weighted": chordal_mean([est[A], est[B]], np.stack([w_vis[A], w_vis[B]]))}
    def segment_rule(wa, wb, ramp_frames: int = 0, extra=None):
        """Per joint and frame: the view(s) whose flag is set, averaged when both; NaN (no output) when neither.
        With ramp_frames > 0 the weights cross-fade over that many frames at the edges of each usable segment;
        extra = (sa, sb) multiplies each view's weights by a per-frame factor (the view's share of the fused root)."""
        fa, fb = (ramp_inside(wa, ramp_frames), ramp_inside(wb, ramp_frames)) if ramp_frames else (wa.astype(float), wb.astype(float))
        if extra is not None:
            fa, fb = fa * extra[0][:, None], fb * extra[1][:, None]
        out = chordal_mean([est[A], est[B]], np.stack([fa, fb]))
        none = ~(wa | wb); out[none] = np.nan
        return out
    hard = {v: (w_vis[v] >= 1.0) for v in views}                                             # truth-based usable flags
    fused["segment rule (truth visibility)"] = segment_rule(hard[A], hard[B])
    fused["segment rule (run-time visibility)"] = segment_rule(in_frame_rt[A], in_frame_rt[B])
    fused["segment rule, cross-faded (run-time visibility)"] = segment_rule(in_frame_rt[A], in_frame_rt[B], ramp_frames=15)
    fused["segment rule, cross-faded + smoothed 5 fr"] = smooth_rotations(fused["segment rule, cross-faded (run-time visibility)"], 5)
    fused["segment rule, cross-faded + smoothed 9 fr"] = smooth_rotations(fused["segment rule, cross-faded (run-time visibility)"], 9)
    # the fused root and each view's share of it (fuse_root); the share also weights the view's body joints
    yaw = {v: fit_yaw(root[v], gt_root, visible[v]) for v in views}
    root_al = {v: (R.from_euler("y", yaw[v], degrees=True) * R.from_rotvec(root[v])).as_rotvec() for v in views}
    root_f, share = fuse_root([root_al[A], root_al[B]], [in_frame_rt[A], in_frame_rt[B]], [visible[A] | True, visible[B] | True], ramp=15)
    fused["segment rule, cross-faded x root share + smoothed 5 fr"] = smooth_rotations(segment_rule(in_frame_rt[A], in_frame_rt[B], ramp_frames=15, extra=(share[0], share[1])), 5)
    pick = np.where((err[A] <= err[B])[:, :, None], est[A], est[B]); fused["oracle pick"] = pick
    err.update({k: geodesic(v_, gt_body) for k, v_ in fused.items()})
    root_err = {v: geodesic(root_al[v], gt_root) for v in views}; root_err["fused"] = geodesic(root_f, gt_root)
    root_step = {k: step_stats(x[:, None, :]) for k, x in (list(root_al.items()) + [("fused", root_f)])}
    # root error decomposed (yaw / pitch / roll about the truth's axes) and the hip joints' error, both by what the
    # along camera saw: a lifter that pitches the root wrong has hips that compensate for it
    root_rpy = {k: (R.from_rotvec(gt_root).inv() * R.from_rotvec(x)).as_euler("YXZ", degrees=True) for k, x in (list(root_al.items()) + [("fused", root_f)])}
    root_by_class = {c: {k: {"yaw": float(np.median(e[classes[A] == c, 0])), "pitch": float(np.median(e[classes[A] == c, 1])), "roll": float(np.median(e[classes[A] == c, 2]))}
                         for k, e in root_rpy.items()} for c in sorted(set(classes[A]))}
    hips_by_class = {c: {k: float(np.nanmedian(err[k][classes[A] == c][:, :2])) for k in list(views) + ["segment rule, cross-faded + smoothed 5 fr", "segment rule, cross-faded x root share + smoothed 5 fr"]} for c in sorted(set(classes[A]))}
    # ---- knee flexion, the gait-relevant scalar
    def knee(aa): return np.degrees(np.linalg.norm(aa[:, [3, 4]], axis=2))
    kg = knee(gt_body); knee_rmse = {k: float(np.sqrt(np.nanmean((knee(x) - kg) ** 2))) for k, x in list(est.items()) + list(fused.items())}
    # ---- tables
    def row(e, m=None):
        m = np.ones(len(frames), bool) if m is None else m; em = e[m]; legs = em[:, LEGS]
        with np.errstate(all="ignore"):
            return {"n_frames": int(m.sum()), "legs": float(np.nanmedian(legs)), "arms": float(np.nanmedian(em[:, ARMS])), "spine": float(np.nanmedian(em[:, SPINE])), "all": float(np.nanmedian(em)),
                    "legs_p90": float(np.nanpercentile(legs, 90)), "frac_legs_over_20": float(np.nanmean(legs > 20)) if np.isfinite(legs).any() else float("nan"),
                    "coverage": float(np.isfinite(em).mean())}
    steps = {k: step_stats(x) for k, x in list(est.items()) + list(fused.items())}
    res = {"n_frames": int(len(frames)), "views": views, "yaw_offset_deg": yaw,
           "overall": {k: row(e) for k, e in err.items()}, "knee_flexion_rmse_deg": knee_rmse, "continuity": steps,
           "root_orientation_median_deg": {**{v: float(np.median(root_err[v][visible[v]])) for v in views}, "fused": float(np.median(root_err["fused"]))},
           "root_step_p99_deg": {k: v_["p99_deg"] for k, v_ in root_step.items()}, "root_share_mean": {v: float(share[i].mean()) for i, v in enumerate(views)},
           "root_error_signed_by_along_visibility": root_by_class, "hip_error_by_along_visibility": hips_by_class,
           "by_visibility": {v: {c: row(err[v], classes[v] == c) for c in sorted(set(classes[v]))} for v in views},
           "fused_by_along_visibility": {c: {k: row(err[k], classes[A] == c) for k in fused} for c in sorted(set(classes[A]))},
           "per_joint_median": {k: dict(zip(SMPLX_BODY_JOINTS, np.median(e, 0).round(2).tolist())) for k, e in err.items()}}
    (out / "two_view_fusion.json").write_text(json.dumps(res, indent=1))
    # ---- markdown
    L = [f"# Two compromised views, one pose: HC901\n", f"{len(frames)} frames with the truth and both views lifted. Errors are geodesic angles (deg) between GVHMR's joint rotation and the rendered truth; medians over frames.\n",
         "| estimate | output coverage | legs | legs p90 | legs > 20 deg | arms | spine | all 21 | knee-flexion RMSE | frame step p99 | steps > 20 deg |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in list(views) + list(fused):
        r = res["overall"][k]; st = steps[k]
        L.append(f"| {k} | {r['coverage']:.0%} | {r['legs']:.1f} | {r['legs_p90']:.1f} | {r['frac_legs_over_20']:.0%} | {r['arms']:.1f} | {r['spine']:.1f} | {r['all']:.1f} | {knee_rmse[k]:.1f} | {st['p99_deg']:.1f} | {st['frac_over_20']:.2%} |")
    L.append("\nErrors of the segment rules are over the joints they emit; coverage is the share of (frame, joint) cells with an output. Both views and the other fusions emit every cell. "
             "Continuity: the largest per-joint rotation step between consecutive emitted frames, its 99th percentile and the share of steps over 20 deg.")
    rm = res["root_orientation_median_deg"]; rs = res["root_step_p99_deg"]
    L.append(f"\nRoot orientation after a constant heading alignment of {yaw[A]:+.0f} deg ({A}) and {yaw[B]:+.0f} deg ({B}): median error {rm[A]:.1f} / {rm[B]:.1f} deg "
             f"(frame step p99 {rs[A]:.1f} / {rs[B]:.1f}); the fused root (weighted by each view's ramped presence and the square of its usable fraction, sharpened towards the heavier view when the two disagree) "
             f"{rm['fused']:.1f} deg (step p99 {rs['fused']:.1f}); mean share of the {A} view {res['root_share_mean'][A]:.2f}.\n")
    L += ["\nRoot error decomposed (signed medians about the truth's axes, yaw / pitch / roll) and the hip joints' error, by what the along camera saw:\n",
          "| along camera saw | n | " + " | ".join(f"root {k}" for k in root_rpy) + f" | hips {A} | hips {B} | hips merge (no share) | hips merge (root share) |", "|---|---|" + "---|" * (len(root_rpy) + 4)]
    for c in root_by_class:
        n_c = int((classes[A] == c).sum()); hb = hips_by_class[c]
        L.append(f"| {c} | {n_c} | " + " | ".join(f"{root_by_class[c][k]['yaw']:+.0f} / {root_by_class[c][k]['pitch']:+.0f} / {root_by_class[c][k]['roll']:+.0f}" for k in root_rpy)
                 + f" | {hb[A]:.1f} | {hb[B]:.1f} | {hb['segment rule, cross-faded + smoothed 5 fr']:.1f} | {hb['segment rule, cross-faded x root share + smoothed 5 fr']:.1f} |")
    for v in views:
        L += [f"\n## {v} view by what the camera saw (truth)\n", "| frames | n | legs | arms | spine | all |", "|---|---|---|---|---|---|"]
        for c, r in res["by_visibility"][v].items():
            L.append(f"| {c} | {r['n_frames']} | {r['legs']:.1f} | {r['arms']:.1f} | {r['spine']:.1f} | {r['all']:.1f} |")
    fk = list(fused)
    L += [f"\n## Fused estimates by what the {A} view saw (legs / all 21; segment rules: over emitted joints, with coverage)\n", "| frames | n | " + " | ".join(fk) + " |", "|---|---|" + "---|" * len(fk)]
    for c, d in res["fused_by_along_visibility"].items():
        cells = [f"{d[k]['legs']:.1f} / {d[k]['all']:.1f}" + (f" ({d[k]['coverage']:.0%})" if k.startswith("segment") else "") for k in fk]
        L.append(f"| {c} | {d[fk[0]]['n_frames']} | " + " | ".join(cells) + " |")
    (out / "two_view_fusion.md").write_text("\n".join(L) + "\n")
    # ---- figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={"height_ratios": [1, 1.2]})
        x = np.arange(21); wdt = 0.16
        names = list(views) + ["mean", "segment rule (run-time visibility)", "segment rule, cross-faded x root share + smoothed 5 fr", "oracle pick"]; wdt = 0.8 / len(names)
        for i, k in enumerate(names):
            ax[0].bar(x + (i - len(names) / 2 + 0.5) * wdt, np.nanmedian(err[k], 0), wdt, label=k)
        ax[0].set_xticks(x); ax[0].set_xticklabels(SMPLX_BODY_JOINTS, rotation=60, ha="right", fontsize=8); ax[0].set_ylabel("median error vs truth (deg)"); ax[0].legend(fontsize=8, ncol=5)
        ax[0].set_title("A. Per-joint error of each view and of the fused estimates", loc="left", fontsize=10.5)
        t = gt["imu_t_s"][sel_gt]
        for k, c in ((A, "tab:blue"), (B, "tab:orange"), ("segment rule, cross-faded x root share + smoothed 5 fr", "tab:green")):
            with np.errstate(all="ignore"):
                ax[1].plot(t, np.nanmedian(err[k][:, LEGS], 1), color=c, lw=1.0 if k in views else 1.6, label=f"{k}: median leg-joint error" + ("" if k in views else " (gaps = no usable view)"))
        cl = classes[A]
        for lab, col in (("hips cut (legs only)", "0.75"), ("shoulders cut", "0.85"), ("head cut", "0.92")):
            m = cl == lab
            if m.any():
                ax[1].fill_between(t, 0, 60, where=m, color=col, step="mid", lw=0, label=f"{A} view: {lab}")
        for lab, col in (("out of view", "mistyrose"),):
            m = classes[B] == lab
            if m.any():
                ax[1].fill_between(t, 0, 60, where=m, color=col, step="mid", lw=0, alpha=0.8, label=f"{B} view: {lab}")
        ax[1].set_ylim(0, 60); ax[1].set_xlabel("IMU time (s)"); ax[1].set_ylabel("leg-joint error (deg)"); ax[1].legend(fontsize=7.5, ncol=3, loc="upper right")
        ax[1].set_title("B. Leg-joint error over the walk, with what each camera could see", loc="left", fontsize=10.5)
        plt.tight_layout(); plt.savefig(out / "two_view_fusion.png", dpi=130); plt.close(fig)
    except Exception as e:   # matplotlib is optional
        print("figure skipped:", e)
    if not a.quiet:
        print("\n".join(L)); print(f"\n-> {out}")
    return res


if __name__ == "__main__":
    main()
