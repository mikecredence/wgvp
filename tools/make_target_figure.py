"""One figure of the training target a recording produces (`merged_smpl.npz`), for the README:

    python tools/make_target_figure.py <deliver/<recording>> [--out docs/figures/merged_target.png] [--gt sample_v2/HC901_gt.npz]

A. per-joint loss weight over IMU time (21 rows), with the export mask and each view's usable fraction above it;
B. the heading yaw and the heading-relative root tilt; C. root speed in the heading frame (forward, lateral) and,
when a ground truth is given, the truth's forward speed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from wgvp.fusion import heading_yaw_deg      # noqa: E402
from wgvp.reconcile import SMPLX_BODY_JOINTS  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec"); ap.add_argument("--out", default=str(PKG / "docs" / "figures" / "merged_target.png")); ap.add_argument("--gt", default=None)
    a = ap.parse_args(argv)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    m = np.load(Path(a.rec) / "merged_smpl.npz"); t = m["imu_t_s"]; views = [str(v) for v in m["views"]]; col = {0: "tab:blue", 1: "tab:orange"}
    fig, ax = plt.subplots(4, 1, figsize=(12, 9.6), gridspec_kw={"height_ratios": [0.9, 2.2, 1.1, 1.1]}, sharex=True)
    # A0: export mask and usable fractions
    frac = m["usable"].mean(2)
    for i, v in enumerate(views):
        ax[0].plot(t, frac[i], color=col.get(i, "k"), lw=1.2, label=f"{v} view: fraction of joints it can vouch for")
    ax[0].fill_between(t, 0, 1, where=m["export_ok"], color="0.88", step="mid", lw=0, label="frames the export keeps")
    ax[0].set_ylim(0, 1.02); ax[0].set_ylabel("usable"); ax[0].legend(fontsize=7.5, ncol=3, loc="lower right"); ax[0].set_title("A. What each view can vouch for, the export mask, and the per-joint loss weight", loc="left", fontsize=10.5)
    # A1: weights heatmap
    W = m["weight"].T; im = ax[1].imshow(W, aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=[t[0], t[-1], 20.5, -0.5], interpolation="nearest")
    ax[1].set_yticks(range(21)); ax[1].set_yticklabels(SMPLX_BODY_JOINTS, fontsize=7); ax[1].set_ylabel("joint")
    cb = fig.colorbar(im, ax=ax[1], pad=0.01, fraction=0.025); cb.set_label("weight", fontsize=8)
    # B: heading yaw and heading-relative tilt
    yaw = m["heading_yaw_deg"]; hr = m["global_orient_heading_rel"]; ok = np.isfinite(hr).all(1)
    tilt = np.full((len(t), 3), np.nan); tilt[ok] = R.from_rotvec(hr[ok]).as_euler("YXZ", degrees=True)
    ax[2].plot(t, yaw, color="k", lw=1.0, label="heading yaw (deg, reference view's world, unwrapped)")
    ax2 = ax[2].twinx(); ax2.plot(t, tilt[:, 1], color="tab:green", lw=0.9, label="heading-relative pitch"); ax2.plot(t, tilt[:, 2], color="tab:purple", lw=0.9, label="heading-relative roll")
    ax2.set_ylim(-30, 30); ax2.set_ylabel("tilt (deg)", fontsize=8); ax[2].set_ylabel("yaw (deg)")
    h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[2].legend(h1 + h2, l1 + l2, fontsize=7.5, ncol=3, loc="upper right")
    ax[2].set_title("B. Root for a sensor model: the heading (what the room knows) and the tilt (what the body can know)", loc="left", fontsize=10.5)
    # C: speed in the heading frame
    v = m["root_vel_heading_rel_m_s"]; rw = m["root_weight"]
    ax[3].plot(t, v[:, 2], color="tab:red", lw=1.0, label="forward speed, merged (m/s)"); ax[3].plot(t, v[:, 0], color="tab:gray", lw=0.8, label="lateral speed, merged")
    if a.gt:
        gt = np.load(a.gt); gy = heading_yaw_deg(gt["pose_aa"][:, :3]); gv = np.gradient(gt["transl"], gt["imu_t_s"], axis=0)
        gvh = R.from_euler("y", -gy, degrees=True).apply(gv); ax[3].plot(gt["imu_t_s"], gvh[:, 2], color="k", lw=0.8, ls="--", label="forward speed, rendered truth")
    ax[3].fill_between(t, -0.5, 2.5, where=rw < 0.3, color="mistyrose", step="mid", lw=0, label="root weight < 0.3")
    ax[3].set_ylim(-0.5, 2.5); ax[3].set_ylabel("m/s"); ax[3].set_xlabel("IMU time (s)"); ax[3].legend(fontsize=7.5, ncol=4, loc="upper right")
    ax[3].set_title("C. Root velocity in the heading frame (translation from a single far camera is the weakest field: see the text)", loc="left", fontsize=10.5)
    plt.tight_layout(); Path(a.out).parent.mkdir(parents=True, exist_ok=True); plt.savefig(a.out, dpi=130); plt.close(fig); print("->", a.out)


if __name__ == "__main__":
    main()
