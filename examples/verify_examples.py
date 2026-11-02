"""Check a pipeline run on the example corpus against the ground truth written by fetch_examples.py.

    python examples/verify_examples.py [--data examples/data]

Per example: the sync offset must be within 2 frames of the planted one, the chosen subject track must sit on the
intended walker's hips (>= 80 % of the shared frames within 36 px), and the QC status is reported. Exit code 1 if
any example fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(HERE / "data")); a = ap.parse_args()
    root = Path(a.data); ok_all = True; rows = []
    for exp_path in sorted(root.glob("EX*/expected.json")):
        exp = json.loads(exp_path.read_text()); sid, rid = exp["session_id"], exp["recording_id"]
        deliver = root / "wgvp_output" / sid / "deliver"
        ss_path = deliver / "session_summary.json"
        if not ss_path.exists():
            rows.append((sid, "NO RUN", "", "", "", f"missing {ss_path.relative_to(root)}")); ok_all = False; continue
        ss = json.loads(ss_path.read_text()); ri = ss["items"].get(rid, {}); it = ss["items"].get(f"{rid}/full", {})
        off = ri.get("offset_s"); tol = 2.0 / exp["fps"]
        off_ok = off is not None and abs(off - exp["offset_s"]) <= tol
        st = deliver / rid / "full" / "subject_track.npz"; frac, n = 0.0, 0
        if st.exists():
            z = np.load(st); lut = {int(f): np.array(h) for f, h in zip(exp["gt_frames"], exp["gt_hip_px"])}
            errs = [np.hypot(*(kp[11:13, :2].mean(0) - lut[int(f)])) for f, kp in zip(z["frame_idx"], z["kp2d"]) if int(f) in lut and np.isfinite(kp[11:13, :2]).all()]
            n = len(errs); frac = float(np.mean(np.array(errs) < 36)) if n else 0.0
        subj_ok = n >= 30 and frac >= 0.8
        ok = off_ok and subj_ok; ok_all &= ok
        rows.append((sid, "PASS" if ok else "FAIL", f"{off if off is None else round(off, 3)} vs {exp['offset_s']}", f"{frac:.0%} of {n}",
                     f"{it.get('selection_source')} r={it.get('r_fine') if it.get('r_fine') is None else round(it['r_fine'], 2)} margin={it.get('margin') if it.get('margin') is None else round(it['margin'], 2)}",
                     f"QC {it.get('status')} {it.get('flags') or ''}"))
    w = [max(len(str(r[i])) for r in rows + [("example", "verify", "offset (found vs planted)", "subject frames on target", "selection", "pipeline QC")]) for i in range(6)]
    hdr = ("example", "verify", "offset (found vs planted)", "subject frames on target", "selection", "pipeline QC")
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    if not rows:
        print(f"no examples under {root} (run fetch_examples.py first)"); return 1
    print("\nALL PASS" if ok_all else "\nSOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
