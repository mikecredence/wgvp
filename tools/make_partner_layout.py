"""Lay the synthetic sessions out the way a data partner might hand over a study (one folder per subject, several
tasks per subject), to test the folder scan and run the whole pipeline on more than one subject and task:

    <dest>/Sub 001/Video/SP_combo.mp4, TUG_combo.mp4 + Sub 001/IMU/Sub001_selfpace.csv, Sub001_tug.csv   (HC900 + HC903)
    <dest>/Sub 002/Video/SP_combo.mp4 ...                                                                  (HC901, the low along camera)
    <dest>/Sub 003/Video/SP_combo.mp4, HP_combo.mp4 ...                                                    (HC902 + HC904, hurried pace)
    <dest>/Sub 004/Video/SPM_combo.mp4 + Sub 004/IMU/Sub004_selfpace_mat.csv                             (HC905, on the mat: the walkway's cells in the CSV)

    python tools/make_partner_layout.py <dest> [--default-camera]
    python -m wgvp run <dest> --cpu-only          # then the GPU stage, `wgvp merge <dest>`, `wgvp demo <dest> --session "Sub 002" --gt sample_v2/HC901_gt.npz`

The default matching rules find all three (the subject from the folder name, the task from the SP token in the
file name, the CSV from <subject>_<task>.csv up to three levels down).

The layout's `wgvp_config.json` carries the sample cameras' field of view (`gvhmr.focal_35mm`), as a site that knows
its cameras would set it: the along view is rendered with a 60 deg vertical field of view on a 519 x 600 panel (28 mm
as a 35 mm equivalent), the side view with 43 deg on 901 x 376 (21 mm). `--default-camera` leaves the option empty,
and the lifter then assumes its default camera in both views.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from wgvp.config import DEFAULT_CONFIG, config_path  # noqa: E402

# 35 mm equivalent of each sample view: f_px * 43.27 / panel diagonal, with f_px = (H / 2) / tan(yfov / 2) from make_sample.py
SAMPLE_FOCAL_35MM = {"left": 28, "right": 21}
# (subject folder, sample session dir, session id, task dir / video token / csv token, operator pick)
SESSIONS = [("Sub 001", PKG / "sample" / "data" / "HC900", "HC900", ("SP", "SP", "selfpace"), None),
            ("Sub 001", PKG / "sample_v4" / "data" / "HC903", "HC903", ("TUG", "TUG", "tug"), None),                       # the same walker's Timed Up and Go
            ("Sub 002", PKG / "sample_v2" / "data" / "HC901", "HC901", ("SP", "SP", "selfpace"), PKG / "sample_v2" / "data" / "wgvp_output" / "HC901" / "config" / "pick_left_hc901_selfpace.json"),
            ("Sub 003", PKG / "sample_v3" / "data" / "HC902", "HC902", ("SP", "SP", "selfpace"), None),
            ("Sub 003", PKG / "sample_v5" / "data" / "HC904", "HC904", ("HP", "HP", "hurriedpace"), None),                 # the same walker at a hurried pace
            ("Sub 004", PKG / "sample_v6" / "data" / "HC905", "HC905", ("SPM", "SPM", "selfpace_mat"), None)]              # a walk on the instrumented mat (the walkway's cells in the CSV)


def main(dest: str, default_camera: bool = False) -> int:
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    if not default_camera and not config_path(dest).exists():
        cfg = json.loads(json.dumps(DEFAULT_CONFIG)); cfg["gvhmr"]["focal_35mm"] = dict(SAMPLE_FOCAL_35MM)
        config_path(dest).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        print(f"wgvp_config.json: the sample cameras' field of view (gvhmr.focal_35mm {SAMPLE_FOCAL_35MM})")
    for sub, src, sid, (tdir, ttok, tcsv), pick in SESSIONS:
        if not src.exists():
            print(f"skipping {sub}/{sid}: {src} is not here"); continue
        rid = f"{sub.replace(' ', '')}_{tcsv}"
        (dest / sub / "Video").mkdir(parents=True, exist_ok=True); (dest / sub / "IMU").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / tdir / f"{sid}_{ttok}_DeID.mp4", dest / sub / "Video" / f"{ttok}_combo.mp4")
        shutil.copy2(src / f"{sid.lower()}_{tcsv}.csv", dest / sub / "IMU" / f"{rid}.csv")
        if pick is not None and pick.exists():
            d = json.loads(pick.read_text(encoding="utf-8")); d["recording_id"] = rid
            cdir = dest / "wgvp_output" / sub / "config"; cdir.mkdir(parents=True, exist_ok=True)
            (cdir / f"pick_{d['camera_id']}_{rid}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(f"{sub}/{ttok}_combo.mp4 <- {sid}" + (" (+ operator pick)" if pick is not None and pick.exists() else ""))
    print(f"-> {dest}: run `python -m wgvp scan \"{dest}\"` then `python -m wgvp run \"{dest}\" --cpu-only`")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--default-camera"]
    sys.exit(main(args[0] if args else "partner_layout", default_camera="--default-camera" in sys.argv[1:]))
