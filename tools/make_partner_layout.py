"""Lay the three synthetic sessions out the way a data partner might hand over a study, to test the folder scan and
run the whole pipeline on more than one subject:

    <dest>/Sub 001/Video/SP_combo.mp4 + Sub 001/IMU/Sub001_selfpace.csv     (HC900, the dry-run session)
    <dest>/Sub 002/...                                                        (HC901, the two compromised views; its operator pick is copied along)
    <dest>/Sub 003/...                                                        (HC902, the third room)

    python tools/make_partner_layout.py <dest>
    python -m wgvp run <dest> --cpu-only          # then the GPU stage, `wgvp merge <dest>`, `wgvp demo <dest> --session "Sub 002" --gt sample_v2/HC901_gt.npz`

The default matching rules find all three (the subject from the folder name, the task from the SP token in the
file name, the CSV from <subject>_<task>.csv up to three levels down).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SESSIONS = [("Sub 001", PKG / "sample" / "data" / "HC900", "HC900", None),
            ("Sub 002", PKG / "sample_v2" / "data" / "HC901", "HC901", PKG / "sample_v2" / "data" / "wgvp_output" / "HC901" / "config" / "pick_left_hc901_selfpace.json"),
            ("Sub 003", PKG / "sample_v3" / "data" / "HC902", "HC902", None)]


def main(dest: str) -> int:
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    for sub, src, sid, pick in SESSIONS:
        if not src.exists():
            print(f"skipping {sub}: {src} is not here"); continue
        rid = f"{sub.replace(' ', '')}_selfpace"
        (dest / sub / "Video").mkdir(parents=True, exist_ok=True); (dest / sub / "IMU").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "SP" / f"{sid}_SP_DeID.mp4", dest / sub / "Video" / "SP_combo.mp4")
        shutil.copy2(src / f"{sid.lower()}_selfpace.csv", dest / sub / "IMU" / f"{rid}.csv")
        if pick is not None and pick.exists():
            d = json.loads(pick.read_text()); d["recording_id"] = rid
            cdir = dest / "wgvp_output" / sub / "config"; cdir.mkdir(parents=True, exist_ok=True)
            (cdir / f"pick_{d['camera_id']}_{rid}.json").write_text(json.dumps(d, indent=1))
        print(f"{sub} <- {sid}" + (" (+ operator pick)" if pick is not None else ""))
    print(f"-> {dest}: run `python -m wgvp scan \"{dest}\"` then `python -m wgvp run \"{dest}\" --cpu-only`")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "partner_layout"))
