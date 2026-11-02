"""Editable matching rules and run parameters: ONE JSON file (`wgvp_config.json` at the data root, written by the
first scan). Every rule the folder scan uses is here, so a different naming convention is a config edit."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CONFIG = {
    "_doc": "Matching rules for the folder scan and run parameters. Edit, then re-scan (GUI 'Scan' or `wgvp scan <root>`). Regexes are Python re, case-insensitive.",
    "subject_folder": {
        "regex": r"^(?P<subject>[A-Za-z]{2,4}\s*\d{2,4}[^/\\]*)$",
        "_doc": "A session = a subject folder (e.g. NLS036, HC100, 'nls207 (control)'); the scan looks at every folder whose name matches. Recording CSVs and task sub-folders are searched inside it."
    },
    "tasks": {
        "selfpace": {"csv_tokens": ["selfpace", "selfpace_mat", "selfpace_matturn"], "dir_names": ["SP", "selfpace", "self-paced", "self_paced", "selfpaced", "self paced"], "video_tokens": ["SP", "selfpace", "self-paced", "self_paced"]},
        "_doc": "Task families to process. csv_tokens: the <task> part of <subject>_<task>.csv recording files that belong to the family; dir_names: sub-folder names holding that task's video(s); video_tokens: tokens in a video file name that mark the task. Add e.g. \"hurriedpace\": {...} to extend."
    },
    "recording_csv": {
        "regex": r"^(?P<subject>[^_]+)_(?P<task>[A-Za-z]+(?:_[A-Za-z]+)*)\.csv$",
        "max_depth": 3,
        "_doc": "Recording CSV name = <subject>_<task>.csv (WearGait export), searched up to max_depth levels below the subject folder."
    },
    "video": {
        "extensions": [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mts", ".wmv"],
        "regex": r"^(?P<subject>[^_]+)_(?P<task>[A-Za-z]+)(?:_(?P<tags>.*))?$",
        "max_depth": 3,
        "_doc": "Combined two-view video files, e.g. NLS036_SP_DeID_90.mp4, anywhere up to max_depth below the subject folder; the task is taken from the file name token or from the sub-folder name (tasks.*.dir_names). One video is paired with every recording CSV of the same task family of the subject (the sync stage then tells which recording the video holds). 'tags' is kept for the QC only."
    },
    "views": {
        "layout": "auto", "n_sample_frames": 40, "black_level": 28, "min_panel_frac": 0.2,
        "override": {},
        "_doc": "layout: auto | horizontal | vertical | single. override: per video file name, explicit panel boxes, e.g. {\"NLS036_SP_DeID_90.mp4\": {\"left\": [0, 0, 760, 600], \"right\": [760, 0, 1420, 500]}} (x0, y0, x1, y1 pixels). The scan writes a preview PNG with the detected boxes next to the QC summary."
    },
    "labels": {"glob": ["*label*.csv", "*Label*.csv", "*labels*.mat", "*gTruth*.mat"], "_doc": "Optional Video Labeler scene-label export (video time) used as a sync cross-check; see README for the MATLAB export line."},
    "sheet": {"glob": "*annot*.csv", "columns": {"recording": ["recording", "file", "trial", "task"], "camera": ["camera", "video", "cam"], "start_frame": ["video_start_frame", "start_frame", "video_start", "frame_start", "sync_frame"]},
              "_doc": "Optional annotation sheet (CSV) with a video start frame per recording, used as a sync cross-check."},
    "ignore_dirs": ["wgvp_output", ".git", "__pycache__", "work", "deliver"],
    "params": {
        "track_max_width": 960, "gvhmr_max_width": 1280, "max_chunk_frames": 3000, "chunk_overlap_frames": 60,
        "margin_s": 1.0, "tol_frames": 3.0, "fps_downsample_above": 45.0, "pose_conf": 0.25,
        "_doc": "track_max_width: each panel is downscaled to this width for the 2D tracker (keypoints reported in original pixels). max_chunk_frames: lifter input chunk size (RAM bound). tol_frames: agreement tolerance between sync sources."
    },
    "selection": {"accept_r": 0.30, "min_excess": 0.15, "lag_tol_s": 0.12, "min_margin": 0.12, "min_r_primary": 0.35, "app_attach": 0.35,
                  "_doc": "accept_r: pelvis correlation needed to accept a track fragment as the subject; min_r_primary/min_margin: confident selection; app_attach: appearance distance for attaching brief re-entries."},
    "gvhmr": {"root": "", "python": "", "timeout_s": 7200,
              "_doc": "root: the GVHMR checkout with inputs/checkpoints; python: the interpreter of the GVHMR environment. Leave empty to run the CPU stages only; the GPU stage can be run later (`wgvp gvhmr <root>` or the GUI with the fields filled)."}
}


def config_path(root: Path) -> Path:
    return Path(root) / "wgvp_config.json"


def load_config(root: Path | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if root is not None and config_path(root).exists():
        _deep_update(cfg, json.loads(config_path(root).read_text()))
    return cfg


def write_default_config(root: Path, overwrite: bool = False) -> Path:
    p = config_path(root)
    if p.exists() and not overwrite:
        return p
    p.write_text(json.dumps(DEFAULT_CONFIG, indent=1)); return p


def _deep_update(base: dict, upd: dict):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k != "override":
            _deep_update(base[k], v)
        else:
            base[k] = v
