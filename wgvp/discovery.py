"""Walk the partner's folder tree (subject folder > task sub-folder > one combined video; recording CSVs under the
subject folder) and list every session, recording, video, label export and everything that did not match."""
from __future__ import annotations

import re
from pathlib import Path


def _iter_files(folder: Path, max_depth: int, ignore: set):
    def walk(d: Path, lvl: int):
        try:
            entries = sorted(d.iterdir())
        except PermissionError:
            return
        for p in entries:
            if p.is_dir():
                if p.name in ignore or lvl + 1 > max_depth:
                    continue
                yield from walk(p, lvl + 1)
            else:
                yield p
    yield from walk(folder, 0)


def _task_family_of(name: str, parent: str, cfg: dict) -> str | None:
    tasks = {k: v for k, v in cfg["tasks"].items() if not k.startswith("_")}
    low = name.lower(); par = parent.lower()
    for fam, t in tasks.items():
        toks = sorted([x.lower() for x in t.get("video_tokens", [])], key=len, reverse=True)
        for tok in toks:
            if re.search(rf"(^|[^a-z0-9]){re.escape(tok)}([^a-z0-9]|$)", low):
                return fam
    for fam, t in tasks.items():
        if par in [x.lower() for x in t.get("dir_names", [])]:
            return fam
    return None


def _csv_family(task: str, cfg: dict) -> str | None:
    for fam, t in cfg["tasks"].items():
        if fam.startswith("_"):
            continue
        if task.lower() in [x.lower() for x in t.get("csv_tokens", [])]:
            return fam
    return None


def scan_root(root: Path, cfg: dict) -> dict:
    root = Path(root).resolve(); ignore = set(cfg["ignore_dirs"])
    rx_subj = re.compile(cfg["subject_folder"]["regex"], re.IGNORECASE)
    rx_csv = re.compile(cfg["recording_csv"]["regex"], re.IGNORECASE)
    rx_vid = re.compile(cfg["video"]["regex"], re.IGNORECASE)
    exts = set(e.lower() for e in cfg["video"]["extensions"])
    sessions, unmatched, seen = {}, [], set()
    # subject folders: any folder (not ignored) whose name matches; nested matches use the deepest
    subj_dirs = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        rel = d.relative_to(root).parts
        if any(x in ignore for x in rel):
            continue
        if rx_subj.match(d.name) and not any(rx_subj.match(x) for x in rel[:-1]):
            subj_dirs.append(d)
    if rx_subj.match(root.name) and not subj_dirs:
        subj_dirs = [root]
    for sd in subj_dirs:
        sid = "__".join(sd.relative_to(root).parts) or sd.name
        subject = rx_subj.match(sd.name).group("subject").strip()
        s = {"session_id": sid, "subject": subject, "dir": str(sd), "recordings": [], "videos": [], "labels": [], "sheet": None, "flags": []}
        for p in _iter_files(sd, max(cfg["recording_csv"]["max_depth"], cfg["video"]["max_depth"]), ignore):
            seen.add(p)
            suf = p.suffix.lower()
            if suf == ".csv" and p.parent != sd and False:
                pass
            if suf == ".csv":
                m = rx_csv.match(p.name)
                if m and _csv_family(m.group("task"), cfg):
                    s["recordings"].append({"recording_id": p.stem, "subject": m.group("subject").strip(), "task": m.group("task").lower(),
                                            "family": _csv_family(m.group("task"), cfg), "csv": str(p), "videos": [], "sheet_start_frames": {}})
                    continue
                if any(Path(p.name).match(g) for g in cfg["labels"]["glob"]):
                    s["labels"].append(str(p)); continue
                if Path(p.name).match(cfg["sheet"]["glob"]):
                    s["sheet"] = str(p); continue
                if m:
                    unmatched.append({"path": str(p.relative_to(root)), "reason": f"task '{m.group('task')}' is not in a configured task family"})
                else:
                    unmatched.append({"path": str(p.relative_to(root)), "reason": "csv name does not match recording_csv.regex"})
            elif suf == ".mat" and any(Path(p.name).match(g) for g in cfg["labels"]["glob"]):
                s["labels"].append(str(p))
            elif suf in exts:
                fam = _task_family_of(p.stem, p.parent.name, cfg)
                mv = rx_vid.match(p.stem)
                if fam is None:
                    unmatched.append({"path": str(p.relative_to(root)), "reason": "video: no task token in the file name and no task sub-folder name (tasks.*.video_tokens / dir_names)"}); continue
                s["videos"].append({"video_id": p.stem, "path": str(p), "family": fam, "tags": (mv.group("tags") if mv and mv.group("tags") else ""), "folder": p.parent.name})
        # pair: a video with every recording of the same family (the sync stage picks the recording a video actually holds)
        for rec in s["recordings"]:
            rec["videos"] = [v for v in s["videos"] if v["family"] == rec["family"]]
            if not rec["videos"]:
                rec["flag"] = "no video for this recording's task family"
            elif len(rec["videos"]) > 1:
                rec["flag"] = f"{len(rec['videos'])} candidate videos (the sync stage keeps the one holding this recording)"
        for v in s["videos"]:
            if not any(v in r["videos"] for r in s["recordings"]):
                unmatched.append({"path": str(Path(v["path"]).relative_to(root)), "reason": f"video in {sid}: no recording CSV of family '{v['family']}' in the subject folder"})
        if not s["recordings"] and not s["videos"]:
            continue
        if not s["recordings"]:
            s["flags"].append("no recording CSV found in the subject folder")
        sessions[sid] = s
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts and p not in seen and not any(x in ignore for x in p.relative_to(root).parts[:-1]):
            unmatched.append({"path": str(p.relative_to(root)), "reason": "video not inside a subject folder (subject_folder.regex)"})
    n_rec = sum(len(s["recordings"]) for s in sessions.values()); n_pairs = sum(len(r["videos"]) for s in sessions.values() for r in s["recordings"])
    return {"root": str(root), "n_sessions": len(sessions), "n_recordings": n_rec, "n_pairs": n_pairs, "n_videos": sum(len(s["videos"]) for s in sessions.values()),
            "sessions": sorted(sessions.values(), key=lambda s: s["session_id"]), "unmatched": unmatched}


def manifest_text(man: dict) -> str:
    lines = [f"root: {man['root']}", f"sessions: {man['n_sessions']}  recordings: {man['n_recordings']}  videos: {man['n_videos']}  recording-video pairs: {man['n_pairs']}"]
    for s in man["sessions"]:
        lines.append(f"  [{s['session_id']}] subject {s['subject']}  videos {len(s['videos'])}  labels {len(s['labels'])}  sheet {Path(s['sheet']).name if s['sheet'] else '-'}" + (f"  <- {'; '.join(s['flags'])}" if s["flags"] else ""))
        for r in s["recordings"]:
            vids = ", ".join(Path(v["path"]).name for v in r["videos"]) or "NO VIDEO"
            lines.append(f"      {r['recording_id']} ({r['family']}): {vids}" + (f"  <- {r['flag']}" if r.get("flag") else ""))
    if man["unmatched"]:
        lines.append(f"unmatched ({len(man['unmatched'])}):")
        lines += [f"  - {u['path']}: {u['reason']}" for u in man["unmatched"][:200]]
    return "\n".join(lines)
