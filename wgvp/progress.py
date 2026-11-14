"""The share of a run that is done, as one number for a progress bar, from the events the pipeline reports."""
from __future__ import annotations

TRACK_SHARE = 0.92            # of a session's CPU stages: finding people frame by frame (0.86-0.97 on the bundled examples)
CPU_SHARE_WITH_LIFT = 0.30    # of a run that includes the lift: the CPU stages
WRAP_SHARE = 0.03             # the QC summary at the end


class RunProgress:
    """Feed it every progress event of one run. `fraction` (0..1) never falls; `text` says what is happening now.

    An event is (stage, i, n, msg):
      "plan"     i = 1 when the lift (GPU stage) will run, n = sessions to run, msg = their sizes ("3.1,4.6", any unit)
      "root"     session i of n begins, msg = its name
      "video"    video i of n of that session begins, msg = the file name
      "frames"   i of n frames of that video have been searched for people
      "session"  recording i of n of that session begins, msg = "<session>: <recording>"
      "lift"     i of n lifter pieces are done, msg = the piece being lifted
      "wrap"     the QC summary is being written
    """

    def __init__(self):
        self.fraction = 0.0; self.text = "starting ..."
        self.lift = False; self.n = 1; self.si = 0; self.sid = ""; self.vi = 0; self.nv = 1; self.video = ""; self.w = [1.0]

    def _cpu_span(self) -> float:
        return CPU_SHARE_WITH_LIFT if self.lift else 1.0 - WRAP_SHARE

    def _session(self, f: float) -> float:
        w = self.w if len(self.w) == self.n else [1.0] * self.n               # each session's slice of the bar follows its size
        return self._cpu_span() * (sum(w[:self.si]) + w[min(self.si, self.n - 1)] * min(max(f, 0.0), 1.0)) / sum(w)

    def update(self, stage: str, i: int, n: int, msg: str = "") -> tuple[float, str]:
        head = f"session {self.si + 1} of {self.n}: {self.sid}"
        x = self.fraction
        if stage == "plan":
            self.lift = bool(i); self.n = max(int(n), 1); x = 0.0
            try:
                self.w = [max(float(v), 1e-9) for v in str(msg).split(",")] if msg else [1.0] * self.n
            except ValueError:
                self.w = [1.0] * self.n
        elif stage == "root":
            self.si, self.n, self.sid, self.vi, self.nv = int(i), max(int(n), 1), str(msg), 0, 1
            head = f"session {self.si + 1} of {self.n}: {self.sid}"
            x = self._session(0.0); self.text = head
        elif stage == "video":
            self.vi, self.nv, self.video = int(i), max(int(n), 1), str(msg)
            x = self._session(TRACK_SHARE * self.vi / self.nv); self.text = f"{head}, reading {self.video}"
        elif stage == "frames":
            x = self._session(TRACK_SHARE * (self.vi + i / max(n, 1)) / self.nv)
            self.text = f"{head}, finding people in {self.video}: frame {i} of {n}"
        elif stage == "session":
            x = self._session(TRACK_SHARE + (1.0 - TRACK_SHARE) * i / max(n, 1))
            rid = str(msg).split(": ", 1)[-1]
            self.text = f"{head}, matching recording {i + 1} of {n} ({rid}) to the video"
        elif stage == "lift":
            span = 1.0 - WRAP_SHARE - CPU_SHARE_WITH_LIFT
            x = CPU_SHARE_WITH_LIFT + span * i / max(n, 1)
            self.text = f"lifting to 3D on the GPU: {i} of {n} piece(s) done" + (f", now {msg}" if msg else "")
        elif stage == "wrap":
            x = 1.0 - WRAP_SHARE; self.text = "writing the QC summary"
        self.fraction = min(max(self.fraction, x), 1.0)
        return self.fraction, self.text


class RunStopped(Exception):
    """The run was asked to stop (the GUI's Stop button): raised at the next check, caught by run_root."""


def check_stop(stop) -> None:
    """stop is None or a callable with no arguments; raise RunStopped when it says the run should end."""
    if stop is not None and stop():
        raise RunStopped("stopped by the user")
