# wgvp - WearGait video-to-pose pipeline

[![tests](https://github.com/mikecredence/wgvp/actions/workflows/ci.yml/badge.svg)](https://github.com/mikecredence/wgvp/actions/workflows/ci.yml)

Gait-laboratory video usually shows more than the participant: staff at a desk, a clinician walking alongside,
people passing in the background. The participant is the one wearing the inertial sensors. wgvp uses that fact to
turn a two-camera session video plus its sensor recording into per-frame 3D body poses of the participant alone.

It runs entirely on the data owner's own computer, needs no network, and ships only **pose-level derivatives**
(SMPL-X body parameters, 2D skeleton keypoints without face points, bounding boxes, sync offsets, QC numbers).
No video frame leaves the site. It was built for the [WearGait-PD](https://doi.org/10.7303/syn52470130) session
videos (two camera views tiled into one frame beside the IMU/walkway CSV export), and the folder-matching rules,
sensor column names and task names are all in one editable JSON file, so other labs' layouts are a configuration
change.

![tracking](docs/figures/tracking.gif)

*Both camera views of the bundled synthetic session (rendered bodies, no real people). Green is the person the
pipeline chose from the pelvis accelerometer; the co-walker, the seated operator and the bystander stay grey.*

## What it does

For each session video (one file, two views tiled side by side or stacked, unequal sizes and letterboxing are fine):

1. **Finds the two view panels** in the frame (black borders plus the persistent content seam).
2. **Tracks every person in each view** with a 2D pose tracker (CPU).
3. **Picks the subject**: the person whose pelvis up-and-down motion matches the pelvis accelerometer in the
   recording CSV. Seated or standing people do not match however large they are in frame; the two views are matched
   jointly; the subject's exits from a view are recorded as gaps instead of being filled with a bystander.
4. **Syncs** the video to the recording timeline with the same cross-correlation (the video frame of IMU t = 0, with
   a confidence), cross-checked against a sync light if one is visible, a Video Labeler label export if present, and
   an annotation-sheet start frame if present.
5. **Lifts the subject to SMPL-X body parameters** per frame and per view with [GVHMR](https://github.com/zju3dv/GVHMR)
   (the only GPU step), driving the lifter with the subject's own bounding boxes so it follows the right person.
6. **Reconciles the two views** on the IMU timeline, writes a QC record per recording and view, skeleton-only
   thumbnails, and one QC summary page across all sessions.
7. **Merges the two views into one pose per frame** (`merged_smpl.npz`): only the joints some view saw, cross-faded
   at the hand-overs, with per-joint loss weights, a heading-relative root and one body shape per recording.
8. **Fits one body to both views at once** (`jointfit_smpl.npz`, CPU, a few seconds per recording) and names the
   training target in `target.json`: the fit where there are two views and its checks pass, the merge everywhere
   else. That file is what a sensor-to-pose model is trained on, and nothing from the body sensors enters it: the
   recording CSV serves the sync and the subject identity, never a joint angle.

![two views](docs/figures/two_views_tracking.png)

*Two frames of the synthetic session with the detected panels (green and orange boxes), every tracked person and,
beside each, that person's pelvis correlation with the IMU at the shared lag. The subject reaches 0.78 in the
along view and 0.63 in the side view; the co-walker stays between 0.04 and 0.31, and nobody else passes 0.38.*

## Demos and results

Each picture below plays a 14-s excerpt of one recording's demo video in a loop and links to the full video in
[docs/demos/](docs/demos/), 1 to 2 MB each. GitHub does not play a video file kept in a repository: use *View raw*
on the page the link opens, or play the file from a clone. A demo shows the two views with the lifter's 2D input,
the reconstruction, the exported training target, the path from above, a filmstrip and knee-flexion timelines.
Every one below was rendered by `python -m wgvp demo` from the pipeline's own outputs, with the target
`target.json` names, which is the joint two-view fit in all seven.

**Six rendered sessions with a known truth.** Rendered rooms laid out like the clinic's, one folder per subject
(HC900 to HC905). The truth stands beside the reconstruction in these, and the leg and root errors of each frame
are printed.

| | |
|:---:|:---:|
| [![HC900 demo](docs/demos/HC900_preview.webp)](docs/demos/HC900_demo.mp4) | [![HC901 demo](docs/demos/HC901_preview.webp)](docs/demos/HC901_demo.mp4) |
| HC900, self-paced walk | HC901, self-paced walk, along camera close |
| [![HC902 demo](docs/demos/HC902_preview.webp)](docs/demos/HC902_demo.mp4) | [![HC903 demo](docs/demos/HC903_preview.webp)](docs/demos/HC903_demo.mp4) |
| HC902, self-paced walk with a co-walker | HC903, timed up and go |
| [![HC904 demo](docs/demos/HC904_preview.webp)](docs/demos/HC904_demo.mp4) | [![HC905 demo](docs/demos/HC905_preview.webp)](docs/demos/HC905_demo.mp4) |
| HC904, hurried pace | HC905, walk on the pressure mat |

The training target against the rendered truth, on the exported frames:

| session | task | frames exported | thigh direction (deg) | knee angle RMSE (deg) | path (m) | speed RMSE (m/s) |
|---|---|---|---|---|---|---|
| HC900 | self-paced walk | 100 % | 3.7 | 4.8 | 0.05 | 0.072 |
| HC901 | self-paced walk | 100 % | 3.2 | 4.7 | 0.04 | 0.082 |
| HC902 | self-paced walk, co-walker | 100 % | 3.9 | 5.5 | 0.04 | 0.049 |
| HC903 | timed up and go | 99.9 % | 6.5 | 7.8 | 0.10 | 0.070 |
| HC904 | hurried pace | 99.8 % | 5.2 | 5.8 | 0.06 | 0.063 |
| HC905 | walk on the mat | 99.8 % | 3.4 | 5.8 | 0.02 | 0.050 |
| median | | | 3.8 | 5.6 | 0.05 | 0.067 |
| median, merge | | | 6.0 | 6.2 | 0.20 | 0.169 |

The last row is the second estimate, the merge of the two single-view lifts, on the same frames. The path is its
median distance from the truth after one rigid fit. The root's rotation is not in the table because the fit does
not improve it: 9.8 deg at the median against the merge's 9.4. The whole table, with the arms, the ankles and the
heading, is under [One body fitted to both views](#one-body-fitted-to-both-views-jointfit_smplnpz), and what the
scores mean under [How good these scores are](#how-good-these-scores-are-and-what-would-improve-them).

**One real two-camera recording (OWN03).** One of us walking back and forth between two chalk lines 13.5 ft apart,
a tablet to the side and a phone at one end. It has no truth for the pose; the chalk lines give the distance
walked. The clip ships at half size in `examples/two_camera/` and runs offline (Quick start, step 2).

[![OWN03 demo](docs/demos/OWN03_preview.webp)](docs/demos/OWN03_demo.mp4)

| on the published clip | merge | joint fit (the target) |
|---|---|---|
| frames exported | 100 % | 100 % |
| distance walked over the truth, in body heights | 0.96 | 1.04 |
| distance walked over the truth, in metres as delivered | 1.07 | 1.13 |
| path off the walk's axis (m, rms) | 0.17 | 0.04 |

Both estimates take their scale from the body the lifter guessed, which is 5 to 8 % taller than the walker, so the
metres read long and the body heights do not. [On one real recording](#one-body-fitted-to-both-views-jointfit_smplnpz)
has the method and the full-size numbers.

## Quick start

Python 3.10-3.12 on Windows, macOS or Linux. Everything below runs on the CPU.

```bash
git clone https://github.com/mikecredence/wgvp.git
cd wgvp
python -m venv wgvp-env
source wgvp-env/bin/activate        # Windows: wgvp-env\Scripts\activate
pip install -r requirements.txt
```

That is the whole CPU install (numpy, scipy, OpenCV, CPU PyTorch, ultralytics, a bundled ffmpeg). The tracker
weights (`weights/yolov8n-pose.pt`, 6 MB) and a synthetic test session (`sample/`, 3 MB) are in the repository.
`pip install -e .` additionally gives you a `wgvp` command equal to `python -m wgvp`.

**1. Dry run (1-6 min on a laptop CPU; about 70 s on an Apple M-series).** The bundled session has a known answer;
every check must pass:

```bash
python -m wgvp dry-run --cpu-only
```

```
  [PASS] two panels detected at the right boxes
  [PASS] sync offset within 2 frames of truth (pelvis cross-correlation): estimated 3.008 s vs true 3.0 s ...
  [PASS] left: chosen track is the rendered subject: 100% of 1458 frames hold the GT hip inside the chosen box ...
  [PASS] right: the chosen track holds the subject wherever the tracker found it: 100% of the 1455 frames ...
  ...
DRY RUN PASS (10/10 checks)
```

**2. Real footage, offline (1 min).** `examples/bundled/` ships a 13-s phone clip of one of us walking down an
office corridor and back, plus a synthetic two-view file made from it, each with a fabricated sensor recording built
from the walker's own motion. No download, no real IMU:

```bash
python -m wgvp run examples/bundled --cpu-only
python examples/verify_examples.py --data examples/bundled
```

`examples/two_camera/` is the same check on a real two-camera recording: 31 s of one of us walking back and forth
between two chalk lines, a tablet's view and a phone's view tiled into one file (6 MB), again with a fabricated
sensor recording. It takes about a minute more:

```bash
python -m wgvp run examples/two_camera --cpu-only
python examples/verify_examples.py --data examples/two_camera
```

The verify script prints `PASS` per session when the sync and the chosen subject match the known answers. The
pipeline's own status for the two-view sessions is `WARN`, and that is expected here: these clips have no sync
light, and the optional cross-check that looks for one reports that what it found does not agree
(`flash_disagrees`). [examples/README.md](examples/README.md) has the measured numbers.

**3. Example corpus on more footage.** Three public YouTube clips (a corridor walk test, a shuttle-walk test with
two standing staff, a crowded plaza) are fetched to your machine, cut, and given the same kind of fabricated
recording. [examples/README.md](examples/README.md) says what each one shows and what to expect.

```bash
pip install yt-dlp
python examples/fetch_examples.py          # add --cookies-from-browser chrome if YouTube asks for a sign-in
python -m wgvp run examples/data --cpu-only
python examples/verify_examples.py
```

**4. Your own data.** Point the pipeline at the top-level folder that holds the subject folders:

```bash
python -m wgvp scan /path/to/data_root       # what was found, what was not, and why
python -m wgvp run  /path/to/data_root --cpu-only
```

or open the GUI (`python -m wgvp`): choose the data folder, **Scan**, **Dry run**, **Run all sessions**,
**Open QC summary**. After a run each session's row in the list shows its PASS, WARN or FAIL with the reason. The
GUI needs tkinter, which the python.org installers include; on Debian/Ubuntu `sudo apt install python3-tk`, with a
Homebrew Python on macOS `brew install python-tk@3.12` (match your Python version). Without tkinter
`python -m wgvp` prints these fixes and the equivalent commands in place of the window.

**5. The GPU step** (GVHMR, NVIDIA GPU with 8 GB or more, CUDA 12) lives in its own Python environment:
[env/install_gvhmr.md](env/install_gvhmr.md). Once it exists, `python -m wgvp check-gpu --gvhmr-root <GVHMR>
--gvhmr-python <that env's python>` lists every missing piece with its fix, and `python -m wgvp run <root>` (without
`--cpu-only`, with `gvhmr.root` / `gvhmr.python` set in the config or passed as flags) runs the lift after the CPU
stages. The CPU stages can be run now and the GPU stage later (`python -m wgvp gvhmr <root>`). The merge runs by
itself after the lift; `python -m wgvp merge <root>` rebuilds it from existing lifts.

**6. The demo video** of a lifted recording (the two views with the lifter's 2D input, the reconstruction, the
export, an aerial view, a filmstrip and knee-flexion timelines; the figure further down is one frame of it) renders
from the pipeline's outputs with `pip install -e .[demo]` and the SMPL-X body model from the GVHMR install:

```bash
python -m wgvp demo /path/to/data_root --session "Sub 001"     # -> wgvp_output/<session>/work/demo/<recording>_demo.mp4
```

It contains video frames, so it is written under `work/` and is not part of the deliverable.

## Expected input

```
<data root>/
  <subject>/                        e.g. NLS036   (folder name pattern: subject_folder.regex in wgvp_config.json)
     <subject>_selfpace.csv         recording CSVs = the WearGait IMU/walkway export (Time, GeneralEvent, LowerBack_FreeAcc_*, ...)
     <subject>_selfpace_mat.csv
     SP/<subject>_SP_<tags>.mp4     one combined two-view video per task (task from the file-name token or the folder name)
     (optional) *label*.csv         Video Labeler scene labels exported in video time (sync cross-check)
     (optional) *annot*.csv         annotation sheet with a video start frame (sync cross-check)
```

A simpler layout with one folder per subject, the videos in a `Video` sub-folder and the recording exports in an
`IMU` sub-folder matches the default rules as well, and is the layout the multi-subject runs below used:

```
<data root>/
  Sub 001/
     Video/SP_combo.mp4             the task comes from the SP token in the file name
     Video/TUG_combo.mp4            another task of the same subject
     IMU/Sub001_selfpace.csv        the recording exports; the pipeline needs them for the sync and the subject identity
     IMU/Sub001_tug.csv
  Sub 002/ ...
```

`python tools/make_partner_layout.py <dest>` builds this layout from the bundled synthetic sessions (self-paced,
hurried-pace, TUG and a walk on the instrumented mat). It also writes a `wgvp_config.json` there that gives the
lifter the two sample cameras' field of view (`gvhmr.focal_35mm`, described below); `--default-camera` leaves that
out.

Three task families are processed by default: self-paced (`selfpace`, `selfpace_mat`, `selfpace_matturn`; video
tokens `SP`, `SPM`, `SPMT`), hurried-pace (`hurriedpace`, `hurriedpace_mat`; `HP`) and the Timed Up and Go (`tug`;
`TUG`). Add others (tandem gait, balance) in `wgvp_config.json` under `tasks`; a config written by an older version
keeps its own token lists, so add `SPM` there by hand if a mat video goes unmatched. A video is paired with every recording CSV of its task family for that
subject and the sync stage reports which recording the video actually holds, so one long video per visit and one
video per trial both work. Single-view videos work too (`"views": {"layout": "single"}`).

The first `scan` writes `wgvp_config.json` into the data root with every matching rule (subject-folder pattern,
task names, CSV and video name patterns, panel boxes per video, selection thresholds, GVHMR paths) and the run
parameters. Edit it and scan again; no code changes are needed for a different naming convention. One parameter
worth knowing: `params.detector_imgsz` (default 640) is the pose detector's input size; 960 keeps a person who
shrinks to under a tenth of the frame height (far end of a corridor, portrait phone video) detected, at about twice
the CPU cost. The bundled example needs it; a clinic walkway seen from its end may too.

Two more parameters change what the lifter is given. `gvhmr.focal_35mm` is each camera's real field of view, as a
35 mm equivalent focal length: one number for every view, or one per view (`{"left": 28, "right": 21}`). Left empty,
GVHMR assumes a focal length equal to the picture's diagonal (a 53 deg diagonal field of view), which is too long
for a wide room camera and turns the walker's heading with their position in the picture (measured under
"How good these scores are" below). For a panel of W by H pixels whose camera has a horizontal field of view
`hfov`, the focal length in pixels is `f_px = (W / 2) / tan(hfov / 2)` and the number to enter is
`f_px * 43.27 / sqrt(W^2 + H^2)`; from a diagonal field of view `dfov` it is `21.63 / tan(dfov / 2)`. If a panel is
a crop of the camera's picture and not a resize of it, the field of view is the crop's. The value used is recorded
in each view's `gvhmr_meta.json`.

When the lens is not known, `python tools/estimate_focal.py <video> [--box x0 y0 x1 y1] [--out folder]` reads the
number from the room. It finds the straight edges in nine frames, or in those at `--times` (walls, floor, door frames),
sorts them into the three directions a room's edges run in, and solves for the focal length at which those three
directions stand at right angles. It needs a camera that sees two of the three directions at a slant. A camera
square to a wall, or looking straight down a walkway's grid, leaves the focal length free, and the tool then prints
that the room's lines do not fix it and returns no number. Its checks so far:

| picture | known | read |
|---|---|---|
| rendered side camera, four subjects | 21.2 mm | 20.9, 21.1, 21.1, 21.4 |
| rendered along camera, square to the floor grid, four subjects | 28.3 mm | refused, four of four |
| bundled phone clip against its own two zoomed panels | 1.171 and 1.858 times the clip's | 17.3, then 20.2 and 32.2 (predicted 20.2 and 32.1) |

`--out` writes each frame with its edges coloured by direction, which is the thing to look at before trusting a
number. `--box` is the panel's box in a tiled video (`views` in the scan report).

`params.rest_hold` (default true) keeps the subject's box through detector
dropouts while the IMU events say they stand or sit (see "Partial bodies, exits and re-entries").

Label export for the optional cross-check, in MATLAB with the labeler session open or the `gTruth` object loaded:

```matlab
T = timetable2table(gTruth.LabelData); writetable(T, 'NLS036_SP_labels.csv');
```

## How the subject is found and synced

Every tracked person gets a *pelvis signal*: the hip row relative to the camera's horizon row, divided by the
torso length, which removes the perspective term that walking toward a low camera leaves in the raw hip row. The
IMU side is the pelvis accelerometer, band-passed to the gait band (0.5-5 Hz) and integrated twice, so both sides
are pelvis displacement. A coarse match on the activity envelope (the stand / walk / turn pattern) finds the
candidate lag, a fine match inside +-0.75 s gives the correlation-versus-lag curve per person, and since the two
views are one file with one unknown lag, the lag is chosen jointly: the one that the most fragments across both
views support. The same lag is the sync offset, to sub-frame precision.

![selection and sync](docs/figures/selection_and_sync.png)

*A: inside a walk bout the subject's video pelvis signal in each view (green) follows the accelerometer (black);
the co-walker's (red) does not. B: correlation versus lag for every scored person; both views' subjects peak at
the shared lag of 3.01 s (true 3.00 s). C: coverage per view on the IMU timeline; both views hold the subject over
the whole recording, the side view short of four frames in all.*

Correlations are judged by their **excess over an empirical chance level**, not by their raw value. Short fragments
reach high correlations by chance (a 4-s fragment reaches r = 0.73 over a full-range lag search; a 45-s one 0.30),
so the acceptance rule is r >= 0.30 *and* excess >= 0.15, and a "high" confidence further needs r >= 0.35, excess
>= 0.20 and a margin >= 0.12 over the best other person at that lag. On crowded public footage with a simulated
reference, subjects visible for 8 s or more were found in every tested case among 300-400 other walkers, 4-8-s
fragments about half the time (otherwise mostly declared low-confidence), with 11 ms median offset error
([docs/design.md](docs/design.md), section 2.4). A clinic recording gives 20-45 s of the subject per view with a
stand / walk / turn pattern, which is the easy regime. Two sanity rules stand behind the confidence, added after a
rendered walker who paced within a metre with a 5-mm pelvis bob produced a "high"-confidence lag 30 s off
(section 2.8 of the design notes): the offset must leave at least 60 % of the recording inside the video, and a
strong persistent light-like brightness step more than 3 s from the offset is raised as a disagreement even though
it was not used. Either turns the sync into a WARN with the reason in `sync_flags`. When such a light contradicts
the pelvis match's winner, the lag candidate at the light is taken instead (a co-walker's gait, or the subject's own
walk one bout later, can correlate above chance at a wrong lag; the light is the evidence that breaks that), and
that lag is then refined on the correlation curve only within a tenth of a second of the light: the pelvis bob
repeats every step, so the wider refinement used for the match's own lag slid onto the neighbouring step's peak,
0.54 s off, on one rendered session. Without a light, a chosen lag whose runner-up carries at least 80 % of its
support is reported with low confidence and `ambiguous_lag` (on the rendered self-paced walks the subject's own
fragments correlate at a bout's shift, 6.5 s, at r 0.5-0.8; a right lag's runner-up carried 0-26 % of its
support, the wrong ones' 88-91 %).

## Where tracking goes wrong, and what the pipeline does about it

Multi-person tracking in a room fails in known ways. Each one is handled, and each leaves a trace in the QC so you
can see it happened.

| Failure | What the pipeline does | Where you see it |
|---|---|---|
| The tracker swaps identities when two people cross or pass close | Tracker ids are split into **fragments** at box jumps, time gaps and persistent appearance changes beside a second tracked person; fragments are judged one by one and re-joined only when they agree on the shared lag | `n_fragments`, `subject_tracks` (the accepted fragment ids) |
| An identity switch inside one fragment (the subject and a co-walker crossing) | A **lag-consistency mask**: the fragment's pelvis signal is correlated with the IMU in 10-s windows at the shared lag; windows below r = 0.25 are dropped | `consistency` in `selection_joint.json` (kept fraction per fragment) |
| The subject leaves a view and someone else is in frame | Exits are recorded as **gaps**; nothing is attached while the subject is out, and a scorable fragment that contradicts the lag is never attached by appearance. Absences over 2 s end the lifter chunk; shorter ones are bridged and flagged `visible = False` | `n_gaps`, `present_frac`, `gaps_imu_time_s`; the dry run asserts "no bystander attached"; `chunk_id` / `visible` per frame |
| Brief re-entries too short to score | Attached by torso-colour appearance only when clearly closer to the subject than to any concurrent person | `attached_by_appearance` |
| The subject's torso is outside the picture (legs only, close to a camera) | No colour descriptor is taken from such a box; the fragment is attached by **continuity**, when at its ends it continues the subject's own box | `attached_by_continuity` |
| Two detection boxes on one person (partial bodies in the along-walkway view) | Recognised by box overlap and not counted as a rival | role `duplicate_of_subject` in the fragment table |
| Someone walking in step with the subject (a clinician alongside) | The pelvis match is sub-frame; a person 0.8 s behind correlates 0.03-0.17 at the shared lag. This is the one configuration where the margin gets thin | `margin`, `lag_ambiguity` (> 0.85 means another lag was almost as well supported) |
| A seated operator large in frame, a bystander at the far wall | No gait-band pelvis oscillation, so they score about 0 whatever their size | their `r_at_lag` in the fragment table |
| Short spurious fragments chaining up at a wrong lag with "high" confidence | Every correlation is judged by its excess over the chance level for its length and the number of lags searched; support = excess x overlap, so one long genuine match outweighs a chain of short ones | `excess`, `second_best_excess` |
| The lifter follows the largest or longest person instead of the subject | GVHMR is driven with the subject's own bounding-box track (`preprocess/bbx.pt`), cut per view and per visible stretch | `gvhmr_frames` versus `present_frac` |
| A confident but wrong lag (a subject who barely moves, short fragments whose own lags disagree) | The offset must leave >= 60 % of the recording inside the video, and a strong light-like step far from it is raised; the QC shows the runner-up lags (`lag_candidates`) and an operator pick re-anchors the choice | `sync_flags`, `lag_ambiguity`, `lag_candidates` |
| The two views disagree after all this | Reconciliation on the IMU timeline: residual lag of the 2D pelvis signals (always) and per-joint disagreement of the SMPL-X poses (after GVHMR), tolerance 3 frames | `reconcile_status`, `residual_lag_s`, `legs_median_deg` |
| The automatic choice is wrong or absent | Two operator tools: `set-roi` (draw the walkway polygon once per view) and `pick-subject` (click the subject on a skeleton-only frame); both are reported with their agreement with the automatic choice | `selection_source` = `walkway_roi` / `operator_pick`, status WARN |
| Variable frame rate, dropped frames | All timing is done on presentation timestamps, never on frame counts | `timing` in the video record (gaps, VFR flag) |
| The walker is small in a room-wide side view (a hundred pixels tall): the vertical pelvis bob is under a pixel, below the keypoint noise, and no fragment of that view correlates with the IMU at any lag | Once the other view's subject is confirmed (by the pelvis match, or by a lag the light validated), that view's fragments are tested by their **progress along the walk** against the confirmed subject at zero lag on the shared video clock (the hip column where the walk runs across the picture, the feet's row or the body's width where it runs into it; |r| >= 0.8 over >= 5 s; windows that stop following while the confirmed subject moves are dropped, windows in which it stands still are no evidence and kept); a fragment that holds only the standing walker, with no progress to correlate, is attached by continuity when it ends where an attached fragment begins (or begins where one ends), within 2 s and half a box height, a frame or two in common allowed and no other fragment fitting the junction; someone walking in step correlates nearly as well at this time scale and is flagged, not resolved | `selection_source` `cross_view`, the candidates' correlations and `attached_by_continuity` in `selection.json` |
| A second person's gait correlates with the recording above the chance model at a wrong lag (a co-walker's 15-s walk reached r 0.78 at a lag 4 s off) | The sync light is used as evidence, not only as a check: when a strong light-like step contradicts the pelvis match's winner, the lag candidate at the light is taken (added if no fragment proposed it) and the fragments are re-accepted there | `sync.json` and the recording's flags say the lag was chosen by the light |
| One camera's picture is mirrored (a webcam's "mirror preview" left on): its left leg is the walker's right | The merge compares the knees of the two views in the swing phase as lifted and with the sides swapped (as lifted 11-16 deg vs swapped 21-40 on the rendered sessions; 43 vs 7 on a mirrored panel of a real clip) and reflects the mirrored view back before fusing | `view_mirrored` in `merged_smpl.npz`, `mirror_check` and a flag in `merge.json` |
| One view's lift takes a 180-degree turn the wrong way round (the monocular front/back ambiguity): the two headings end up opposite until that lift snaps back | The merged root stays with the view that turns like a person: while the views disagree, a view turning faster than 150 deg/s or far from the previous fused root is discounted, and, reading the whole recording, a view whose root turns more than 90 deg within a quarter of a second during a disagreement episode, or whose coverage ends or begins inside one while the other view continues, carries nothing over that episode (the snap is its own correction; a departure leaves the disagreement unresolved) | `merge.json` `continuity.root_flips_over_90deg`, a flag when any remain; the demo's heading trace |

Two of these rules were learned the hard way while testing (the perspective term in the pelvis signal, and the
chance level of short fragments); [docs/design.md](docs/design.md) section 2.6 has the numbers before and after.

## The GPU step and the two-view agreement

GVHMR lifts each view independently; the pipeline only hands it the subject's cropped frames and boxes, converts
its output to NPZ on the IMU timeline, and then compares the two views, which share nothing but the sync.

![mesh](docs/figures/gvhmr_two_views_mesh.png)

![agreement](docs/figures/gvhmr_two_views_agreement.png)

*On the synthetic session, lifted on a Tesla T4: the two views' knee-flexion curves line up on the IMU timeline
(residual lag -61 ms, within the 3-frame tolerance) and the leg joints disagree by a median of 7.8 degrees. Body
shape differs between views, as expected from a per-frame estimate when one view truncates the body.*

GVHMR takes about 0.5 s per frame per view on a T4 (12-15 min for a 50-s view). Its weights and the SMPL / SMPL-X
body models are downloaded by you under their own licences (registration required for the body models; they are not
in this repository or the Docker image). `docker/` holds a CUDA 12.1 recipe pinned to the versions we ran; its build
is untested because our GPU box had no disk left for it.

## Partial bodies, exits and re-entries

A gait-lab camera at the end of the walkway cuts the head off when the participant is close, and a side camera
loses the participant beyond the ends of the mat. A lifter run naively on such footage either crumples the body
or jumps to whoever else is in frame. What the pipeline does, and what the synthetic session measured on a T4
(its first render, whose narrow side camera lost the walker at both ends of the mat; the shipped render has the
room-wide side camera described further down, and its numbers are in the sections that follow):

* **Who gets lifted.** GVHMR is driven with the subject's own box track, so an exit never becomes a jump to a
  bystander; the measured per-frame joint-rotation step in the lifted output is 1-2 deg median, and the two frames
  of the side view that step more than 30 deg (one bridged, one near the end of the recording; 39 deg at most) are
  flagged as such. Body shape is constant within a chunk.
* **Exits longer than `absence_split_s` (2 s)** end the lifter chunk; a new chunk starts at the re-entry
  (`chunk_id` changes, and `transl` is only continuous within a chunk, since the world frame is re-anchored by each
  GVHMR run). Nothing is lifted in between: on the sample's side view the two exits (3.4 s and 2.1 s) become gaps
  in the output (three chunks of 885, 340 and 127 frames), and GVHMR's first second after each re-entry disagrees
  with the other view by 10.4 and 12.9 deg against 5.3 deg in steady state, a warm-up and no jump. Shorter gaps (a
  brief occlusion, a lost detection) are bridged with interpolated boxes so GVHMR keeps its temporal context, and
  those frames are flagged `visible = False`. The flag matters: bridged frames of a true exit disagree with the
  other view by 16.5 deg median (p90 38), whereas the 90 bridged frames of the along view, where the tracker merely
  lost a person who was still in frame, agree at 5.0 deg. Drop them, or weigh them by that flag.
* **Head or feet cut off** is handled by GVHMR itself: on the 777 along-view frames with the head out of frame, the
  leg joints agree with the side view at a median 4.4 deg (p90 5.7 deg), closer than on the frames where that view
  sees the whole body from far away (9.5 deg), and nothing collapses. The
  cost is a per-view body-shape bias (`betas` differ between a truncating and a non-truncating view), which is why
  shape should be taken from the view that sees the whole body. Such frames carry `edge_truncated = True`.
* **A subject at rest whom the detector loses.** A person standing small and still in a room-wide view can go
  undetected for seconds, and that view then has no box to lift from. With `params.rest_hold` (default on), inside
  the intervals the recording's events mark as standing or sitting, a frame without a detection keeps the subject's
  box: interpolated between two detections whose boxes overlap (IoU 0.8 or more), or the nearest detection's box
  before the first and after the last. Those frames are lifted with the rest, and the merge uses one only if the
  lifter's 2D keypoints, taken inside the held box, find the shoulders, hips, knees and ankles where the nearest
  detected frame has them (median confidence 0.5 or more, displaced by at most 5 % of the box height). On the six
  rendered recordings 409 held frames are accepted, 232 and 164 of them the standing starts of two side views, and
  two frames of the TUG are refused. Against the truth their legs are 11.8 deg off at the median, as good as the
  rest frames the detector did find (11.4).
* **Per-frame quality, so nobody has to infer it:** `visible`, `edge_truncated`, `rest_held`, `pose_step_deg`
  (largest joint step from the previous frame) and `chunk_id` in every SMPL NPZ; `pose_agreement_<a>_<b>.npz` beside
  `reconcile.json` with the per-joint disagreement between the two views on every IMU frame; and a QC flag when any
  frame moves a joint by more than 30 deg in one frame.

The synthetic sessions are real three-dimensional scenes, not composites: `tools/make_sample.py` places the
walker (an SMPL body driven by a public WearGait inertial fit), the co-walker, the seated operator, the desk, the
bystander, the mat, the floor tiles and the walls in one world, puts two perspective cameras where the aerial panel
of the demo draws them (their poses and intrinsics are saved in the truth file as `<view>_cam_pose` and
`<view>_intrinsics`), and renders each camera's picture with pyrender, an OpenGL rasteriser. The per-view truth
(`<view>_joints2d`, the in-frame flags) is the projection of the same joints through the same cameras, so a view's
truncation, a camera's height and tilt and the panel sizes are what the geometry gives, not what was drawn by
hand. The bodies wear flat colours, a top and trousers different for each person: a uniformly pale body standing
small and still in the side view was missed by the person detector, and a room of identical bodies does not exercise
the torso-colour descriptor. What the renders still lack is appearance (fabric, lighting, lens distortion, motion
blur), which a game engine would add and which matters for the 2D detectors rather than for the geometry; the
real-footage examples carry that test.

These numbers come from rendered bodies. GVHMR on real clothed, truncated people is on the list of things only the
first real run can show.

### 3D reconstruction from two partial views

The real session layout is harsher than the dry-run sample: the along-walkway camera sits at the end of the mat,
so the participant fills and overflows the frame when close, and the side camera's framing cuts the head. To ask
whether two such views still give back one correct 3D pose, `sample_v2/` renders the same walk from an along view
that keeps only the legs when the walker is near (hips out of frame half of the time) and saves the walker's true
SMPL pose and 3D joints beside it. The experiment below was run on its first render, whose side view was aimed at
hip height (head above the frame in 77 % of frames); the shipped `sample_v2/` has been rendered twice since, first
with the room-wide side camera of every other sample and then with the corrected walking path and the dressed
bodies of the current samples, and its current numbers are under "The merge" and "How good these
scores are". `tools/two_view_fusion.py` measures, against that truth, each view's GVHMR error per joint and by what
the camera could see, and the error of three fused estimates (plain rotation mean, ViTPose-confidence-weighted
mean, and an oracle per-joint pick as the upper bound). The rotations GVHMR returns are camera-independent apart
from a constant heading offset between the two cameras' world frames, which the script fits and removes.

Two things this sample taught before any GPU was involved. First, with the camera 0.5 m from the end of the mat,
the automatic pelvis match cannot confirm the along view: close to the camera, the walker's forward-speed
oscillation moves the hip row more than the vertical bob does, a step-periodic signal with the wrong phase. The
side view confirms and synchronises the recording on its own (offset 2.995 s versus 3.000 planted, 100 % of
frames on the true hips), and the along view is taken from an operator pick, which is why picks can now be
given as a frame and a point and shipped with a data set (HC901 carries one). Second, the pose signal survives a
cut head: a truncated box's height follows the frame edge, but the gait-band content of the pelvis signal does not
depend on it.

Lifted on the T4 (CPU stages locally in the pinned environment, `wgvp gvhmr` alone on the box; 1395 frames with
the truth and both views lifted), the answer is measured rather than argued
([docs/two_view_fusion/two_view_fusion.md](docs/two_view_fusion/two_view_fusion.md)):

| estimate against the rendered truth | leg joints, median | legs > 20 deg | all 21 joints | knee-flexion RMSE |
|---|---|---|---|---|
| along view alone (hips out of frame 60 % of the time) | 14.9 deg | 34 % | 21.7 deg | 14.8 deg |
| side view alone (head cut 77 % of the time) | 10.3 deg | 11 % | 11.5 deg | 8.2 deg |
| fused, plain rotation mean | 12.7 deg | 27 % | 16.2 deg | 8.6 deg |
| fused, weighted by per-joint visibility | 9.6 deg | 9 % | 11.9 deg | 10.3 deg |
| **segment rule**: a joint only from the view(s) whose keypoints for it are in frame, averaged when both, nothing when neither (98 % of cells emitted) | 10.1 deg | 9 % | 13.1 deg | 8.6 deg |
| **segment rule, cross-faded** over the first and last 0.5 s of each usable segment | 10.1 deg | 9 % | 12.7 deg | 9.9 deg |
| **segment rule, cross-faded and smoothed** (5-frame rotation average within emitted segments) | 10.1 deg | 9 % | 12.7 deg | 9.9 deg |
| **segment rule, cross-faded, each view weighted by its share of the fused root, smoothed** (the demo's output) | 10.0 deg | 8 % | 11.4 deg | 7.4 deg |
| fused, oracle per-joint pick (needs the truth) | 9.4 deg | 5 % | 10.7 deg | 5.1 deg |

![two-view fusion](docs/two_view_fusion/two_view_fusion.png)

What this says. A cut head costs nothing: the side view's head-cut frames are at least as accurate as its
whole-body frames (8.5 versus 14.3 deg on the legs). The along view's leg joints hold at 14-17 deg whatever is cut,
down to the frames where only the legs remain (14.4 deg), but on those frames everything above the hips is
invented (34 deg over all joints) and, as the next paragraph shows, the root pitches forward by 50 deg. Two partial
views do give one 3D pose, and the rule that works is the plain one: cut each view into the segments where it can
actually see a joint, combine on the shared timeline, and output nothing where neither view saw it. Implemented
with a run-time test only (the ViTPose keypoints a joint moves must be confident and inside the panel), that rule
emits 98 % of joint-frames with the leg error and leg outliers slightly below the better view's (10.1 deg and 9 %
against 10.3 and 11 %). Hand-overs between views are where discontinuities would come from, since the two views
disagree by about 10 deg where both see a joint: cross-fading the weights over the first and last half-second of
each usable segment brings the merge's frame-to-frame steps down from a 99th percentile of 16.8 deg (0.79 % of
steps over 20 deg) to 5.5 deg (0.03 %), fewer than either view alone, at the same error. What remains is the
lifter's own frame-to-frame jitter inside a segment; a 5-frame moving average in rotation space, computed within
runs of emitted frames so that gaps stay gaps, takes the merge to 4.1 deg (no step over 20 deg) without changing
the error. Soft weighting by ViTPose confidence does none of this (it never drops a view). But note the
knee-flexion column: up to this point the merge is no better than the side view alone (9.9 against 8.2 deg),
because wherever both views qualify for a leg joint the along view's poorer estimate is averaged in at equal
weight. The missing piece is a per-view reliability signal, and the root turned out to provide one. The oracle
per-joint pick shows what is still left on the table after that (leg outliers 8 % to 5 %, knee-flexion RMSE 7.4
to 5.1 deg) for a selector that knows more than visibility.

The root orientation needs its own rule, and it turned out to carry the reliability signal. GVHMR's root for
the along view pitches forward once the body is cut: +17 deg (signed median) with the shoulders out of its frame
and +50 deg with the hips out, against -5 and -3 deg for the side view, and its hip joints absorb the difference
(22 against 15 deg of hip error on the shoulders-cut frames, 56 against 8 on the hips-cut frames): the body looks
right in that camera and wrong in the world. A root mean weighted by usable-joint counts let that pitch in and out
of the reconstruction whenever the along view's count changed, a visible rocking of the whole body. The merge's
root is now the quaternion mean of the views' roots, each weighted by its presence (ramped over 0.5 s at the edges
of its lifted runs) times the square of the fraction of joints it finds usable (averaged over the same 0.5 s),
sharpened towards the heavier view when the two roots disagree by more than 30 deg: two good lifts of one walker
agree to about 10 deg, so a larger gap means one of them is wrong. When they disagree and weigh the same, two
continuity terms decide first: a view whose root turned faster than a person can (over 150 deg/s) since its
previous frame is discounted, and so is a view far from the previous fused root. That is the case of a 180-degree
turn that one view's lift takes the wrong way round, snapping through the front/back ambiguity: on the TUG sample
the side view's heading jumped 120 deg in a quarter of a second while the chair-facing view turned with the truth,
and the plain mean of two opposite headings had snapped the reconstruction instead of turning it; with the
continuity terms the merged heading stays within 5 deg of the truth through the turn (2.5 deg median over the
recording, no step over 90 deg; `merge.json` counts such steps). Those two terms are causal and cannot settle the
other form of the same failure: a view that turns the wrong way round at a human rate, indistinguishable from the
right one while both turn, and gives itself away only by snapping onto the true heading once the turn is over
(Sub 003's self-paced walk did this at both of its turns, 170-deg snaps in the side view, and the merge, which had
locked onto that heavier view, snapped with it). The merge runs offline, so a third rule reads the whole recording
first: within every episode of disagreement, a view whose own root turns more than 90 deg within a quarter of a
second, during the episode or just after it, is the one that took the turn the wrong way (a person cannot; the
snap is the lift's own correction), and it carries no weight over that episode, up to 4 s either side of its
snap. The same episode can also end without a snap when the wrong view's coverage simply runs out (the walker
turns out of its frame at the end of the mat, as the hurried walker did in the side view): nothing resolves the
disagreement, and at the hand-off the merged root would jump by the whole gap onto the surviving view, so a view
whose coverage ends, or begins, inside an episode while another view continues through it is treated like a
snapping one and carries nothing over that episode. When no view snaps or departs, or more than one does, the rule
is silent and the causal terms decide. That removed every flip of the rendered set. Every term is continuous in
its inputs, so nothing in the rule can step the root (`wgvp.fusion.fuse_root`).

Two more things had to change before the turns themselves read right. A view's lift can flip its root for a frame
or two and come straight back (16 frames of HC901's along view and 2 in each view of HC902 on the earlier
renders; 2 frames of an along view in two of the shipped recordings). Read as a snap, that
glitch made the offline rule set aside the view that was right, and the merged turn then completed in 0.6 s where
the walker took 1.6. Such an excursion, a step of more than 60 deg that is back where it was within 0.2 s, is now
interpolated across in each view before anything else reads that view's root (`repair_root_glitches`; counted per
view in `merge.json`). And the arbitration between the views is averaged over the cross-fade time, with the fused
root smoothed over 5 frames after it, so a hand-over inside a turn no longer shows as jitter. Over the 24 turns of
the six shipped recordings, lifted with each camera's field of view (below), the merged turn takes within 0.12 s of
the truth's time on average (10-90 % of the heading change; 0.27 s at worst) and peaks at 1.08 times the truth's
rate (1.29 at worst; before these changes, on the earlier renders, 2.9 on average and 29 at worst). Its heading is
3.8 deg from the truth at the median of a turn, 10.5 deg at the worst moment of an average turn, and 18 deg at the
worst of all (HC901). Under the lifter's default camera the same three numbers are 8.6, 18.0 and 41 deg, and one
turn of HC902 that takes 1.37 s in the truth lasts 2.03 s in the merge; with the field of view it lasts 1.40 s.
`merge.json` also counts the turns no person makes (`root_turns_over_90deg_in_quarter_s`): none on the six.

The same share then scales that view's weight for all
of its body joints, since a lifter that got the gravity frame wrong solved the whole body in it. On the
experiment's render this took the fused root to 6.6 deg (the views: 49.6 and 7.3) with a frame step at the 99th
percentile of 5.4 deg (the side view alone 7.3), and the merge below the better view in every column: legs 10.0
deg, outliers 8 %, all 21 joints 11.4 deg, knee-flexion RMSE 7.4 against the side view's 8.2 (last segment-rule
row of the table), at 98 % coverage and steps of 4.8 deg at the 99th percentile.

On the second render of HC901, with the room-wide side camera, the along view is the same stress case (its track
holds the walker for 99 % of the span, the standing phase attached by continuity, but its legs are usable on 35 %
of the span, its hips on 35 % of its own frames and the shoulders on 9 %; knee-flexion RMSE 46 deg over all of its
frames and 11 where its legs are in frame) and the side view now sees the whole walker but for the top of the head
(legs usable 95 %): alone it is at 11.4 deg on the legs (7 % of its leg-joint errors over 20 deg), 11.6 over all 21
joints, knee-flexion RMSE 5.9. The merge matches it on the legs (11.3 deg, 7 %, 11.6 over all joints) at 91 %
export, and on the 511 frames where both views have the legs it is below either view (13.0 deg against 13.6 and
15.5; knee-flexion RMSE 7.4 against 7.6 and 11.3), the along view's share of the root being 12 %. Over the export
its knee-flexion RMSE is 6.6 against the side view's 5.9, and the difference is seven frames (without them the
merge is at 5.9): the side view returns from its gap at 3.8 s while the walker stands, the cross-fade at the edge
of its run hands the along view the weight for a fifth of a second, and the along view's knees on a legs-only crop
of a standing walker are 20-75 deg wrong; those seven frames carry a knee weight of 0.1-0.7. Elsewhere in the
standing phase the merged knee is the side view's, 3 deg off, and it keeps its weight: the per-joint loss weight
reads a view's dissent by its share of the root (read in full, the along view's dissent would have put those knees
at 0.2). Its root error, 14.8 deg, is not the along view's pitch this time but the side view's: lifted from a
camera 2.3 m up looking down at the mat, GVHMR's gravity frame holds the pelvis pitched 6-15 deg from the truth on
every rendered session (the along views: 3-9 deg), the views are aligned by a yaw and an offset only, and the
merged root carries the views' tilts in proportion to their shares of it: on that round's renders, 14.8 and 15.4
deg here and in the TUG, where the side view held most of the root, and 6.4-9.1 on the other walks, where it held a
fifth to a half. A full-rotation alignment of the views onto the more upright one is the open item for that.

The side view's heading carried a second error, with a known cause and now a remedy. Left to itself, GVHMR
assumes a camera whose focal length equals the image diagonal, which for these panels is 1.5 times the along
camera's real focal length and 2.1 times the side camera's. A lifter that takes the lens for longer than it is reads
a walker off the optical axis as turned: on the six shipped recordings lifted that way, the side view's heading
error follows the walker's bearing from the axis at 0.86 deg per degree (r = 0.86). Taking that term out inside the
merge did not carry through to the body, which the lifter solved under the same wrong camera. The place for it is
the lift. With `gvhmr.focal_35mm` GVHMR is given each camera's real field of view, and the slope falls to 0.11 deg
per degree (r = 0.44), the side view's median heading error from 5.3-14.3 deg to 3.2-7.5 over the six recordings,
and the merged root's error from 13.3 to 10.3 deg. The bundled samples are lifted that way, and the two lifts are
compared under "How good these scores are". The tilt remains: the merged root is still 7-15 deg off, more than the
heading's 2-4 deg accounts for, and a full-rotation alignment of the views stays open.

What an export keeps is stricter than what the reconstruction shows. The red parts are filled only so the mesh can
be drawn; the export (`export_mask`) keeps a frame only when every leg joint and every torso joint (spine, neck and
head) is output, in runs of at least 0.5 s, and on those frames contains only the joints some view saw. An arm no
view saw is absent from an exported frame (NaN, weight 0); a frame without the whole legs or the whole torso is not
exported at all. On HC901 that is all of the IMU span (the along view alone has the legs for 41 % of it, the side
view for all of it). The torso is what the rule costs where one camera stands close: a view that shows the walker
from the chest down cannot vouch for the upper spine, and when the other view has no subject on those frames they
are dropped (`merge.json` gives the share as an `info:` flag, which does not change the status). On the earlier
renders that took 7 % of HC900's frames, 9 % of HC902's and 37 % of the hurried walk's. Two changes brought the
shipped six to 99.8-100 %. The shoulders now count down to 4 px from the top edge of the picture (8 px for every
other keypoint), and down to 2 px on a frame joined to a run that passes at 4: the spine's error does not rise
before that (8.3 deg at the median with the rule, 9.0 at 8 px), and of 15,855 view frames the trunk passes on 14,344
where it passed on 10,875. And the rest hold (above) gives the side view the standing walker the detector had
lost.

### The training target: `target.json`

Every recording gets two estimates of the same thing, one SMPL-X pose per frame on the IMU timeline, and a small
file that says which of them to train on. `merged_smpl.npz` is the mean of the two single-view lifts, joint by joint
(the next section). `jointfit_smpl.npz` is one body fitted to both views' keypoints at once (the section after "How
good these scores are"). `target.json` beside them names the target and gives the reason in a sentence:

| the recording | `target` |
|---|---|
| two views, and the joint fit passed its checks | `jointfit_smpl.npz` |
| two views, and the fit raised a warning (two cameras that look along one line, for example) | `merged_smpl.npz`; the fit is kept as `second_estimate` |
| one view | `merged_smpl.npz`: the fit takes two views |
| the fit did not run (no SMPL-X model file on the machine), failed, or is switched off | `merged_smpl.npz` |
| the merge itself failed | none (`target` is null), and the merge's own item says why |

```json
{"recording": "HC901_selfpace", "target": "jointfit_smpl.npz", "method": "jointfit",
 "why": "two views fitted as one body, and the fit's checks passed", "second_estimate": "merged_smpl.npz",
 "jointfit_status": "PASS", "merge_status": "PASS", "merge_stamp": "6f742a6b86ffd90b", "body_height_m": 1.7387}
```

The two files hold the same frames, the same joints and the same export mask, and the fields a learner trains on
have the same names in both (`body_pose`, `global_orient`, `transl`, `betas`, the three heading fields, `export_ok`,
the three weights, `transl_mat`, `frame_idx`, `imu_t_s`), so a loader reads `target.json` and then one file,
whichever it names. The merge's bookkeeping, which view made each joint and how the views were aligned, is in the
merge's file alone. The fit is made from the merge and carries its stamp: a fit left over from an earlier merge is never named, and `wgvp merge` and `wgvp
jointfit` both settle the file again. `jointfit.enabled` in `wgvp_config.json` switches the fit off for a data set,
which makes the merge the target everywhere.

**Why the fit, where it passes.** Against the rendered truth it is the better body and by far the better path: thigh
direction 6.0 to 3.8 deg, upper arm 9.0 to 6.0 deg, path 0.20 to 0.05 m and speed error 0.17 to 0.07 m/s at the
median, with the thigh, the shank, the ankle and the path better on all six recordings. On the one real recording
with a measured distance its path is the straighter (0.04 m off the walk's axis against 0.12), and it held its shape
when the side view was cut below the head, as a clinic's side camera frames a walker, where the merge's did not
(spread 0.05 against 0.14 m).

**What that choice does not buy.** Four things, and they travel with the target:

- **Distance reads longer than in the merge.** On the rendered rooms the fit's speed is 1.04 of the truth and the
  merge's 0.97. On the real recording the fit reads the walk 9 % long and the merge 2 % with the subject's height
  trusted, 18 and 11 % with the tape trusted. Both take their scale from the lifted body, which stood 5 to 8 % too
  tall. `body_height_m` in `target.json` is the fitted body's height, so a consumer who knows the subject's can
  rescale; the pipeline does not, and the rescale is a part of the cure: it leaves the fit 4 % long (13 % with the
  tape trusted).
- **The root's rotation, the heading and the seated phase of a TUG are where the merge has them.** The fit holds the
  root close to the merge's and does not try.
- **Both cameras must stand still.** The fit solves one camera pose for the whole recording and nothing detects a
  camera that moved: for hand-held or panned video, set `jointfit.enabled` to false.
- **The evidence is thin where it matters most.** The fit's settings were chosen on the same six rendered
  recordings it is scored on, with none held out, and one real recording has a truth, for distance alone.

### The merge: `merged_smpl.npz`

The merge is a pipeline stage (`wgvp/merge.py`, run after the lift and by `python -m wgvp merge <root>`). It is the
target wherever the joint fit is not, and the fit starts from it and takes its frames, export mask and weights. Its
file is built to supervise a model that goes from body-worn sensors to pose, and two things follow from that. The
body IMUs never enter it: the pipeline reads the recording CSV for the pelvis acceleration that picks the subject and
places IMU t = 0, and for nothing else, so a sensor-to-pose model trained on this target is not shown its own
input. And everything a learner needs to weigh the target is in the file.

| field | what it is |
|---|---|
| `body_pose` (T, 63) | the merged SMPL-X body pose on the video's frames, NaN for a joint no view saw |
| `global_orient`, `transl` (T, 3) | the merged root in the reference view's GVHMR world (y up); the other view is turned onto it by one yaw and one offset fitted on frames where both see most of the body. The pelvis path is fused from the views' per-frame steps, not their positions: each view's step counts in full across its camera's line of sight and 0.05 of that along it, where it is a change of monocular depth |
| `heading_yaw_deg`, `global_orient_heading_rel`, `root_vel_heading_rel_m_s` | the root split into the heading (what only the room knows) and the tilt and velocity in the heading frame (what the body can know): a sensor model is asked for the latter. The velocity is the fused video path's; on the mat its mean over 1.5 s is the mat path's |
| `transl_mat` (T, 3) | the pelvis **on the mat**, metres in the mat frame (X along, height from the video, Y across): the walkway's footprint under each foot in stance plus the video pose's foot-to-pelvis offset; a measured position with no depth estimate in it; NaN off the mat or without walkway data |
| `transl_mat_video`, `transl_from_walkway` (T, 3) | the video's own path turned into the mat frame, and `transl_mat` turned back into the reference view's world |
| `transl_scaled` (T, 3), `walkway_scale` (V), `transl_calibrated` | the translation after the per-view depth-scale fit against the footfalls (equal to `transl` without walkway data), the fitted factors, whether the fit ran |
| `view_mirrored` (V) | views found to be mirror images of the reference (a camera's picture flipped by its software): reflected back before merging, and flagged |
| `betas` (10) | one body shape for the recording, the median of GVHMR's per-frame estimates over whole-body frames of both views |
| `export_ok` (T) | the export rule above |
| `weight` (T, 21), `root_weight`, `transl_weight` (T) | per-joint, root and translation loss weights in [0, 1]: one confident view or two agreeing ones carry 1, two views that both vouched for a joint and disagree are discounted by the disagreement the merged joint contains (their difference times twice the smaller root share; 0.5 at 25 deg for an even split), nothing outside the export. `transl_weight` is the root's weight wherever the path exists |
| `transl_disagree_m` (T) | the distance between the two views' aligned pelvis positions (NaN with one view): recorded, not weighted, since on the samples it says nothing about the velocity's error (rank correlation -0.03) and little about the path's (+0.41) |
| `usable` (V, T, 21), `source` (T, 21), `share` (V, T) | which joints each view could vouch for, which views made each output, each view's share of the root |
| `side_on` (V, T), `sight_line` (V, T, 3) | how side-on each camera sees the walker (1: the legs flex in the image plane; 0: the walker comes straight at the camera), which sets whose stride swing the hips and knees take; and each camera's line of sight to the pelvis in the merged world, which sets how its pelvis step is counted |
| `view_alignment_yaw_deg`, `view_alignment_offset_m` | how each view was turned onto the reference view's world |

`merge.json` beside it holds the numbers: export fraction, continuity, the alignment residuals, the views' body
shapes and their difference, flags. On two of the synthetic sessions, against the rendered truth: HC901 (the
compromised views) exports all of the lifted frames with legs at 10.7 deg median, knee-flexion RMSE 5.3 deg, root
12.5 deg. HC900 (the dry-run session) exports all of them at 9.5 deg legs, 5.0 deg knee RMSE, 8.1 deg root, with
the views' roots within 10 deg of each other after alignment. The weight separates good cells from bad ones on the
five walks (median leg error 10.6 deg where weight > 0.5 and 13.4 below on HC901; 9.4 and 12.6 on HC900) and not on
the TUG (16.8 and 12.9): in the chair the two views mostly agree with each other (weight 0.77 at the median) on legs
that are 26.5 deg from the truth, and agreement between the views is all the weight reads. The weakest field is the
translation. A view's position is a monocular depth estimate, and after alignment the two
views' pelvis positions stand 0.35 to 1.7 m apart at the median on the six recordings. The merged path does not
average those positions. It sums the views' per-frame steps, each counted in full across its camera's line of sight
and at 0.05 along it, and lies 0.16-0.34 m from the truth at the median after one rigid fit. The distance between
the views is recorded as `transl_disagree_m` and not weighted (see the table). When the recording
CSV carries the pressure walkway, the merge uses it two ways. The WearGait export carries the walkway in 1,574 of
its 1,577 files, whatever the task (the mat records all through a session): at every 10-ms sample the lists of
loaded cells, `Walkway_X` / `Walkway_Y` as `'x1|x2|...'` cell indices on the GAITRite's 384 x 48 grid of half-inch
cells (4.88 x 0.61 m; `params.walkway_cell_cm`, 1.27), with each cell's footprint label in `WalkwayFoot` (`L`, `R`,
or `Other 1` for what the software did not assign) and its pressure level. The reader (`wgvp/imu_ref.py`) turns
them into the pressure-weighted centre of each labelled footprint per sample and groups those into stances (first
and last sample, median centre; 0.6 s each in a self-paced walk); unlabelled cells are counted and ignored. From
the stances the merge (a) fits per view one depth-scale factor of its path about the path's own centre, the mat's
yaw and offset in that view's world, against the landing foot's position (a model-free forward kinematics of the
mean body gives the feet), accepted only when its residual stays under 0.5 m, writing `transl_scaled` and
`walkway_scale`; and (b) builds the pelvis path **on the mat**, `transl_mat`: during each stance the walkway knows
where that foot is and the video pose knows where the pelvis is relative to it, both feet count during double
support, each stance's say ramps in and out over 0.12 s, gaps under 0.5 s are bridged. On the rendered mat
session (`sample_v6`, whose CSV carries cells fabricated from the rendered truth in the export's own form) the
video's path is 0.21 m from the truth at the median and 0.28 m at the 90th percentile, the scale-calibrated one
0.18 / 0.28 m, and the path from the walkway 0.03 / 0.11 m over the 91 % of frames it covers; in the mat frame,
with no fitting at all, its bias is 0.3 cm along the mat and 2.8 cm across. That measures the mechanism, not the
mat: the fabricated footprints come from the same truth as the video. The scale fit, by contrast, does not help
much (it fixes the path's length, not the per-frame error), because GVHMR's world path is integrated by its
network, not projected from the camera, so a scale applied afterwards has little to correct. What did move the
path is the camera given at the lift: with each camera's field of view the views' own paths are 0.19-0.50 m from
the truth at the median, where under the default camera they were 0.31-0.77. One caution for anyone testing
this on the rendered samples with the public recordings' own walkway rows: the render walks the neutral SMPL body
along a stance-pinned path that is 22 % shorter than the real walk the mat recorded (hc105: 3.4 m of footfalls
against 4.5 m), so the real footprints do not describe the rendered video; the samples' CSVs carry empty walkway
columns except the mat session's fabricated ones. The heading-frame velocity is the fused video path's
everywhere. On the mat its mean over 1.5 s is set to the mat path's, which keeps the video's detail within a stride
and takes the mat's measured progress: on the mat session the exported forward speed is 0.185 m/s RMS from the
truth, against 0.196 from the video alone and 0.348 when the velocity was the mat path's own derivative.

![the merge's file](docs/figures/merged_target.png)

*The merge of HC901 (`tools/make_target_figure.py`). A: what each view can vouch for, the export mask and the
per-joint weight; the along view (blue) vouches for a third to two thirds of the joints while the walker stands at
its feet (to 24 s) and for all of them only in parts of the walks, the side view (orange) for nearly all of them
throughout, and the export keeps the whole span. The weight dips in the first two seconds and during the walks,
where both views vouch for a joint and disagree. B: the heading, with the turns near 30, 37 and 43 s, and the
heading-relative tilt (the pitch sits 10 deg below zero while the walker stands: the side view's gravity frame, see
the text). C: the root's forward speed in the heading frame against the rendered truth: zero to 24 s, then four
bouts at about 1.2 m/s.*

![two-view demo](docs/figures/two_view_demo.gif)

*Fourteen seconds of HC901's walk, IMU 26-40 s (`python -m wgvp demo`, full video in the results of the run). Top: the two views with
ViTPose's keypoints for the chosen subject (the lifter's input; the left/right labels of the arms and legs are
kept consistent from frame to frame for the drawing, because 2D estimators flip sides on a side-on walker, and
the 3D stage does not use them) and what each view can use; the bright square in the side view is the sample's
sync light, lit while the IMU records, which the flash cross-check reads. Middle, left to right, three close-ups
that stand in the truth's room (its position, its floor and its mat; the reconstruction and the export are turned
into the room by one yaw fitted over the whole recording, so a walker facing another way than the truth is a
heading error, and the leg and root-orientation errors of the frame are printed): the rendered
truth on the mat; the reconstruction (red = a part no view could see, filled only so the mesh can be drawn; the along view
holds only the legs when the walker is near, and the upper body GVHMR invents from them is exactly what gets
excluded; near 38 s, the walker at its feet, it vouches for 8 of the 21 joints); the export, in which such parts are absent and a dropped frame shows the empty mat; and the whole room
from above: mat, walker, the operator at the desk, the bystander, the along camera (blue) and the side camera
(orange). Without a ground truth (a partner's recording) the truth panel and row are absent and the aerial view
shows the reconstruction on its own path. Bottom, the readout in two forms (`--readout strips|charts|both`): a filmstrip with a small render
every 2 s of what each view outputs (joints it cannot see in red), the merge and the truth, with the well-tracked
blocks marked at frame resolution by a bar under each row and a box around the consecutive thumbnails inside one
block (a view: its legs usable; the merge: the frames the export keeps; thumbnails outside a block are dimmed);
then the same as timelines on the shared IMU axis: each view's knee-flexion output (blue, orange), the merge
(green, with gaps where no view saw the joint) and the truth (white). The side view has the walker throughout;
the merge follows whichever view can see and hands over without a step.*

### Three subjects in a partner-style folder

To check that the package runs off a study folder rather than off its own samples, three synthetic sessions
were laid out as `Sub 001/Video/SP_combo.mp4` with the recording export under `Sub 001/IMU/` (the layout shown
under "Expected input"; `tools/make_partner_layout.py` builds it), scanned with the default rules, run on the CPU,
lifted on a T4, merged, and rendered with `wgvp demo`. Every sample's room follows the real session's, measured
on a Video Labeler frame: the side camera elevated, about 4 m from the mat with the operators' desk in the
foreground, so that the 16 x 2 ft mat spans some 60 % of its panel and a walker on it has the top of the head at
the top edge; the along camera at one end of the mat about 1 m high, so that a walker 2-3 m away shows from the
chest down (`sample_v2` keeps its lower, closer along camera as the truncation stress test). The self-paced and
hurried walks step off the mat's ends to turn, as in the clinic, and the side view keeps the walker through the turns
and, with the rest hold, through the standing start at the near end of the mat, where the detector loses a walker
who stands half out of the panel's edge. The third session is a new room
(`sample_v3/`, HC902): another walker (hc105), a co-walker on the far side of the mat, the along camera at the
other end of the mat and the side camera on the other side of the room.

The scan paired every video with its recording without a config edit. Against each session's rendered truth, on
the exported frames:

| session | export | legs (median) | knee-flexion RMSE | root | views turned onto each other to |
|---|---|---|---|---|---|
| Sub 001 (HC900) | 100 % | 9.5 deg | 5.0 deg | 8.1 deg | 10 deg, 1.70 m |
| Sub 002 (HC901) | 100 % | 10.7 deg | 5.3 deg | 12.5 deg | 9 deg, 0.76 m |
| Sub 003 (HC902) | 100 % | 11.1 deg | 6.9 deg | 7.4 deg | 11 deg, 0.57 m |

All three export the whole of the lifted span. In the earlier renders the legs-and-torso rule left out 9-10 % of
each: the frames in which the walker stood or turned close to the along camera, in its picture from the chest down,
while the side view had no subject. The shoulder rule and the rest hold (above) closed that.

The position figures in the last column are informational. After the alignment the two views put the pelvis 0.6-1.7
m apart on the frames both see, each a monocular depth, and the merge records that as an `info:` flag and discounts
nothing for it: the merged path is summed from each view's steps and not averaged from its positions (see "The
merge"), and `transl_weight` is 0.99-1.00 on all three, against 0.85-0.92 for the pose weights. The roots
turn onto each other to 9-11 deg. Every view found its subject from the accelerometer (r = 0.74-0.79 in the along
views, 0.48-0.63 in the side views), and the sync landed within 13 ms of the truth, confirmed by the light. Sub 003
finishes at PASS, Sub 001 and Sub 002 at WARN: Sub 001's side view has two frames with a joint-rotation jump over
30 deg per frame, and Sub 002's along view has three, together with a tracker fragment cut at an identity switch
(its track is four fragments, two accepted on their own timing and two attached by continuity). The CPU stages took
50-70 s per recording on a laptop, and the lift about 10 minutes per recording on a T4.

The first version of the third session used a walker whose sensor-derived path covered a single metre with a 5-mm
pelvis bob, and it produced the one confidently wrong result of these runs: the side view fragmented, the fragments'
own lags disagreed, and the joint lag landed 30 s off with a "high" confidence while the sync light sat uncompared
3 s into the video. That is where the two sanity rules above come from; both fire on the case. The walker was
replaced for the shipped sample, and the episode is written up in the design notes (section 2.8).

### Other tasks, and a mat walk with the walkway

Three more recordings were rendered and run through the same folder: a **Timed Up and Go** (`sample_v4`, HC903,
Sub 001's second recording: hc142's three repetitions of sit, stand, walk 3 m, turn, walk back and sit, with a
chair; the along camera faces the chair from beyond the turn), a **hurried-pace walk** (`sample_v5`, HC904, Sub
003's second recording: hc105 at speed with a co-walker, overshooting the mat at both ends) and a **self-paced walk
on the instrumented mat** (`sample_v6`, HC905, as Sub 004: hc105 in HC902's room, with the recording's walkway
columns fabricated from the truth in the export's own form, see above). Tandem gait has no public IK sequence to
render from. All three paired, synced and selected without a config edit (sync within 16 ms of the truth; every view
found its subject from the accelerometer, at r = 0.78 and 0.63 on the TUG, 0.85 and 0.65 on the hurried walk and
0.67 and 0.51 on the mat walk), and lifted and merged like the others:

| session | export | legs (median) | knee-flexion RMSE | root | two views' leg disagreement |
|---|---|---|---|---|---|
| TUG (HC903, hc142) | 99.9 % | 16.4 deg | 11.9 deg | 15.1 deg | 10.8 deg |
| hurried pace (HC904) | 99.8 % | 12.0 deg | 7.0 deg | 10.5 deg | 8.9 deg |
| mat walk (HC905) | 99.8 % | 9.2 deg | 5.7 deg | 8.4 deg | 9.2 deg |

The TUG walker is hc142: the first render used hc100's TUG, whose inertial fit carries its left ankle 163 deg off
(a visibly wrong foot in the rendered truth, and 26 deg of "error" that belonged to the truth), while hc142's fit
passes the ankle check (median ankle angles 10 deg, knees to 85 deg when seated). The TUG's errors are the largest
of the set and they sit in the chair. By phase, its legs are 12.9 deg off at the median while walking, 13.9 in the
turn and 15.1 turning to sit, like the walks, and 22.5 rising and 26.5 seated; its root is 9.7 deg off walking, 14.2
turning and 21.5 seated. The merge reads the seated knee 14 deg short of the truth. Both views have the seated legs
in frame and they mostly agree there (the weight on the seated legs is 0.77 at the median), so the weight does not
mark these frames, and the seated posture is the largest error left in the set.

The hurried walk was the hard case of the earlier renders. For its first 11 s the walker waits at the near end of
the mat, in the along camera's picture from the chest down, and the side view's detector loses them for about half
of that time. No view then has the torso, and the rule drops those frames: the export was 62 % on the earlier
render, and 81 % on this one before the rest hold. With the hold the side view keeps its box through the standing
start (164 frames, accepted by the keypoint check), the export is 99.8 % and the merge has the legs on every frame. On an earlier render the side view's lift
turned the wrong way at the final turn and ran out of frame mid-turn, the case that gave the root rule its departure
clause.

### How good these scores are, and what would improve them

Against the rendered truth, on the frames each recording exports. The six columns are the shipped samples, lifted
with each camera's field of view; the last column is the mean of the same six lifted with GVHMR's default focal
length.

| | HC900 | HC903 (TUG) | HC901 | HC902 | HC904 | HC905 | mean | mean, default camera |
|---|---|---|---|---|---|---|---|---|
| export (% of lifted frames) | 100 | 99.9 | 100 | 100 | 99.8 | 99.8 | 99.9 | 99.9 |
| legs, median (deg) | 9.5 | 16.4 | 10.7 | 11.1 | 12.0 | 9.2 | 11.5 | 11.8 |
| knee-flexion RMSE (deg) | 5.0 | 11.9 | 5.3 | 6.9 | 7.0 | 5.7 | 7.0 | 7.0 |
| root, median (deg) | 8.1 | 15.1 | 12.5 | 7.4 | 10.5 | 8.4 | 10.3 | 13.3 |
| heading, median (deg) | 1.7 | 3.9 | 4.1 | 2.7 | 2.7 | 3.3 | 3.1 | 5.9 |
| spine, median (deg) | 9.7 | 10.4 | 9.1 | 8.4 | 9.2 | 10.2 | 9.5 | 9.1 |
| arms, median (deg) | 15.6 | 16.8 | 17.0 | 15.0 | 15.8 | 16.3 | 16.1 | 16.5 |
| path after one rigid fit, median (m) | 0.16 | 0.19 | 0.33 | 0.18 | 0.34 | 0.21 | 0.24 | 0.38 |
| speed RMSE while moving (m/s) | 0.134 | 0.153 | 0.148 | 0.216 | 0.221 | 0.185 | 0.176 | 0.193 |

The leg, root, spine and arm figures are three-axis rotation errors per joint, a stricter measure than a flexion
angle. They are the lifter's numbers more than the merge's, and the merge adds a little to them: over each view's
own frames the merged legs are below both views on five recordings of the six (on the hurried walk they are 0.9 deg
above the along view's), and the merged knee RMSE is below both views on five (on HC902 it is 0.1 deg above the
side view's). The walks' legs and headings are the strong part. The seated phase of the TUG and the arms are the
weak part, and four measurements say where the rest sits.

**Field of view.** Giving the lifter each camera's real field of view (`gvhmr.focal_35mm`) in place of its default
focal length, which is 1.5 and 2.1 times too long for these two cameras, moves the root from 13.3 to 10.3 deg and
the heading from 5.9 to 3.1 at the mean, the path from 0.38 to 0.24 m, and the worst heading error inside a turn
from 18.0 to 10.5 deg (mean over the 24 turns). The legs and the knee do not move, and the spine is 0.4 deg worse.
A second pair of lifts, without the rest hold, gave the same step (root 12.8 to 10.2 deg, heading 5.6 to 3.0). The
samples' config carries the two focal lengths. A real camera's has to be read from its lens or from the picture
(`tools/estimate_focal.py`, under "Expected input"), and no clinic recording has had either yet.
The bundled phone clip has: see "When there is no truth" below.

**Amplitude.** The lifter draws every stride toward an average one. In a share-weighted mean of the two views the
knee's flexion has 0.71 of its true range and the hip's 0.62 (the slope of the estimate on the truth, outside the
rest intervals). The views do not read it alike: the side view, which sees flexion in its image plane, has slopes of
0.77-0.90 for the knee and 0.80-0.91 for the hip; the along view, which sees it as depth, 0.38-0.83 and 0.16-0.48.
Weighting whole leg joints toward the side view was tried and raises the three-axis leg error, since the along view
is the better one about the hips' other two axes. So the merge splits the flexion of hips and knees in two bands:
the slow part, the posture, stays the weighted mean, and the stride's swing comes from a second mean weighted by how
side-on each view sees the walker (`side_on` in the NPZ). That raises the slopes to 0.82 and 0.76 and lowers the
mean knee RMSE from 7.65 to 7.01 deg with the three-axis leg error unchanged; the hurried walk's knee RMSE is 0.3
deg worse and the TUG's does not move. What remains of the amplitude is the side view's own under-read.

**Timing.** Each view's flexion fits the truth best a quarter to half a frame ahead of it (8-17 ms), in every
recording. The 2D keypoints show no lead, so it arises inside the lifter. It is worth 0.3-0.9 deg of knee error and
is not corrected: a constant fitted to six rendered recordings is a property of one lifter's weights, and shifting
the export against the sensor timeline is the kind of error the pipeline exists to prevent.

**Path and speed.** The merged pelvis path is summed from the views' steps and not averaged from their positions,
which disagree by 0.35-1.7 m after alignment. Under the averaging rule the merged pelvis slid over the floor
whenever the views' shares changed, with the feet planted: the root's horizontal speed had an RMSE of 0.323 m/s
while the walker moved and 0.152 while they stood. It is 0.176 and 0.056 now, and the slope of estimated on true
speed is 0.88 instead of 0.70. The price is 0.05 m at the median of the path error (0.19 to 0.24 m), with the 90th
percentile better (0.55 to 0.44 m).

What would move the scores, largest first: (1) one body fitted to both views' keypoints at once, in place of the
mean of two single-view lifts. It is built, in the next section, and is the training target where its checks pass: it
supplies the depth each
view lacks (path 0.20 to 0.05 m, thigh direction 6.0 to 3.8 deg) and does not correct the tilted root, which was
expected of it; its own next step is a known camera pose; (2) the seated posture, where the root is 21.5 deg off,
the legs 26.5 and the knee 14 deg short while the two views agree; (3) the arms, at 16 deg the least accurate part
of the export and outside the export rule by design; (4) for real cameras, a measured field of view, worth 3 deg of
root and close to half of the heading error on these renders (`tools/estimate_focal.py` reads it from the room where
the camera sees the room's edges at a slant).

### One body fitted to both views: `jointfit_smpl.npz`

The merge is a mean of two single-view lifts, joint by joint. The joint fit is the stage after it: it fits one
SMPL-X body per frame to both views' 2D keypoints at once and writes `jointfit_smpl.npz` beside `merged_smpl.npz`,
on the same frames and the same joints. It runs after every merge (inside `run`, `gvhmr` and `merge`) and alone
with `python -m wgvp jointfit <root>`, and where it passes its checks it is the training target (`target.json`,
above). The stage runs on the CPU in 3-7 s per recording and needs the SMPL-X model file: a folder that holds
`smplx/SMPLX_NEUTRAL.npz`, named by `--body-models DIR` or by `jointfit.body_models` in `wgvp_config.json`, and
looked for in the GVHMR checkout's `inputs/checkpoints/body_models` otherwise. Without the file the fit is not run,
the merge is the target, and `target.json` says so.

No calibration comes with the videos, so the fit finds the pose of one camera against the other from the walker,
starting from the two lifts. The body starts at the merge's pose and the root is held close to the merge's: twelve
keypoints say where the pelvis is and little about how it is turned. The up direction is the cross product of the
two cameras' level directions, which each lift knows from its camera's roll. Joints and frames the merge does not
export are not exported here either. The output is a 3-frame moving average (the merge uses 5).

Scored against the rendered truth on the merge's exported frames (`tools/compare_jointfit.py`; every number is in
`docs/jointfit.json`). Each cell is merge / joint fit. All rows but the last two come from joint positions: the truth
is an SMPL body and both estimates are SMPL-X, and the two models' rotations for one posture differ by 3-19 deg
depending on the joint, a floor under the three-axis errors of the table above. The knee row is therefore the angle
between thigh and shank and not the flexion component of that table.

| | HC900 | HC903 (TUG) | HC901 | HC902 | HC904 | HC905 | median |
|---|---|---|---|---|---|---|---|
| thigh direction (deg) | 4.2 / 3.7 | 9.6 / 6.5 | 4.4 / 3.2 | 6.6 / 3.9 | 6.8 / 5.2 | 5.5 / 3.4 | 6.0 / 3.8 |
| shank direction (deg) | 5.0 / 3.8 | 5.8 / 4.6 | 6.3 / 4.9 | 5.2 / 5.0 | 7.4 / 4.8 | 4.0 / 2.6 | 5.5 / 4.7 |
| upper arm direction (deg) | 9.1 / 6.2 | 9.5 / 6.5 | 9.5 / 9.3 | 8.8 / 5.8 | 7.8 / 5.6 | 8.5 / 4.8 | 9.0 / 6.0 |
| knee angle RMSE (deg) | 6.0 / 4.8 | 10.0 / 7.8 | 5.6 / 4.7 | 5.9 / 5.5 | 6.5 / 5.8 | 6.7 / 5.8 | 6.2 / 5.6 |
| ankle position about the pelvis (cm) | 11.0 / 7.8 | 11.3 / 9.9 | 8.1 / 6.6 | 8.7 / 6.9 | 8.4 / 6.7 | 6.4 / 5.7 | 8.6 / 6.8 |
| path after one rigid fit, median (m) | 0.16 / 0.05 | 0.19 / 0.10 | 0.33 / 0.04 | 0.18 / 0.04 | 0.34 / 0.06 | 0.21 / 0.02 | 0.20 / 0.05 |
| speed RMSE while moving (m/s) | 0.134 / 0.072 | 0.153 / 0.070 | 0.148 / 0.082 | 0.216 / 0.049 | 0.221 / 0.063 | 0.185 / 0.050 | 0.169 / 0.067 |
| pelvis frame (deg) | 3.6 / 3.2 | 8.4 / 8.5 | 6.3 / 6.1 | 4.7 / 4.4 | 5.1 / 4.6 | 5.0 / 5.0 | 5.1 / 4.8 |
| root, three-axis (deg) | 8.1 / 8.1 | 15.1 / 15.1 | 12.5 / 12.5 | 7.4 / 7.7 | 10.5 / 10.6 | 8.4 / 9.0 | 9.4 / 9.8 |
| heading (deg) | 1.7 / 1.8 | 3.9 / 4.2 | 4.1 / 3.8 | 2.7 / 2.7 | 2.7 / 2.7 | 3.3 / 3.6 | 3.0 / 3.2 |

**What improves.** The thigh, the shank, the ankle and the path are better than the merge's on every one of the six
recordings. The path is the largest step, 0.20 to 0.05 m at the median, with the speed's RMSE from 0.17 to 0.07 m/s.
The thigh's sagittal range rises from 0.76 to 0.83 of the truth's and the knee's from 0.74 to 0.77.

**What does not.** The root's rotation, the heading and the seated phase of the TUG (root 21.5 and 22.6 deg) stay
where the merge has them. A second view of the same keypoints does not say how the pelvis is turned, and the fit
does not try.

**Limits.** These are rendered rooms. The fit's settings were chosen on these same six recordings, with none held
out. The fitted camera pose is 1.6-3.7 deg off the true one on five
recordings and 7.2 on HC901, where the along camera stands so close that both views show the whole body in 30 % of
the frames (`jointfit.json` notes the case: `views_complete_frac`). The scene's scale rests on the body's size as the
lifts saw it, and the speed reads 1.04 times the truth. With the render's true camera pose held fixed the path error
is 0.02 m and the speed ratio 1.00, so a camera pose measured once per room is the next step.

**What the file carries.** The fields a learner trains on are here under the merge's names. The pose, the path
(`transl`) and the body shape are the fit's. The heading, the heading-relative root and the
heading-relative velocity are computed from the fit's own pose and path by the merge's code. The export mask, the
loss weights and the walkway path (`transl_mat`) are the merge's, copied. One difference follows: on a walkway the
merge's velocity is tied to the mat's measured progress and the fit's is not. On the one rendered session with a
mat the fit's velocity is 0.05 m/s RMS from the truth without it, against the merge's 0.18 with it, and the fit's
path lies 0.02 m from the truth after one rigid fit, against 0.03 m for the merge's path on the mat.

`python -m wgvp demo <root>` renders the demo video of the recording's target, whichever file `target.json` names
(`<recording>_demo.mp4`). `--target merge` and `--target jointfit` render one estimate by name
(`<recording>_demo_merge.mp4`, `<recording>_demo_jointfit.mp4`), to watch the two side by side. Design and every
measurement: [docs/design.md](docs/design.md) sections 2.15 to 2.18.

**On one real recording.** One real two-camera recording has a truth for the distance walked: our own (OWN03), a
walk back and forth between two chalk lines 13.5 ft apart on pavement, centre to centre, a tablet to the side and a
phone at one end, both still, synced by a clap. It ships at half size as `examples/two_camera/`. `tools/chalk_walk.py`
reads the pelvis's position along the walk from the side view (the two lines' vanishing point and the known distance;
no lens, camera height or subject height) and scores each estimate against it (`docs/chalk_walk.json`). Both targets
ran unchanged, 100 % of the frames exported, and the fit placed the two cameras by itself. Counted in body heights,
so that the body's size drops out, the merge reads 0.955 of the true distance and the joint fit 1.043: about 5 %
short and 4 % long, the rendered rooms' result over again, and not separable within that measure's 2 to 3 %. The
fit's path is the straighter (0.04 m off the walk's axis against 0.12 m). In metres, as delivered, the walk reads 7 %
long in the merge and 14 % long in the joint fit. The same geometry measures the walker, and two inputs that should
agree are 4.8 % apart: the subject stands 1.651 m, which puts the lines 14.2 ft apart, not 13.5, and the recording
cannot say which is off. With the height trusted the bodies are 5 to 8 % too tall (1.74 and 1.78 m) and the walk
reads 2 % long in the merge and 9 % long in the joint fit. Every estimate takes its scale from its body, and the
body's height is the lifter's guess. If distances in metres matter, record the subject's height; a rescale by it is
not built.

**The same recording with the side view cut below the head**, as a clinic's side camera frames a walker: everything
still ran and exported 100 %, and the leg angles held (knee flexion within 1.8 deg rms of the full-view run, in both
targets). Two things changed. The side view's lift put the walker 8 % further away, so both targets' paths grew by
about 7 %, and the merge's path lost shape (spread about its line 0.04 to 0.14 m) while the fit's did not (0.05 m).
And the neck and head, which the side view no longer shows, are still taken from both views in 80 % of the frames,
because the shoulder keypoints at the picture's edge pass as seen; on rendered truth such a neck is 21 to 27 deg off.
Choosing the view that shows the head instead was tested on the rendered sessions and gained 1 to 3 deg on the neck
while losing up to 3 on the head, so the merge is unchanged. Sections 2.16 and 2.17 of the design document have the
method and every number, and section 2.18 the choice of target that rests on them.

### When there is no truth: a real clip

Every panel above that says "truth" exists because the walker was rendered. On a real recording the demo drops
them and shows what can be checked instead: each view's own lift beside the reconstruction and the export (so the
hand-overs are visible), the path from above, and in place of the truth curve the two views' disagreement on the
leg joints where both see them. The bundled example of our own clip (`examples/bundled`) is the test case for this
mode, lifted on a T4 like the rest:

![demo without a truth](docs/figures/demo_no_truth.png)

*One frame of `wgvp demo` on the two-view composite of the bundled phone clip (OWN02, 393 frames, 100 % exported),
lifted with the field of view read from the corridor's own lines: the right panel is marked as the mirror image it
is and shown reflected back, each view's own lift stands beside the reconstruction and the export, the path from
above replaces the aerial truth, the text column ends with each view's lifted floor slope, and the bottom chart is
the two views' leg disagreement (5.1 deg median) in place of an error against a truth. On real video the close-ups
show a floor and no mat (the mat is drawn only when the room is known, i.e. for a rendered truth), and a leg pair
the 2D detector put on one leg is drawn in amber: on the rendered sessions that happens on 0.1-23 % of a view's
frames (a side-on walker's legs cross once a stride) and the lifter's leg error on those frames is 8-19 deg
against 9-25 elsewhere, so the merge does not weight them down. The single-view clip (OWN01: 95 % exported, two
runs with a 0.7-s gap where the walker turned at the far end) renders the same way with one view.*

The two-view composite of the same clip (OWN02) is built the way a careless second camera would be: the left panel
is the clip zoomed in a little, the right panel the clip **mirrored** and zoomed in hard (1.16x and 1.64x,
different letterboxing; head and feet leave the right panel when the walker is close), which makes the right
panel's left leg the walker's right leg. On its lift the merge's mirror check reads the knees 44 deg apart as
lifted and 8 deg apart with the sides swapped, reflects that view back, and the two "views" then agree on the legs
to 5.1 deg with their knee curves on top of each other (root residual 7 deg, pelvis positions 0.10 m apart). It is
still one camera (the two panels carry the same information), so the synthetic sessions above remain the test of
the merge itself; what this clip tests is the panel detector on two zoom levels, the per-view scale, the mirror
detection, truncation, the field of view and the no-truth layout. The truncation found a limit worth knowing: on
the frames where the walker's feet are below the hard-zoomed panel, ViTPose still reports ankles with confidence
0.9 some 200 px inside the panel, so the usable-joint rule (confident and inside the panel) does not notice, and
the lifter extrapolates the legs there; the two views' agreement factor is what discounts those frames. A
geometric truncation test (the body's box against the panel edge) is the open item.

**The clip's field of view.** The file does not say which of the phone's lenses filmed it, and the lifter's
default assumes a 43.3-mm one. `tools/estimate_focal.py` reads 17.3 mm from the corridor's edges (quartiles 17.2
to 17.5 over 12 frames, a 103-deg diagonal, wider than a phone's main camera), and 20.2 and 32.2 mm from the two
panels of the composite. With no truth for the body, the check on that number is the floor: the corridor is level,
so the lifted pelvis should neither climb nor sink along the walk. `merge.json` now carries that slope per view
(`floor_slope_deg`) and the demo prints it. The clip was lifted again at a ladder of focal lengths, each panel
given the full picture's value times its own zoom:

| focal length at the lifter (mm, 35-mm equivalent, full picture) | lifted floor slope (deg): full view, left panel, right panel | pelvis height range over the walk (m) | the two panels' pelvis positions apart (m, median) |
|---|---|---|---|
| 43.3 (the lifter's default, both panels alike) | 21.9, 17.9, 5.2 | 2.83 | 0.54 |
| 33 | 14.7, 15.2, 14.9 | 1.90 | 0.10 |
| 27 | 9.7, 10.1, 9.0 | 1.31 | 0.08 |
| 20 | 3.1, 2.8, 1.6 | 0.58 | 0.10 |
| 18 | 1.3, 1.1, 0.0 | 0.41 | 0.11 |
| **17 (what the bundled config's 17.3 becomes at the lifter)** | **0.5, 0.5, 0.4** | **0.34** | **0.10** |
| 16 | 0.2, 0.1, 0.9 | 0.26 | 0.10 |
| 14 | 0.6, 0.4, 0.9 | 0.17 | 0.10 |

Under the default the walker climbs 2.8 m along a level corridor, and moves at 2.8 to 3.1 m/s in the camera's
frame while the world-frame path says 1.25: the lifter set the body two and a half times too far away (camera
depth out to 21 m, against 8.6 at 17 mm). At 17 mm the floor is level to 0.5 deg in all three views, the two
speeds are 1.16 to 1.24 and 1.10 to 1.13 m/s, and the walk reaches 6.8 m instead of 7.8. The 0.54 m by which the
two panels' pelvis positions differed was the same error, two zoom levels handed one lens: given their own focal
lengths they agree to about 0.10 m at every rung. What does not move is the body. Export fraction, mirror check, leg
agreement (5.3 deg under the default, 5.1 at 17 mm), forward lean (5 to 6 deg) and the knee curves are the same at
every rung, so the demo's close-ups look as before and the change is in where the body is, i.e. in the exported
path.

The floor does not pick 17.3 by itself. It is level to 1.3 deg or better anywhere from 14 to 18 mm (the signed slope
crosses zero near 16 mm for the full view and the left panel and at 18 mm for the right), so it brackets the lens
and the room's lines choose inside the bracket. Three limits go with that. The slope senses the field of view only
through the camera's tilt (this one looks about 14 deg down), so a level camera reads zero whatever the lens, and
it is an indicator and not a measurement: on the six rendered sessions, whose cameras are known, handing the
lifter the true field of view takes the median slope over the twelve views from 2.6 to 0.9 deg and levels 8 of the
12, while one view goes from 11.3 to 18.4. The 0.34 m of height range left at 17 mm, against the few centimetres a
pelvis rises and falls in a stride, says the lifted floor is level on average and not flat. And the clip is
hand-held, with about 26 deg of pan in its first 3 s, under a lift that assumes a fixed camera; neither the pan
nor the lens has been checked against anything but the picture.

**The joint fit on this clip.** `wgvp jointfit` skips the one-view clip (OWN01: the fit takes two views) and fits the
composite (OWN02) in 3 s. It meets the two panels' keypoints to 1.1 and 1.5 px and finds the two "cameras" 2 cm apart
and 0.2 deg turned against each other, which is what they are: one camera. `jointfit.json` says so in a flag (`the
views are one viewpoint`), next to the two this clip also raises: the up direction is the first view's own gravity
estimate, since two parallel level axes have no cross product, and the mirrored panel was reflected back. The body
hardly moves against the merge (root 0.5 deg, legs 1.3, arms 0.9 at the median), so the two demo videos look alike in
the close-ups. The path does move, by 0.48 m at the median:

| OWN02 | merge | joint fit |
|---|---|---|
| reach of the walk (m) | 6.9 | 7.7 |
| walking speed, median (m/s) | 1.09 | 1.29 |
| path length (m) | 13.8 | 15.4 |
| the slower ankle's speed while walking, median (m/s) | 0.04 | 0.40 |

There is no truth for the clip, so neither column can be scored. A tape measure along the corridor would settle it.

**What one camera knows about distance.** The rendered sessions can say which column is the likelier one.
`tools/one_camera_speed.py` takes each of their views by itself and reads the pelvis speed two ways: from the
lifter's own world path, the one the merge is fused from, and from the joint fit handed that one view twice, where
the distance to the walker comes from the body's size in the picture and the lens (`docs/one_camera.json`). Each
cell is the mean speed over the truth's, lifter / fit:

| | HC900 | HC903 (TUG) | HC901 | HC902 | HC904 | HC905 | median |
|---|---|---|---|---|---|---|---|
| the along camera, walks along its line of sight | 0.51 / 1.00 | 0.95 / 1.06 | 0.55 / 0.94 | 0.87 / 1.02 | 0.83 / 1.04 | 0.80 / 0.99 | 0.82 / 1.01 |
| the side camera, walks across its line of sight | 1.02 / 1.07 | 1.12 / 1.12 | 1.01 / 1.07 | 1.02 / 1.07 | 1.01 / 1.07 | 1.07 / 1.07 | 1.02 / 1.07 |

A camera that looks down the walk under-reads its speed through the lifter, by a fifth at the median and by half at
worst, and a keypoint fit of the same single view does not. The merge of two real views never shows this, because
the side camera's steps carry the path there. On this clip both panels look down the walk, the merge's path is the
lifter's along-sight steps, and its speed is 0.85 of the joint fit's, inside the lifter's range in the first row. So
the joint fit's longer path is the likelier one. Two limits on that reading: the rendered cameras' lenses are known
exactly and this clip's is an estimate, and the fit's distance scales with it; and the three short stretches (35 to
45 frames around the turns) where the side camera sees the walker along its line of sight are noisy under both
(0.45 to 0.99 for the lifter, 0.61 to 1.18 for the fit).

**Two faults of the fit on this clip.** It assumes still cameras, and this one is hand-held: in the seconds in which
the camera pans (the first 3 s and around second 13) the fitted body slides sideways at 0.3 to 0.5 m/s, and close to
zero in between. And the feet are not planted. The slower ankle moves at 0.40 m/s while walking, a third of the
pelvis's speed: the legs' swing toward and away from the camera is seen by neither panel, stays as lifted, and the
path grew longer around it. The merge's 0.04 m/s is the lifter's own foot lock and not a sign that its path is
right: on the rendered sessions the same measure reads 0.18 to 0.24 m/s for the merge and 0.13 to 0.18 for the joint
fit. For a recording like this one the body can come from either file, the path's length is likelier right in the
joint fit, and the feet are steadier in the merge. `target.json` names the merge for both clips: OWN01 has one view,
and OWN02's parallel level axes are a warning, which keeps its fit a second estimate. The fit does not take a
one-view recording yet; the table above is the case for letting it, with the camera's own motion among the unknowns.

## Outputs

`<data root>/wgvp_output/`

```
QC_SUMMARY.html, qc_summary.csv, qc_summary.json, manifest.json        cross-session QC (read this first)
<session>/deliver/                                                      the deliverable per session
   session_summary.json
   <recording>/sync.json              offset (video time / frame of IMU t=0), confidence, every cross-check and its agreement
   <recording>/selection_joint.json   the joint two-view subject selection (per-fragment scores, appearance attachments)
   <recording>/reconcile.json         agreement between the two views on the IMU timeline (2D and, after GVHMR, SMPL joints)
   <recording>/pose_agreement_<a>_<b>.npz   after GVHMR: per-frame, per-joint disagreement between the two views (t_imu_s, geodesic_deg (T,21), legs_median_deg)
   <recording>/target.json            after GVHMR: WHICH FILE IS THE TRAINING TARGET, and why (see "The training target" above)
   <recording>/merged_smpl.npz        after GVHMR: one pose per frame, the mean of the two views; the target where the joint fit is not
   <recording>/merge.json             its numbers: export fraction, continuity, view alignment, body shapes, each view's floor slope, flags
   <recording>/jointfit_smpl.npz      after GVHMR: one body fitted to both views' keypoints, on the merge's frames and joints; the target where its checks pass
   <recording>/jointfit.json          its numbers: reprojection per view, the fitted camera pose and how far it moved from its seed, flags
   <recording>/qc.json                walk-bout coverage per view, flags
   <recording>/<view>/subject_track.npz   frame_idx, pts_ms, bbox_xyxy, kp2d (COCO-17, face points blanked), visibility, source fragment id
   <recording>/<view>/smpl_global.npz     GVHMR SMPL-X parameters per frame, gravity-aligned world frame: global_orient (T,3), body_pose (T,63), betas (T,10), transl (T,3), frame_idx, pts_ms, imu_t_s,
                                          visible, edge_truncated, rest_held, pose_step_deg, chunk_id (per-frame quality fields, see "Partial bodies, exits and re-entries")
   <recording>/<view>/smpl_incam.npz      the same in the camera frame
   <recording>/<view>/kp2d_vitpose.npz    GVHMR's ViTPose 2D keypoints and boxes, the camera intrinsics the lifter was given (K_fullimg_cut;
                                          the 35-mm focal length behind them is in gvhmr_meta.json when gvhmr.focal_35mm is set)
   <recording>/<view>/gvhmr_meta.json, selection.json, qc.json
   validation_bundle/  logs/, thumbnails/ (skeleton-only PNGs), panels/ (edge-map previews with the view boxes), environment.json, privacy_check.json, metrics.json
<session>/work/     intermediates that stay on the machine (all-person tracks, cut clips, GVHMR dumps, demo/ videos)
```

Frame indices refer to the original combined video (`frame_idx`), times to its presentation timestamps (`pts_ms`),
and `imu_t_s = pts_ms / 1000 - offset_s` is the recording time.

![QC summary](docs/figures/qc_summary.png)

The QC summary has one row per recording and view. What good looks like:

| field | meaning |
|---|---|
| `selection_source`, `r_fine`, `margin` | `imu_xcorr` with r_fine >= 0.35 and margin >= 0.12 over the next person = confident. `imu_xcorr(low)`, `walkway_roi`, `operator_pick` = WARN: check the thumbnails |
| `present_frac`, `full_frac`, `partial_frac`, `n_gaps`, `lag_ambiguity` | share of the recording with the subject in this view, fully / partly in frame (head or feet cut), how many exits; lag_ambiguity > 0.85 = another person matched nearly as well at another lag |
| `walk_bouts_ok` | annotated Walk bouts covered (>= 80 %) by at least one view / total; a recording needs at least one |
| `sync_source`, `offset_s`, `offset_frame` | pelvis cross-correlation; `(validated by flash,labels,sheet)` lists the agreeing cross-checks; `*_minus_xcorr_s` are the differences |
| `reconcile_status`, `residual_lag_s`, `legs_median_deg` | the two views after the sync: residual lag about 0 (tolerance 3 frames); median leg-joint disagreement after GVHMR |
| `gvhmr_frames` | frames lifted in this view; `session_summary.json` adds bridged, edge-truncated and jump counts per view |
| `merge_status`, `export_frac` | the merge: PASS / WARN with the reason in `flags`, and the share of lifted frames its export keeps |
| `jointfit_status`, `target` | the joint two-view fit: PASS / WARN / FAIL, or SKIP where it does not apply (one view); and the file `target.json` names for the recording, the fit on PASS and the merge otherwise |
| `status` | PASS / WARN (chosen but unconfirmed, or a cross-check disagrees) / FAIL (nothing chosen, no sync, lifter failed), with the reason in `flags` |

Every stage writes its record; failures are explicit messages that say what to do (`pick-subject`, `set-roi`,
`check-gpu`, the config key to edit).

## Command line

```
python -m wgvp                         GUI
python -m wgvp scan <root>             find sessions / recordings / videos; write wgvp_config.json
python -m wgvp run <root> [--cpu-only] [--only <session> ...] [--force] [--gvhmr-root .. --gvhmr-python ..]
python -m wgvp gvhmr <root>            the GPU stage alone, on pending chunks
python -m wgvp merge <root> [--gvhmr-root DIR] [--body-models DIR]
                                       rebuild merged_smpl.npz, the joint fit and target.json from existing lifts; no GPU
                                       (exits non-zero and says so when a recording's merge failed)
python -m wgvp jointfit <root> [--session <id>] [--recording <id>] [--body-models DIR]
                                       the joint two-view fit alone (jointfit_smpl.npz), then target.json again; no GPU
python -m wgvp demo <root> [--session <id>] [--recording <id>] [--gt <sample>_gt.npz] [--t0 S --t1 S] [--gif FILE]
                    [--target auto|merge|jointfit]
                                       the demo video of the training target, or of one estimate by name (demo extras;
                                       written under work/)
python -m wgvp summary <root>          rebuild the QC summary
python -m wgvp dry-run [--cpu-only] [--out DIR]
python -m wgvp check-gpu --gvhmr-root <GVHMR> --gvhmr-python <python>
python -m wgvp set-roi <root> --session <id> --camera <view> --kind walkway|light
python -m wgvp pick-subject <root> --session <id> --recording <id> --camera <view> [--pick FRAME:X,Y ...]
                                       (a pick by frame and point is resolved against the tracking at run time and can ship with the data)
```

Videos are tracked once and cached by file hash; re-running after a config change is fast.

## Privacy statement

* The software runs entirely on the data owner's computers; it has no network function and downloads nothing.
* The deliverable (`deliver/` folders and the QC summary) contains **no image or video data**: SMPL-X body-model
  parameters (joint rotations, body-shape coefficients, root position), 2D skeleton keypoints and bounding boxes of
  the consented subject only (face keypoints are blanked; other people in the room are not exported), scalar
  sync/QC values, logs, and skeleton-only PNG thumbnails drawn on a black background from keypoints. The panel
  previews are Canny edge maps of one frame with the view boxes drawn; drop `validation_bundle/panels/` if an IRB
  prefers.
* `privacy_check.json` in every validation bundle lists every delivered file and asserts the rule (no video/image
  files, no image-like arrays in NPZ files, PNGs are sparse skeleton renders). A session fails its privacy check
  otherwise. (The sparseness test is tuned for a room with a handful of people; a crowd scene can trip it, see the
  examples.)
* Intermediates with pixels (cut clips for the lifter, GVHMR's input copies) are written under `work/`, deleted by
  the lifter stage after use where GVHMR produced them, and stay on the machine in any case.
* Body-shape coefficients (`betas`) and gait kinematics are weak re-identifiers in principle; they are the data the
  analysis needs.

## What has not been tested on real clinic recordings

The pipeline was validated on a rendered two-view session with a known answer, on unit tests, and on real
multi-person public footage with a simulated sensor reference. It has not yet run on real clinic video against a
real IMU: the panel geometry of real encodes, the tracker under clinic lighting with sensor straps and
de-identification blur, GVHMR on real truncated bodies, and the Walk-label and annotation-sheet cross-checks are
exercised only on synthetic inputs. The merge's alignment of the two views' GVHMR worlds (one yaw and one offset
from shared frames) and GVHMR's translation scale have been checked on rendered cameras only; the walkway reader
has been run on the public recordings (1,574 files parse; stances alternate feet 90 % of the time), but the
pelvis path on the mat has only been checked against a rendered truth whose footprints were fabricated from that
truth, never against a real mat under a real video. The field of view at the lift was measured on rendered pinhole cameras whose
intrinsics are known exactly: a real lens's distortion, and a panel that the recording software has cropped or
scaled, are untested. `tools/estimate_focal.py` has read one real camera, a phone's, and no clinic camera; on that
clip the only check of its number is the lifted floor coming out level. The rest hold has been run only on rendered people. The step-summed path,
the stride-band flexion rule and the velocity on the mat were tuned and scored on the same six rendered recordings,
and so were the settings of the joint two-view fit, which is the training target wherever it passes its own checks.
Those checks do not include a camera that moves: the fit assumes both stand still. Its self-calibration of the
cameras has met rendered pinhole cameras and one real pair, a tablet and a phone outdoors, where the only truth was the distance walked and
neither lens was measured. On that recording the lifted bodies stand 5 to 8 % taller than the
subject (1.651 m), and the subject's height and the 13.5 ft measured between the chalk lines disagree by a further
4.8 % that is not resolved: GVHMR's scale on a real person is unverified.
The decisive number, the subject's `r_fine` in real footage, comes out of the first real run and is printed in the QC summary. [docs/design.md](docs/design.md) section 3 lists everything, and
section 4 the questions to settle with a dataset's owners before that run.

## Repository layout

```
wgvp/            the package: discovery, video_io (panels, cutting), tracking, signals, imu_ref, selection, sync,
                 flash, labels, reconcile, fusion + merge (the two views' mean), jointfit (the two-view fit, and
                 the choice of training target), demo, gvhmr_stage, qc, summary,
                 session (orchestration), cli, gui, dryrun
weights/         yolov8n-pose.pt (ultralytics, AGPL-3.0)
sample/          the synthetic dry-run session (video + recording CSV + expected answers)
sample_v2/       a second synthetic session with two deliberately compromised views and the walker's ground-truth
                 3D pose, for the two-view reconstruction experiment (tools/two_view_fusion.py)
sample_v3/       a third room: another walker, a co-walker, the along camera at the far end (legs cut when close),
                 the side camera on the other side; ground truth saved as well
sample_v4/       a Timed Up and Go with a chair (three repetitions), ground truth saved
sample_v5/       a hurried-pace walk with a co-walker, ground truth saved
sample_v6/       a self-paced walk on the instrumented mat, with walkway columns in the recording CSV
examples/        bundled/ (our own clip + a two-view composite of it, runs offline); two_camera/ (our own
                 two-camera recording between two chalk lines, runs offline); the fetched public-footage
                 corpus: sources.json, fetch_examples.py, verify_examples.py; make_bundled.py
tests/           unit tests (python -m pytest tests -q)
tools/           make_sample.py (how the samples were rendered), make_partner_layout.py (the six samples in a
                 Sub 00N/Video + IMU layout), make_readme_figures.py, two_view_fusion.py (the fusion experiment
                 against the truth), make_target_figure.py, estimate_focal.py (a camera's field of view from the
                 room's straight edges), compare_jointfit.py (the merge and the joint two-view fit against the truth),
                 one_camera_speed.py (one view's walking speed: the lifter's path against a keypoint fit),
                 chalk_walk.py (a real walk between two ground marks a known distance apart: the truth from one side
                 view, and every estimate against it)
env/             GVHMR environment notes; docker/ container recipe (untested build)
docs/            design.md (design, test results, open questions), jointfit.json (the joint fit's comparison),
                 chalk_walk.json (one real two-camera recording against a chalk truth), figures/,
                 demos/ (the demo videos of the six rendered sessions and of the two-camera recording, with a looping excerpt each)
```

## Citing

If this pipeline is useful in your work, cite it with the metadata in [CITATION.cff](CITATION.cff), and cite GVHMR
(Shen et al., "World-Grounded Human Motion Recovery via Gravity-View Coordinates", SIGGRAPH Asia 2024) for the
lifting step.

## Licences

wgvp is released under the GNU Affero General Public License v3.0 ([LICENSE](LICENSE)). The 2D tracker it imports
(`ultralytics`) and the bundled `yolov8n-pose.pt` weights are AGPL-3.0 as well. GVHMR is licensed by its authors for
research and non-commercial use and is not redistributed here; the SMPL and SMPL-X body models require registration
with their authors. The fetched example corpus is not redistributed: `fetch_examples.py` downloads each public video onto
your machine, and the sources are attributed in `examples/sources.json`. The clips in `examples/bundled/` and
`examples/two_camera/`, and the demo video made from the latter, are our own recordings, published with the consent
of the person in them, under the same licence as the code.

Mike Caiola, Credence Management Solutions, LLC - mcaiola@credence-llc.com
