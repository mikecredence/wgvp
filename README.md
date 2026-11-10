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
7. **Merges the two views into one pose per frame** and writes the training target, `merged_smpl.npz`: only the
   joints some view saw, cross-faded at the hand-overs, with per-joint loss weights, a heading-relative root and one
   body shape per recording. This is what a sensor-to-pose model is trained on, and nothing from the body sensors
   enters it: the recording CSV serves the sync and the subject identity, never a joint angle.

![two views](docs/figures/two_views_tracking.png)

*Two frames of the synthetic session with the detected panels (green and orange boxes), every tracked person and,
beside each, that person's pelvis correlation with the IMU at the shared lag. In the along view the subject
reaches 0.70; at the side view's distance the pelvis bob is below the keypoint noise, everyone it scores stays at
0.0-0.16, and its subject is attached by the cross-view cue, the motion along the mat against the along view's
subject (|r| 0.96).*

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

**1. Dry run (3-6 min on a laptop CPU).** The bundled session has a known answer; every check must pass:

```bash
python -m wgvp dry-run --cpu-only
```

```
  [PASS] two panels detected at the right boxes
  [PASS] sync offset within 2 frames of truth (pelvis cross-correlation): estimated 2.999 s vs true 3.0 s ...
  [PASS] left: chosen track is the rendered subject: 100% of 1458 frames hold the GT hip inside the chosen box ...
  [PASS] right: the chosen track holds the subject wherever the tracker found it: 97% of the 1308 frames ...
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
**Open QC summary**. The GUI needs tkinter, which the python.org installers include; on Debian/Ubuntu
`sudo apt install python3-tk`.

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
hurried-pace, TUG and a walk on the instrumented mat).

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

*A: inside a walk bout the subject's video pelvis signal (green; the along view's closely, the side view's at the
keypoint noise of its scale) follows the accelerometer (black); the co-walker's (red) does not. B: correlation
versus lag for every scored person; the along view's subject peaks at the shared lag of 3.00 s (true 3.00 s), and
the side view's subject (fragments 5, 6, 27 and 29) is attached by the cross-view cue and by continuity rather
than by a peak of its own. C: coverage per view on the IMU timeline; the side view's gaps are in the first eleven
seconds, the walker standing at the near end of the mat half out of the panel's edge, where the detector has it
only in pieces, and half a second at the turn there (36 s); the along view has them.*

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
(residual lag -33 ms, within the 3-frame tolerance) and the leg joints disagree by a median of 7.3 degrees. Body
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
* **Per-frame quality, so nobody has to infer it:** `visible`, `edge_truncated`, `pose_step_deg` (largest joint
  step from the previous frame) and `chunk_id` in every SMPL NPZ; `pose_agreement_<a>_<b>.npz` beside
  `reconcile.json` with the per-joint disagreement between the two views on every IMU frame; and a QC flag when any
  frame moves a joint by more than 30 deg in one frame.

The synthetic sessions are real three-dimensional scenes, not composites: `tools/make_sample.py` places the
walker (an SMPL body driven by a public WearGait inertial fit), the co-walker, the seated operator, the desk, the
bystander, the mat, the floor tiles and the walls in one world, puts two perspective cameras where the aerial panel
of the demo draws them (their poses and intrinsics are saved in the truth file as `<view>_cam_pose` and
`<view>_intrinsics`), and renders each camera's picture with pyrender, an OpenGL rasteriser. The per-view truth
(`<view>_joints2d`, the in-frame flags) is the projection of the same joints through the same cameras, so a view's
truncation, a camera's height and tilt and the panel sizes are what the geometry gives, not what was drawn by
hand. What the renders lack is appearance (textures, clothing, lighting, lens distortion, motion blur), which a
game engine would add and which matters for the 2D detectors rather than for the geometry; the detectors have
coped with the grey bodies so far, and the real-footage examples carry that test.

These numbers come from rendered bodies. GVHMR on real clothed, truncated people is on the list of things only the
first real run can show.

### 3D reconstruction from two partial views

The real session layout is harsher than the dry-run sample: the along-walkway camera sits at the end of the mat,
so the participant fills and overflows the frame when close, and the side camera's framing cuts the head. To ask
whether two such views still give back one correct 3D pose, `sample_v2/` renders the same walk from an along view
that keeps only the legs when the walker is near (hips out of frame half of the time) and saves the walker's true
SMPL pose and 3D joints beside it. The experiment below was run on its first render, whose side view was aimed at
hip height (head above the frame in 77 % of frames); the shipped `sample_v2/` has since taken the room-wide side
camera of every other sample, and its current numbers close this section. `tools/two_view_fusion.py` measures, against that truth, each view's GVHMR error per joint and by what
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
or two and come straight back (16 frames of HC901's along view, 2 in each view of HC902). Read as a snap, that
glitch made the offline rule set aside the view that was right, and the merged turn then completed in 0.6 s where
the walker took 1.6. Such an excursion, a step of more than 60 deg that is back where it was within 0.2 s, is now
interpolated across in each view before anything else reads that view's root (`repair_root_glitches`; counted per
view in `merge.json`). And the arbitration between the views is averaged over the cross-fade time, with the fused
root smoothed over 5 frames after it, so a hand-over inside a turn no longer shows as jitter. Over the 24 turns of
the six rendered recordings the merged turn takes within 0.11 s of the truth's time on average (10-90 % of the
heading change; 0.47 s at worst) and peaks at 1.2 times the truth's rate (1.5 at worst; before these changes 2.9 on
average and 29 at worst). Its heading is 7.5 deg from the truth at the median of a turn, 18 deg at the worst moment
of an average turn, and 43 deg at the worst of all (HC901, where the side view holds the root nearly alone).
`merge.json` also counts the turns no person makes (`root_turns_over_90deg_in_quarter_s`): none on the six.

The same share then scales that view's weight for all
of its body joints, since a lifter that got the gravity frame wrong solved the whole body in it. On the
experiment's render this took the fused root to 6.6 deg (the views: 49.6 and 7.3) with a frame step at the 99th
percentile of 5.4 deg (the side view alone 7.3), and the merge below the better view in every column: legs 10.0
deg, outliers 8 %, all 21 joints 11.4 deg, knee-flexion RMSE 7.4 against the side view's 8.2 (last segment-rule
row of the table), at 98 % coverage and steps of 4.8 deg at the 99th percentile.

On the shipped render of HC901, with the room-wide side camera, the along view is the same stress case (its track
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
merged root carries the views' tilts in proportion to their shares of it: 14.8 and 15.4 deg here and in the TUG,
where the side view holds most of the root, 6.4-9.1 on the other walks, where it holds a fifth to a half. A
full-rotation alignment of the views onto the more upright one is the open item for that.

The side view's heading carries a second error, with a known cause. GVHMR is handed a default camera whose focal
length equals the image diagonal, which for these panels is 1.5 times the along camera's real focal length and 2.0
times the side camera's. A lifter that takes the lens for longer than it is reads a walker off the optical axis as
turned: over the rendered sessions the side view's heading error follows the walker's bearing from the axis at 0.96
deg per degree (r = 0.69; its spread is 14 deg against the along view's 4), and taking that term out with the known
bearing halves that view's heading error (7.6 to 3.9 deg median). Taking it out inside the merge did not carry
through to the body, which the lifter solved under the same wrong camera (HC901's legs got worse), so none of it is
shipped. The place for it is the lift, with each camera's real field of view given to GVHMR; that needs the
recordings lifted again and is the first item on the list of what would improve the scores below.

What an export keeps is stricter than what the reconstruction shows. The red parts are filled only so the mesh can
be drawn; the export (`export_mask`) keeps a frame only when every leg joint and every torso joint (spine, neck and
head) is output, in runs of at least 0.5 s, and on those frames contains only the joints some view saw. An arm no
view saw is absent from an exported frame (NaN, weight 0); a frame without the whole legs or the whole torso is not
exported at all. On HC901 that is 91 % of the IMU span (the along view alone has the legs for 35 % of it, the side
view for 95 %). The torso is what the rule costs where one camera stands close: a view that shows the walker from
the chest down cannot vouch for the upper spine, and when the other view has no subject on those frames they are
dropped (7 % of HC900's frames, 9 % of HC902's and 37 % of the hurried walk's, see below; `merge.json` gives the
share as an `info:` flag, which does not change the status).

### The training target: `merged_smpl.npz`

The merge is a pipeline stage (`wgvp/merge.py`, run after the lift and by `python -m wgvp merge <root>`), and its
file is built to supervise a model that goes from body-worn sensors to pose. Two things follow from that. The body
IMUs never enter it: the pipeline reads the recording CSV for the pelvis acceleration that picks the subject and
places IMU t = 0, and for nothing else, so a sensor-to-pose model trained on this target is not shown its own
input. And everything a learner needs to weigh the target is in the file.

| field | what it is |
|---|---|
| `body_pose` (T, 63) | the merged SMPL-X body pose on the video's frames, NaN for a joint no view saw |
| `global_orient`, `transl` (T, 3) | the merged root in the reference view's GVHMR world (y up); the other view is turned onto it by one yaw and one offset fitted on frames where both see most of the body |
| `heading_yaw_deg`, `global_orient_heading_rel`, `root_vel_heading_rel_m_s` | the root split into the heading (what only the room knows) and the tilt and velocity in the heading frame (what the body can know): a sensor model is asked for the latter |
| `transl_mat` (T, 3) | the pelvis **on the mat**, metres in the mat frame (X along, height from the video, Y across): the walkway's footprint under each foot in stance plus the video pose's foot-to-pelvis offset; a measured position with no depth estimate in it; NaN off the mat or without walkway data |
| `transl_mat_video`, `transl_from_walkway` (T, 3) | the video's own path turned into the mat frame, and `transl_mat` turned back into the reference view's world |
| `transl_scaled` (T, 3), `walkway_scale` (V), `transl_calibrated` | the translation after the per-view depth-scale fit against the footfalls (equal to `transl` without walkway data), the fitted factors, whether the fit ran |
| `view_mirrored` (V) | views found to be mirror images of the reference (a camera's picture flipped by its software): reflected back before merging, and flagged, each view's path scale, and whether the fit ran; see the translation paragraph below |
| `betas` (10) | one body shape for the recording, the median of GVHMR's per-frame estimates over whole-body frames of both views |
| `export_ok` (T) | the export rule above |
| `weight` (T, 21), `root_weight`, `transl_weight` (T) | per-joint, root and translation loss weights in [0, 1]: one confident view or two agreeing ones carry 1, two views that both vouched for a joint and disagree are discounted by the disagreement the merged joint contains (their difference times twice the smaller root share; 0.5 at 25 deg for an even split), nothing outside the export |
| `usable` (V, T, 21), `source` (T, 21), `share` (V, T) | which joints each view could vouch for, which views made each output, each view's share of the root |
| `view_alignment_yaw_deg`, `view_alignment_offset_m` | how each view was turned onto the reference view's world |

`merge.json` beside it holds the numbers: export fraction, continuity, the alignment residuals, the views' body
shapes and their difference, flags. On two of the synthetic sessions, against the rendered truth: HC901 (the compromised views) exports 91 % of the lifted frames with legs at 11.3 deg median, knee-flexion RMSE 6.6 deg, root 14.8 deg; the weight separates good cells from bad ones (median leg error 11.3 deg where weight > 0.5, 14.1 below). HC900 (the dry-run session) exports 90 % at 8.7 deg legs, 6.6 deg knee RMSE, 7.7 deg root, with the views turned onto each other to within 12 deg and 0.17 m. The weakest field is the translation: GVHMR's position is a
monocular depth estimate, within 0.1 m near the middle of the walk and off by 1 to 3 m at the far ends of it in a
view that sees the walker small (an assumed focal length). Where both views see the subject their disagreement
exposes this (`transl_disagree_m`, discounted by `transl_weight`); a single far view does not. When the recording
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
video's path is 0.28 m from the truth at the median and 0.65 m at the 90th percentile, the scale-calibrated one
0.29 / 0.49 m, and the path from the walkway 0.034 / 0.089 m over the 86 % of frames it covers; in the mat frame,
with no fitting at all, its bias is 0.1 cm along the mat and 4 cm across. That measures the mechanism, not the
mat: the fabricated footprints come from the same truth as the video. The scale fit, by contrast, does not help
much (it fixes the path's length, not the per-frame error), because GVHMR's world path is integrated by its
network, not projected from the camera (its camera-frame and world-frame translations differ by 0.4 to 1.6 m RMS
as a rigid pair), so neither a focal-length nor a body-scale correction applies. One caution for anyone testing
this on the rendered samples with the public recordings' own walkway rows: the render walks the neutral SMPL body
along a stance-pinned path that is 22 % shorter than the real walk the mat recorded (hc105: 3.4 m of footfalls
against 4.5 m), so the real footprints do not describe the rendered video; the samples' CSVs carry empty walkway
columns except the mat session's fabricated ones. The heading-frame velocity comes from the walkway path where
it exists and from `transl_scaled` elsewhere.

![the training target](docs/figures/merged_target.png)

*The target of HC901 (`tools/make_target_figure.py`). A: what each view can vouch for, the export mask and the
per-joint weight; the along view (blue) vouches for half of the joints at most, and the weight dips only where
both views vouch for a joint, disagree, and the along view holds a share of the root (the walks), not through the
standing phase, where its legs-only crop dissents from a side view that holds the whole root. The gaps in the
export are the legs-and-torso rule: at 1-4 s the side view has lost the standing walker and the along view holds
only the legs, and at 36-38 s the side view's torso flickers in runs shorter than the 0.5-s minimum. B: the heading and the heading-relative tilt (the pitch sits 10 deg below zero
while the walker stands: the side view's gravity frame, see the text). C: the root velocity in the heading frame
against the rendered truth, with the frames of low root weight shaded: the side view's gaps, where only the far
along view has the walker.*

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
excluded; at 35-39 s, the walker at its feet, it holds nothing); the export, in which such parts are absent and a dropped frame shows the empty mat; and the whole room
from above: mat, walker, the operator at the desk, the bystander, the along camera (blue) and the side camera
(orange). Without a ground truth (a partner's recording) the truth panel and row are absent and the aerial view
shows the reconstruction on its own path. Bottom, the readout in two forms (`--readout strips|charts|both`): a filmstrip with a small render
every 2 s of what each view outputs (joints it cannot see in red), the merge and the truth, with the well-tracked
blocks marked at frame resolution by a bar under each row and a box around the consecutive thumbnails inside one
block (a view: its legs usable; the merge: the frames the export keeps; thumbnails outside a block are dimmed);
then the same as timelines on the shared IMU axis: each view's knee-flexion output (blue, orange), the merge
(green, with gaps where no view saw the joint) and the truth (white). The side view has the walker throughout but for half a second at 36 s;
the merge follows whichever view can see and hands over without a step.*

### Three subjects in a partner-style folder

To check that the package runs off a study folder rather than off its own samples, the three synthetic sessions
were laid out as `Sub 001/Video/SP_combo.mp4` with the recording export under `Sub 001/IMU/` (the layout shown
under "Expected input"; `tools/make_partner_layout.py` builds it), scanned with the default rules, run on the CPU,
lifted on a T4, merged, and rendered with `wgvp demo`. Every sample's room follows the real session's, measured
on a Video Labeler frame: the side camera elevated, about 4 m from the mat with the operators' desk in the
foreground, so that the 16 x 2 ft mat spans some 60 % of its panel and a walker on it has the top of the head at
the top edge; the along camera at one end of the mat about 1 m high, so that a walker 2-3 m away shows from the
chest down (`sample_v2` keeps its lower, closer along camera as the truncation stress test). The self-paced and
hurried walks step off the mat to turn, as in the clinic, and the side view keeps the walker through the turns;
its track holds the walker wherever the tracker found one, standing frames included, and what it lacks is where
the detector lost the walker, standing half out of the panel's edge at the near end of the mat before the first
bout (the along view has those frames). The third session is a new room
(`sample_v3/`, HC902): another walker (hc105), a co-walker on the far side of the mat, the along camera at the
other end of the mat and the side camera on the other side of the room.

The scan paired every video with its recording without a config edit. Against each session's rendered truth, on
the exported frames:

| session | export | legs (median) | knee-flexion RMSE | root | views turned onto each other to |
|---|---|---|---|---|---|
| Sub 001 (HC900) | 90 % | 8.7 deg | 6.6 deg | 7.7 deg | 12 deg, 0.17 m |
| Sub 002 (HC901) | 91 % | 11.3 deg | 6.6 deg | 14.8 deg | 23 deg, 0.77 m (flagged) |
| Sub 003 (HC902) | 90 % | 11.3 deg | 9.7 deg | 6.4 deg | 12 deg, 0.70 m (flagged) |

The exports are the legs-and-torso rule at work: the frames left out are those in which the walker stands or turns
close to the along camera, in its picture from the chest down, while the side view has no subject (Sub 001 and Sub
003, 7 and 9 % of the frames), or in which neither view has the legs (Sub 002).

The flags are the translation: on Sub 002 and Sub 003 the two views put the pelvis 0.7-0.8 m apart on the frames
both see, each a monocular depth under an assumed focal length, and the merge says so (`transl_weight` averages
0.07 on Sub 002, 0.41 on Sub 003 and 0.57 on Sub 001, against 0.87-0.93 for the pose weights); Sub 001's views, aligned on
1,163 shared frames now that its side view holds the standing walker, agree to 0.17 m and raise none. The roots
turn onto each other to 12-23 deg, which is the side view's pitch (above). Sub 002's side view confirmed the sync, which its along view cannot, and the light settled the lag (the
pelvis match had locked onto the subject's own walk a bout later, the case written up under "How the subject is
found and synced"); Sub 003's side view attached its fragment at the lag of the other view (a WARN, as designed:
that view's track has the walker for 70 % of the recording, the standing start being undetected at the panel's
edge, and 98 % of the frames inside the walk bouts). The whole run, scan to demos, took the CPU stages at about 90 s per session on a
laptop and the lift at about 10 minutes per recording on a T4.

The first version of the third session used a walker whose sensor-derived path covered a single metre with a 5-mm
pelvis bob, and it produced the one confidently wrong result of these runs: the side view fragmented, the fragments'
own lags disagreed, and the joint lag landed 30 s off with a "high" confidence while the sync light sat uncompared
3 s into the video. That is where the two sanity rules above come from; both fire on the case. The walker was
replaced for the shipped sample, and the episode is written up in the design notes (section 2.8).

### Other tasks, and a mat walk with the walkway

Three more sessions were rendered and run through the same folder: a **Timed Up and Go** (`sample_v4`, HC903: hc142's
three repetitions of sit, stand, walk 3 m, turn, walk back and sit, with a chair; the along camera faces the chair
from beyond the turn), a **hurried-pace walk** (`sample_v5`, HC904: hc105 at speed with a co-walker, overshooting
the mat at both ends) and a **self-paced walk on the instrumented mat** (`sample_v6`, HC905: hc105 in HC902's room,
with the recording's walkway columns fabricated from the truth in the export's own form, see above). Tandem gait has no public IK
sequence to render from. All three paired, synced and selected without a config edit (sync within 7 ms of the
truth; the TUG's selection r = 0.75 and 0.59, the hurried walk 0.54 in the along view with the side view attached
by the cross-view cue, the mat walk 0.51 and the cue), and lifted and merged like the others:

| session | export | legs (median) | knee-flexion RMSE | root | two views' leg disagreement |
|---|---|---|---|---|---|
| TUG (HC903, hc142) | 100 % | 15.7 deg | 10.1 deg | 15.4 deg | 9.8 deg |
| hurried pace (HC904) | 62 % | 12.1 deg | 8.6 deg | 9.1 deg | 9.7 deg |
| mat walk (HC905) | 99 % | 10.7 deg | 7.2 deg | 7.0 deg | 8.5 deg |

The TUG walker is hc142: the first render used hc100's TUG, whose inertial fit carries its left ankle 163 deg off
(a visibly wrong foot in the rendered truth, and 26 deg of "error" that belonged to the truth), while hc142's fit
passes the ankle check (median ankle angles 10 deg, knees to 85 deg when seated). The TUG's errors are the largest
of the set and they sit in the chair: by phase, its walking frames are at 13.2 deg on the legs (1 % over 20 deg)
and 9.7 on the root, like the walks, its seated and rising frames at 19-20 deg on the legs (38-55 % over 20) and
24-25 on the root. The merge under-reads the seated knee by 5 deg, and the side view, which is the reference there
(the along view has the seated legs in frame on half of those frames), sees the chair obliquely and close and
carries the pitched gravity frame described above. The hurried walk is the hard case it was built to be: the side
view's track holds the walker for two thirds of the recording (legs usable 67 %; the standing start is undetected
at the panel's edge, the walk bouts are covered but for 3 % of their frames), and at the final turn that view's
lift turned the wrong way and ran out of frame mid-turn, the case that gave the root rule its departure clause; the
along view holds the legs throughout (99 %) and the two views share 686 exported frames with their legs usable in
both. Its export is 62 % of the frames, and what is left out is the standing start: for the first 12 s the walker
waits at the near end of the mat, in the along camera's picture from the chest down and undetected in the side
view, so no view has the torso and the rule drops those frames (the walk and both turns are exported; the errors in
the table are larger than the 10.4 deg on the legs that the same merge has with the standing frames counted,
because standing is the easy part).

### How good these scores are, and what would improve them

Against the rendered truth the merged legs are 9-12 deg off at the median on the walks and 16 in the TUG (a
three-axis rotation error per joint, which is a stricter measure than a flexion angle), with a knee-flexion RMSE of
6.6-10 deg. These are the lifter's numbers more than the merge's. On the frames where both views have the legs, the
merged legs are below both views on five sessions of the six (by 0.2-1.2 deg under the better one) and 0.5 deg
above the better one on the sixth (HC905), and the merged knee RMSE is below both views or within 0.7 deg of the
better one. What is left is in each view's lift, and three measurements say where.

**Amplitude.** The slope of the estimated knee flexion on the true one is 0.63-0.71 on the walks (0.86 in the TUG),
and of hip flexion 0.47-0.59: the lifter draws every stride toward an average one, so ranges of motion read at
about 0.7 of their size. Each view alone has that slope (on the walks, HC901's truncated along view aside, knee
0.64-0.79 and hip 0.42-0.69), and the merge lies between its two views. There is no bias on the walks (under 1
deg), and the under-read amplitude is 35-50 % of the knee's squared error.

**Which view sees what.** The side view has the better slope on hip flexion (0.64-0.69 against the along view's
0.42-0.55): it sees flexion in its image plane, the along view sees it as depth. Weighting each view's leg joints
by how face-on it sees the sagittal plane was tried on the six sessions: the knee-flexion RMSE fell by 0.3 deg on
average (6.6 to 5.8 on HC900, worse by 0.2-0.4 on the TUG and the hurried walk), the hip slope rose by 0.03, and
the three-axis leg error rose by 0.6 deg, the along view being the better one on the other two axes. It is not
shipped; the gain is there for a combination per axis rather than per joint.

**Timing.** Scored against the truth one frame later, the knee RMSE of HC900 is 5.9 instead of 6.6 (HC901 6.2
instead of 6.6, the others within 0.3): half a frame of timing between the video and the sensor clock is worth up
to 0.7 deg on this measure.

What would move the scores, largest first: (1) the cameras' real fields of view at the lift, in place of the
default focal length (the heading term above, and a perspective the body is solved under; a re-lift); (2) one fit
to both views in place of two single-view lifts: the two cameras stand nearly at right angles, each has the
walker's 2D keypoints to the pixel, and a fit to both removes the depth ambiguity that is behind the amplitude and
behind the tilted root (it needs the cameras' relative pose, which the mat gives); (3) the full-rotation alignment
of the views, for the root (14.8 and 15.4 deg where the side view holds it); (4) the per-axis weighting above. The
first two are changes at the GPU stage, the last two in the merge.

### When there is no truth: a real clip

Every panel above that says "truth" exists because the walker was rendered. On a real recording the demo drops
them and shows what can be checked instead: each view's own lift beside the reconstruction and the export (so the
hand-overs are visible), the path from above, and in place of the truth curve the two views' disagreement on the
leg joints where both see them. The bundled example of our own clip (`examples/bundled`) is the test case for this
mode, lifted on a T4 like the rest:

![demo without a truth](docs/figures/demo_no_truth.png)

*One frame of `wgvp demo` on the two-view composite of the bundled phone clip (OWN02, 393 frames, 100 %
exported): the left panel is marked as the mirror image it is and shown reflected back, each view's own lift
stands beside the reconstruction and the export, the path from above replaces the aerial truth, and the bottom
chart is the two views' leg disagreement (5.4 deg median) in place of an error against a truth. On real video the
close-ups show a floor and no mat (the mat is drawn only when the room is known, i.e. for a rendered truth), and a
leg pair the 2D detector put on one leg is drawn in amber: on the rendered sessions that happens on 0.1-23 % of a
view's frames (a side-on walker's legs cross once a stride) and the lifter's leg error on those frames is 8-19 deg
against 9-25 elsewhere, so the merge does not weight them down. The single-view clip (OWN01: 95 % exported, two
runs with a 0.7-s gap where the walker turned at the far end) renders the same way with one view.*

The two-view composite of the same clip (OWN02) is built the way a careless second camera would be: the left
panel is the clip zoomed in a little, the right panel the clip **mirrored** and zoomed in hard (1.16x and 1.64x,
different letterboxing; head and feet leave the right panel when the walker is close), which makes the right
panel's left leg the walker's right leg. On its lift the merge's mirror check reads the knees 44 deg apart as
lifted and 8 deg apart with the sides swapped, reflects that view back, and the two "views" then agree on the legs
to 5.4 deg with their knee curves on top of each other (root residual 8 deg; the pelvis positions still differ by
0.54 m, the depth estimate being what two zoom levels disagree about). It is still one camera (the two panels
carry the same information), so the synthetic sessions above remain the test of the merge itself; what this clip
tests is the panel detector on two zoom levels, the per-view scale, the mirror detection, truncation and the
no-truth layout. The truncation found a limit worth knowing: on the frames where the walker's feet are below the
hard-zoomed panel, ViTPose still reports ankles with confidence 0.9 some 200 px inside the panel, so the
usable-joint rule (confident and inside the panel) does not notice, and the lifter extrapolates the legs there;
the two views' agreement factor is what discounts those frames. A geometric truncation test (the body's box
against the panel edge) is the open item.

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
   <recording>/merged_smpl.npz        after GVHMR: THE TRAINING TARGET, one pose per frame from the two views (see "The training target" above)
   <recording>/merge.json             its numbers: export fraction, continuity, view alignment, body shapes, flags
   <recording>/qc.json                walk-bout coverage per view, flags
   <recording>/<view>/subject_track.npz   frame_idx, pts_ms, bbox_xyxy, kp2d (COCO-17, face points blanked), visibility, source fragment id
   <recording>/<view>/smpl_global.npz     GVHMR SMPL-X parameters per frame, gravity-aligned world frame: global_orient (T,3), body_pose (T,63), betas (T,10), transl (T,3), frame_idx, pts_ms, imu_t_s,
                                          visible, edge_truncated, pose_step_deg, chunk_id (per-frame quality fields, see "Partial bodies, exits and re-entries")
   <recording>/<view>/smpl_incam.npz      the same in the camera frame
   <recording>/<view>/kp2d_vitpose.npz    GVHMR's ViTPose 2D keypoints and boxes, camera intrinsics K
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
| `merge_status`, `export_frac` | the merged training target: PASS / WARN with the reason in `flags`, and the share of lifted frames its export keeps |
| `status` | PASS / WARN (chosen but unconfirmed, or a cross-check disagrees) / FAIL (nothing chosen, no sync, lifter failed), with the reason in `flags` |

Every stage writes its record; failures are explicit messages that say what to do (`pick-subject`, `set-roi`,
`check-gpu`, the config key to edit).

## Command line

```
python -m wgvp                         GUI
python -m wgvp scan <root>             find sessions / recordings / videos; write wgvp_config.json
python -m wgvp run <root> [--cpu-only] [--only <session> ...] [--force] [--gvhmr-root .. --gvhmr-python ..]
python -m wgvp gvhmr <root>            the GPU stage alone, on pending chunks
python -m wgvp merge <root>            rebuild the merged training target (merged_smpl.npz) from existing lifts; no GPU
python -m wgvp demo <root> [--session <id>] [--recording <id>] [--gt <sample>_gt.npz] [--t0 S --t1 S] [--gif FILE]
                                       the demo video from the pipeline's outputs (demo extras; written under work/)
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
truth, never against a real mat under a real video. The decisive number,
the subject's `r_fine` in real footage, comes out of the first real run and is printed in the QC summary. [docs/design.md](docs/design.md) section 3 lists everything, and
section 4 the questions to settle with a dataset's owners before that run.

## Repository layout

```
wgvp/            the package: discovery, video_io (panels, cutting), tracking, signals, imu_ref, selection, sync,
                 flash, labels, reconcile, fusion + merge (the training target), demo, gvhmr_stage, qc, summary,
                 session (orchestration), cli, gui, dryrun
weights/         yolov8n-pose.pt (ultralytics, AGPL-3.0)
sample/          the synthetic dry-run session (video + recording CSV + expected answers)
sample_v2/       a second synthetic session with two deliberately compromised views and the walker's ground-truth
                 3D pose, for the two-view reconstruction experiment (tools/two_view_fusion.py)
sample_v3/       a third room: another walker, a co-walker, the along camera at the far end (legs cut when close),
                 the side camera on the other side; ground truth saved as well
examples/        bundled/ (our own clip + a two-view composite of it, runs offline); the fetched public-footage
                 corpus: sources.json, fetch_examples.py, verify_examples.py; make_bundled.py
tests/           unit tests (python -m pytest tests -q)
tools/           make_sample.py (how the samples were rendered), make_partner_layout.py (the three sessions in a
                 Sub 00N/Video + IMU layout), make_readme_figures.py, two_view_fusion.py (the fusion experiment
                 against the truth), make_target_figure.py
env/             GVHMR environment notes; docker/ container recipe (untested build)
docs/            design.md (design, test results, open questions), figures/
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
your machine, and the sources are attributed in `examples/sources.json`. The bundled clip in `examples/bundled/` is
our own recording, published with the consent of the person in it, under the same licence as the code.

Mike Caiola, Credence Management Solutions, LLC - mcaiola@credence-llc.com
