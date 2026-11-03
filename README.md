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

![two views](docs/figures/two_views_tracking.png)

*Two frames of the synthetic session with the detected panels (green and orange boxes), every tracked person and,
beside each, that person's pelvis correlation with the IMU at the shared lag. The subject reaches 0.72-0.77; the
co-walker, who starts 0.8 s later, -0.01 to 0.14; the seated operator and the bystander about 0.05.*

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
  [PASS] sync offset within 2 frames of truth (pelvis cross-correlation): estimated 3.009 s vs true 3.0 s ...
  [PASS] left: chosen track is the rendered subject: 99% of 1442 frames within 36 px of the GT hip ...
  [PASS] right: the subject's exits are recorded as gaps (no bystander attached)
  ...
DRY RUN PASS (11/11 checks)
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
stages. The CPU stages can be run now and the GPU stage later (`python -m wgvp gvhmr <root>`).

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

Only the self-paced family is processed by default (`selfpace`, `selfpace_mat`, `selfpace_matturn`); add other
tasks in `wgvp_config.json` under `tasks`. A video is paired with every recording CSV of its task family for that
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

*A: inside a walk bout the subject's video pelvis signal (green, both views) follows the accelerometer (black); the
co-walker's (red) does not. B: correlation versus lag for every scored person; the subject's fragments peak at the
shared lag of 3.01 s (true 3.00 s). C: coverage per view on the IMU timeline; the side view's two exits become gaps.*

Correlations are judged by their **excess over an empirical chance level**, not by their raw value. Short fragments
reach high correlations by chance (a 4-s fragment reaches r = 0.73 over a full-range lag search; a 45-s one 0.30),
so the acceptance rule is r >= 0.30 *and* excess >= 0.15, and a "high" confidence further needs r >= 0.35, excess
>= 0.20 and a margin >= 0.12 over the best other person at that lag. On crowded public footage with a simulated
reference, subjects visible for 8 s or more were found in every tested case among 300-400 other walkers, 4-8-s
fragments about half the time (otherwise mostly declared low-confidence), with 11 ms median offset error
([docs/design.md](docs/design.md), section 2.4). A clinic recording gives 20-45 s of the subject per view with a
stand / walk / turn pattern, which is the easy regime.

## Where tracking goes wrong, and what the pipeline does about it

Multi-person tracking in a room fails in known ways. Each one is handled, and each leaves a trace in the QC so you
can see it happened.

| Failure | What the pipeline does | Where you see it |
|---|---|---|
| The tracker swaps identities when two people cross or pass close | Tracker ids are split into **fragments** at box jumps, time gaps and persistent appearance changes; fragments are judged one by one and re-joined only when they agree on the shared lag | `n_fragments`, `subject_tracks` (the accepted fragment ids) |
| An identity switch inside one fragment (the subject and a co-walker crossing) | A **lag-consistency mask**: the fragment's pelvis signal is correlated with the IMU in 10-s windows at the shared lag; windows below r = 0.25 are dropped | `consistency` in `selection_joint.json` (kept fraction per fragment) |
| The subject leaves a view and someone else is in frame | Exits are recorded as **gaps**; nothing is attached while the subject is out, and a scorable fragment that contradicts the lag is never attached by appearance. Absences over 2 s end the lifter chunk; shorter ones are bridged and flagged `visible = False` | `n_gaps`, `present_frac`, `gaps_imu_time_s`; the dry run asserts "no bystander attached"; `chunk_id` / `visible` per frame |
| Brief re-entries too short to score | Attached by torso-colour appearance only when clearly closer to the subject than to any concurrent person | `attached_by_appearance` |
| Two detection boxes on one person (partial bodies in the along-walkway view) | Recognised by box overlap and not counted as a rival | role `duplicate_of_subject` in the fragment table |
| Someone walking in step with the subject (a clinician alongside) | The pelvis match is sub-frame; a person 0.8 s behind correlates 0.03-0.17 at the shared lag. This is the one configuration where the margin gets thin | `margin`, `lag_ambiguity` (> 0.85 means another lag was almost as well supported) |
| A seated operator large in frame, a bystander at the far wall | No gait-band pelvis oscillation, so they score about 0 whatever their size | their `r_at_lag` in the fragment table |
| Short spurious fragments chaining up at a wrong lag with "high" confidence | Every correlation is judged by its excess over the chance level for its length and the number of lags searched; support = excess x overlap, so one long genuine match outweighs a chain of short ones | `excess`, `second_best_excess` |
| The lifter follows the largest or longest person instead of the subject | GVHMR is driven with the subject's own bounding-box track (`preprocess/bbx.pt`), cut per view and per visible stretch | `gvhmr_frames` versus `present_frac` |
| The two views disagree after all this | Reconciliation on the IMU timeline: residual lag of the 2D pelvis signals (always) and per-joint disagreement of the SMPL-X poses (after GVHMR), tolerance 3 frames | `reconcile_status`, `residual_lag_s`, `legs_median_deg` |
| The automatic choice is wrong or absent | Two operator tools: `set-roi` (draw the walkway polygon once per view) and `pick-subject` (click the subject on a skeleton-only frame); both are reported with their agreement with the automatic choice | `selection_source` = `walkway_roi` / `operator_pick`, status WARN |
| Variable frame rate, dropped frames | All timing is done on presentation timestamps, never on frame counts | `timing` in the video record (gaps, VFR flag) |

Two of these rules were learned the hard way while testing (the perspective term in the pelvis signal, and the
chance level of short fragments); [docs/design.md](docs/design.md) section 2.6 has the numbers before and after.

## The GPU step and the two-view agreement

GVHMR lifts each view independently; the pipeline only hands it the subject's cropped frames and boxes, converts
its output to NPZ on the IMU timeline, and then compares the two views, which share nothing but the sync.

![mesh](docs/figures/gvhmr_two_views_mesh.png)

![agreement](docs/figures/gvhmr_two_views_agreement.png)

*On the synthetic session, lifted on a Tesla T4: the two views' knee-flexion curves line up on the IMU timeline
(residual lag -46 ms, within the 3-frame tolerance) and the leg joints disagree by a median of 5.9 degrees. Body
shape differs between views, as expected from a per-frame estimate when one view truncates the body.*

GVHMR takes about 0.5 s per frame per view on a T4 (12-15 min for a 50-s view). Its weights and the SMPL / SMPL-X
body models are downloaded by you under their own licences (registration required for the body models; they are not
in this repository or the Docker image). `docker/` holds a CUDA 12.1 recipe pinned to the versions we ran; its build
is untested because our GPU box had no disk left for it.

## Partial bodies, exits and re-entries

A gait-lab camera at the end of the walkway cuts the head off when the participant is close, and a side camera
loses the participant at both ends of the walkway. A lifter run naively on such footage either crumples the body
or jumps to whoever else is in frame. What the pipeline does, and what the synthetic session measured on a T4:

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

These numbers come from rendered bodies. GVHMR on real clothed, truncated people is on the list of things only the
first real run can show.

### 3D reconstruction from two partial views

The real session layout is harsher than the dry-run sample: the along-walkway camera sits at the end of the mat,
so the participant fills and overflows the frame when close, and the side camera's framing cuts the head. To ask
whether two such views still give back one correct 3D pose, `sample_v2/` renders the same walk from an along view
that keeps only the legs when the walker is near (hips out of frame 60 % of the time) and a side view aimed at
hip height (head above the frame in 77 % of frames), and saves the walker's true SMPL pose and 3D joints beside
it. `tools/two_view_fusion.py` then measures, against that truth, each view's GVHMR error per joint and by what
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
agree to about 10 deg, so a larger gap means one of them is wrong. Every term is continuous, so nothing in the
rule can step the root (`tools/two_view_fusion.fuse_root`). The same share then scales that view's weight for all
of its body joints, since a lifter that got the gravity frame wrong solved the whole body in it. This takes the
fused root to 6.6 deg (the views: 49.6 and 7.3) with a frame step at the 99th percentile of 5.4 deg (the side
view alone 7.3), and the merge below the better view in every column: legs 10.0 deg, outliers 8 %, all 21 joints
11.4 deg, knee-flexion RMSE 7.4 against the side view's 8.2 (last segment-rule row of the table), at 98 %
coverage and steps of 4.8 deg at the 99th percentile.

What an export keeps is stricter than what the reconstruction shows. The red parts are filled only so the mesh
can be drawn; the export (`export_mask`) keeps a frame only when every leg joint is output, in runs of at least
0.5 s, and on those frames contains only the joints some view saw. On HC901 that is 94 % of the IMU span (the
along view alone has the legs for 37 % of it, the side view for 91 %).

![two-view demo](docs/figures/two_view_demo.gif)

*Fourteen seconds of HC901 (`tools/make_demo.py`, full video in the results of the run). Top: the two views with
ViTPose's keypoints for the chosen subject (the lifter's input; the left/right labels of the arms and legs are
kept consistent from frame to frame for the drawing, because 2D estimators flip sides on a side-on walker, and
the 3D stage does not use them) and what each view can use; the bright square in the side view is the sample's
sync light, lit while the IMU records, which the flash cross-check reads. Middle, left to right: the rendered
truth on the mat; the reconstruction (red = a part no view could see, filled only so the mesh can be drawn; at
35-38 s the along view holds only the legs and the upper body GVHMR invents from them is exactly what gets
excluded); the export, in which such parts are absent and a dropped frame shows the empty mat; and the whole room
from above: mat, walker, the operator at the desk, the bystander, the along camera (blue) and the side camera
(orange). Bottom, the readout in two forms (`--readout strips|charts|both`): a filmstrip with a small render
every 2 s of what each view outputs (joints it cannot see in red), the merge and the truth, with the well-tracked
blocks marked at frame resolution by a bar under each row and a box around the consecutive thumbnails inside one
block (a view: its legs usable; the merge: the frames the export keeps; thumbnails outside a block are dimmed);
then the same as timelines on the shared IMU axis: each view's knee-flexion output (blue, orange), the merge
(green, with gaps where no view saw the joint) and the truth (white). The side view leaves the frame at 29-32 s;
the merge follows whichever view can see and hands over without a step.*

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
   <recording>/qc.json                walk-bout coverage per view, flags
   <recording>/<view>/subject_track.npz   frame_idx, pts_ms, bbox_xyxy, kp2d (COCO-17, face points blanked), visibility, source fragment id
   <recording>/<view>/smpl_global.npz     GVHMR SMPL-X parameters per frame, gravity-aligned world frame: global_orient (T,3), body_pose (T,63), betas (T,10), transl (T,3), frame_idx, pts_ms, imu_t_s,
                                          visible, edge_truncated, pose_step_deg, chunk_id (per-frame quality fields, see "Partial bodies, exits and re-entries")
   <recording>/<view>/smpl_incam.npz      the same in the camera frame
   <recording>/<view>/kp2d_vitpose.npz    GVHMR's ViTPose 2D keypoints and boxes, camera intrinsics K
   <recording>/<view>/gvhmr_meta.json, selection.json, qc.json
   validation_bundle/  logs/, thumbnails/ (skeleton-only PNGs), panels/ (edge-map previews with the view boxes), environment.json, privacy_check.json, metrics.json
<session>/work/     intermediates that stay on the machine (all-person tracks, cut clips, GVHMR dumps)
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
| `status` | PASS / WARN (chosen but unconfirmed, or a cross-check disagrees) / FAIL (nothing chosen, no sync, lifter failed), with the reason in `flags` |

Every stage writes its record; failures are explicit messages that say what to do (`pick-subject`, `set-roi`,
`check-gpu`, the config key to edit).

## Command line

```
python -m wgvp                         GUI
python -m wgvp scan <root>             find sessions / recordings / videos; write wgvp_config.json
python -m wgvp run <root> [--cpu-only] [--only <session> ...] [--force] [--gvhmr-root .. --gvhmr-python ..]
python -m wgvp gvhmr <root>            the GPU stage alone, on pending chunks
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
exercised only on synthetic inputs. The decisive number, the subject's `r_fine` in real footage, comes out of the
first real run and is printed in the QC summary. [docs/design.md](docs/design.md) section 3 lists everything, and
section 4 the questions to settle with a dataset's owners before that run.

## Repository layout

```
wgvp/            the package: discovery, video_io (panels, cutting), tracking, signals, imu_ref, selection, sync,
                 flash, labels, reconcile, gvhmr_stage, qc, summary, session (orchestration), cli, gui, dryrun
weights/         yolov8n-pose.pt (ultralytics, AGPL-3.0)
sample/          the synthetic dry-run session (video + recording CSV + expected answers)
sample_v2/       a second synthetic session with two deliberately compromised views and the walker's ground-truth
                 3D pose, for the two-view reconstruction experiment (tools/two_view_fusion.py)
examples/        bundled/ (our own clip + a two-view composite of it, runs offline); the fetched public-footage
                 corpus: sources.json, fetch_examples.py, verify_examples.py; make_bundled.py
tests/           unit tests (python -m pytest tests -q)
tools/           make_sample.py (how the sample was rendered), make_readme_figures.py
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
