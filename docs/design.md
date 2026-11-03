# Design, tests and open questions

This document records how the pipeline works stage by stage, what was tested and with what result, what cannot be
tested without a partner's own recordings, and the questions to settle with the owners of a dataset before the
first real run. The README covers installation and use.

## 1. Design

Per video (CPU, one decode pass, cached by file hash):

1. **Panel detection.** 40 sampled frames; black-border pixels (dark in at least 50 % of samples); the seam between
   the two views is the line boundary with the strongest *persistent* edge (median over frames of the mean absolute
   difference across the boundary where both sides are content, relative to a typical boundary), tempered by the
   temporal co-fluctuation of the two sides (independent views do not co-vary) and with a bonus for a change of
   letterboxing; black separator bands between two content regions are explicit candidates. Horizontal versus
   vertical by the better score; otherwise a single view. Each panel's content box trims edge-adjacent all-black
   lines only, so a black walkway mat inside a view is not taken for a border. The boxes are written to the QC and
   drawn on an edge-map preview; `views.override` in the config forces boxes per file.
2. **Tracking per view.** YOLOv8n-pose + ByteTrack (one tracker per view), frames downscaled to 960 px for the
   detector, keypoints reported in full-frame pixels, with per detection a torso colour histogram (12 x 4 H-S bins,
   for re-identification only, kept in `work/`) and a visibility class (full: shoulders, hips and ankles seen;
   partial; none). Tracker ids are split into fragments at box-centre jumps, time gaps and persistent appearance
   changes.
3. **Subject selection (primary: pelvis cross-correlation, joint over both views).** Every fragment gets a
   perspective-robust pelvis signal: (hip row - fitted horizon row) / smoothed torso length, which removes the
   per-step term that walking toward a low camera leaves in the raw hip row. The IMU side is the recording CSV's
   `LowerBack_FreeAcc_U`, band-passed 0.5-5 Hz and integrated twice, so both sides are pelvis displacement. A
   two-scale match (RMS-envelope correlation for the coarse lag, i.e. the stand/walk/turn pattern, then the
   oscillation inside +-0.75 s with a parabolic sub-sample peak) gives a correlation-versus-lag curve per fragment.
   The video holds ONE unknown lag (the views were tiled from synchronised cameras), so the candidate lags from the
   best fragments of both views are evaluated jointly: at each lag, fragments with r >= 0.30 and an excess over the
   chance level >= 0.15 are accepted greedily without temporal overlap, the lag with the largest support wins, is
   refined on the combined curve, and brief re-entries too short to score are attached by appearance when they are
   clearly closer to the subject than to any concurrent person. Seated and standing people have no gait-band
   pelvis oscillation and score about 0 whatever their size; a second detection box on the subject (partial bodies
   in the along-walkway view) is recognised by box overlap and not counted as a rival. Confidence: best r >= 0.35 and
   a margin >= 0.12 over the best other person at that lag; lag ambiguity is reported as the support ratio of the
   runner-up lag. Secondary: a walkway polygon drawn once per view by the operator. Fallback: a one-click pick on a
   skeleton-only frame. All three are reported with their agreement; a confident primary wins, else the pick, else
   the polygon (WARN).
4. **Sync.** Offset = video presentation time of IMU t = 0 from the joint cross-correlation (sub-frame).
   Cross-checks, each reported with its difference and a PASS/WARN: the light flash if a persistent local brightness
   step is visible (operator rectangle or automatic grid search, global exposure removed), the Walk-interval
   correlation against an exported Video Labeler label file (video time) when present, and an annotation-sheet start
   frame when present. Variable frame rate and dropped frames are handled by working on presentation timestamps
   (the `timing` QC lists gaps).
5. **Coverage and gaps.** Per view: present / full / partial / absent fractions over the recording window, the gap
   list in IMU time, and for every annotated Walk bout the covered fraction; a recording needs at least one view
   covering a bout (>= 80 %), and the QC says which view covered which bout.
6. **Lifter inputs.** Per view, the subject's frames inside the recording window are cut from the panel (cropped,
   <= 1280 px wide, 30 fps - every second frame above 45 fps), split at absences longer than `absence_split_s`
   (2 s; a new chunk starts at the re-entry, nothing is lifted in between) and into <= 3000-frame chunks at IMU
   "Standing" moments, with the subject's own bounding-box track (interpolated across shorter gaps, which are
   flagged `visible = False` in the output) written as GVHMR's `preprocess/bbx.pt`, so the lifter follows the
   subject and not the longest track.
7. **GPU stage** (`wgvp.gvhmr_stage`, runs in the GVHMR environment): GVHMR demo per chunk, static camera, no QC
   renders (`GVHMR_RENDER=0`); `hmr4d_results.pt` converted to `smpl_global.npz` (gravity-aligned world frame) and
   `smpl_incam.npz` (camera frame) with `frame_idx`, `pts_ms`, `imu_t_s`, `visible`; ViTPose 2D keypoints mapped
   back to full-frame pixels; GVHMR's input copies and overlays deleted. Chunks are merged at the overlap midpoint;
   every frame carries `visible`, `edge_truncated` (the subject box touches a view edge), `pose_step_deg` (largest
   joint-rotation step from the previous frame) and `chunk_id`.
8. **Reconciliation between the views** on the IMU timeline: correlation and residual lag of the 2D pelvis signals
   (always) and, after GVHMR, per-joint geodesic disagreement of the SMPL-X body pose, knee/hip flexion correlations
   and the knee-signal residual lag (tolerance 3 frames); the per-frame disagreement is written to
   `pose_agreement_<a>_<b>.npz`.
9. **QC and privacy.** Every stage writes JSON; skeleton-only thumbnails (black background, green = subject) at the
   sync frame and along the video; `QC_SUMMARY.html` + `qc_summary.csv` across sessions with the panel previews
   (edge maps); `privacy_check.json` asserts no video/image file, no image-like array, only sparse skeleton PNGs in
   `deliver/`. Only the subject's track is exported (face keypoints blanked); all-person tracks, cut clips and GVHMR
   dumps stay in `work/`.

Failure modes are explicit messages: no video matched (and why), panel detection fell back to a single view, no
fragment reaches the acceptance correlation (-> `pick-subject` / `set-roi`), sync from a cross-check only, flash or
labels disagree, subject not visible inside the recording window in a view, GVHMR missing weights / CUDA / failed
chunk (log path), non-finite SMPL output, privacy problems.

## 2. Test results

### 2.1 Unit tests (CPU, anywhere)

`tests/`: cross-correlation sign and per-overlap Pearson normalisation; offset recovery at +7.3 / -3.0 / +12.345 /
0 / +25 s with 0-40 dropped frames (errors < 20 ms, r > 0.8); a gait-matched decoy with the same activity pattern
scores > 0.4 lower; double-integration phase; light-step detection and rejection of transient bumps; automatic grid
search under exposure drift; the selection and sync decision rules; two-view reconciliation with different frame
rates; the privacy check; folder-scan conventions (subject folders, task families, candidate videos, label export,
unmatched files); panel detection on a 520 + 900 x 500 letterboxed frame, a stacked layout with a black separator
and side bars, and a single letterboxed view; the fabricated example recording read back through the IMU reader.

### 2.2 Panel detection on the real layout

A 3-s video synthesised from a screenshot of a real session in MATLAB's Video Labeler (the 1420 x 595 video area,
each panel given its own small motion): seam found at x = 513 (score 2.6, cross-seam co-fluctuation 0.06,
persistent-edge ratio 13.5), left box [0, 31, 513, 594], right box [513, 126, 1420, 494]: the right panel's top and
bottom letterbox bands and the dark walkway mat inside the left panel are handled. Bundled sample: left
[0, 0, 519, 600] versus a true 520, right [519, 0, 1420, 500] versus [520, 0, 1420, 500].

### 2.3 Dry run on the bundled sample (CPU stages; the GPU stage is in 2.5)

Sample = one 1420 x 600, 30 fps, 53-s combined video rendered from the public WearGait recording hc100_selfpace
(IK poses walking a 4.9-m corridor; pelvis vertical motion = the recording's own double-integrated
LowerBack_FreeAcc_U, so the video and the CSV describe the same walk), a co-walker 1.5 m beside the subject started
0.8 s later with its own IMU motion (hc102), a seated operator fidgeting 2.9 m from the side camera (large in
frame), a bystander at the far wall, a sync light in the side panel, 3.0 s of pre-roll; along-walkway view with the
head cut off when close (878 partial-body frames of 1459), side view that the subject leaves at both ends (107
absent + 88 partial frames). Result: **PASS, 11/11 checks** (81 s for 1594 frames x 2 views on an Apple M-series
CPU; 50 s on a Tesla T4):

* panels: exact layout, boxes within 1 px;
* sync: offset 3.009 s versus 3.000 s true (9 ms = 0.3 frame), light cross-check agrees within 9 ms, source
  `pelvis_xcorr(validated by flash)`;
* subject: along view fragment [3], 99 % of 1442 frames within 36 px of the ground-truth hip (median 21 px),
  r 0.77, margin 0.64; side view fragments [2, 8, 13] merged across the two exits, 100 % of 1300 frames (median
  10 px), r 0.74, margin 0.56; 0 frames claimed while the subject is out of a view; co-walker, seated operator and
  bystander rejected; runner-up lag (the co-walker's timing) at 22 % of the winner's support;
* coverage: along 0.99 present (0.97 full / 0.02 partial - the tracker still reports hips for most truncated
  frames), side 0.89 present with the two exits reported as gaps at IMU 28.6-31.9 s and 42.3-44.3 s; 4/4 walk bouts
  covered;
* the two views place IMU t = 0 within 5 ms of each other;
* privacy check: 27 delivered files, no problems.

### 2.4 Selection and sync on real multi-person footage (simulated IMU)

Two 30-s excerpts of a public plaza video (crowd; 404 and 272 track fragments) and two single-person phone clips
(4K, 13 and 40 s). Each fragment of at least 4 s with visible hips played the subject in turn: an IMU-like reference
was fabricated from that fragment's own pelvis signal (second derivative, white noise at SNR 2), placed at a random
offset within +-8 s, with 0-15 dropped video frames; then the same reference with the subject's fragment removed
(absent case, which must be rejected). This is the worst case by construction: everybody walks continuously (flat
activity envelope), fragments are short, and the reference is noise outside the subject's span.

| subject fragment length | n | correct | high-confidence | precision when high | median offset error | within 1 frame |
|---|---|---|---|---|---|---|
| 4-8 s (crowd) | 46 | 0.52 | 0.54 | 0.84 | 12 ms | 1.00 |
| 8-15 s (crowd) | 4 | 1.00 | 1.00 | 1.00 | 8 ms | 1.00 |
| >= 15 s (crowd) | 2 | 1.00 | 1.00 | 1.00 | 11 ms | 1.00 |
| single-person clips | 4 | 1.00 | 1.00 | 1.00 | 11 ms | 1.00 |

All 52 present cases: accuracy 0.58, high-confidence precision 0.87; offset error over the correct selections
11 ms median, 18 ms at the 90th percentile, 1.00 within one frame. Absent-subject references rejected (no
high-confidence pick): 0.90 of 50 (5 false high-confidence picks, all with 4-8-s subjects in the crowd).

Reading: from 8 s of visible walking the right person was found in every tested case, even among 300-400 other
walkers with a reference that has no stand/walk structure; 4-8-s fragments were found half the time and otherwise
mostly declared low-confidence. A clinic recording gives 20-45 s of the subject per view with a stand/walk/turn
pattern, the regime of the dry run (r 0.74-0.77, margins 0.55-0.64). Sync precision when the person is right is
sub-frame (11 ms median at 30 fps). The `examples/` corpus repeats a small version of this test on footage anyone
can fetch.

### 2.5 GPU stage and end-to-end run

On a Tesla T4 cloud instance (Ubuntu 24.04, Python 3.10, torch 2.5.1+cu121, ultralytics 8.3.80, GVHMR commit
088caff4): `check-gpu` all items OK; tracking 1594 frames x 2 views in 50 s; panels exact; sync 3.007 s versus 3.000 s
(7 ms), flash agrees; subject: along view r 0.85, margin 0.67, 99 % of 1387 frames on the ground-truth hip; side view
r 0.82, 99 % of 1301 frames (median 10 px), exits as gaps, no bystander frames; runner-up lag (co-walker) at 45 %
support; the two views place IMU t = 0 within 19 ms. GVHMR: along view 1457 frames in 365 s, side view 1458 frames in
331 s (ViTPose and feature extraction dominate; the HMR network itself takes 1.5-1.8 s per view), output finite,
GVHMR's input copies deleted, 36 delivered files, privacy check clean. Pose-level reconciliation between the views on
the IMU timeline (47.8 s overlap): median geodesic disagreement 8.1 deg over the leg joints (hips 6.5 / 8.9, knees
9.1 / 11.3, ankles 4.8 / 7.4 deg), 8.0 deg over all 21 joints, knee-flexion correlation 0.55-0.60 between views,
knee-signal residual lag -73 ms (2.2 frames; tolerance 3 frames, so `reconcile.json` reads PASS). `betas` differ
between views (the along view sees truncated bodies), as expected from a per-frame shape estimate.

### 2.6 Two findings that changed the design during testing

* **Pelvis signal.** The first version divided the hip row by the per-frame torso length. In the along-walkway view
  the forward progression leaves a term proportional to 1/distance that oscillates with every step, and in every
  view the per-step modulation of the torso length and a zero-padded smoothing at fragment ends produced strong
  correlation peaks at stride-aliased lags (the subject's own side-view fragment: r 0.30 at the true lag, 0.73 at
  +3.3 s). The shipped signal is (hip row - fitted horizon row) / torso length smoothed over 2 s with edge padding,
  one scale reference per track: r 0.77 / 0.75-0.80 at the true lag with the alias 0.05 lower in both views, and
  the real-footage accuracy rose from 0.27 to 0.58 (8-s-plus subjects from 0.83 to 1.00) with the same thresholds.
* **Chance level.** 680 unrelated fragment/reference pairs from the plaza tracks give a 95th-percentile fine
  correlation of 0.82 / 0.73 / 0.65 over a full-range lag search and 0.52 / 0.52 / 0.33 inside the acceptance window
  for overlaps of 1.5-3 / 3-5 / 5-8 s (band-limited theory with 6 independent samples/s predicts 0.99 / 0.92 / 0.73
  and 0.73 / 0.54 / 0.43). The selector judges every correlation by its excess over this empirical curve (extended
  with the 1/sqrt(T) theory for long overlaps, blended by the number of independent lags searched, raised for the
  number of competing fragments): accept at r >= 0.30 and excess >= 0.15, "high" at r >= 0.35, excess >= 0.20 and a
  margin >= 0.12 over the best other person; the operating point was chosen on the same cached footage (min_excess
  0.10-0.18 x window 0.08-0.12 s: accuracy for 8-s-plus subjects constant, precision 0.71-1.00, absent rejected
  0.86-0.98). With raw r the crowd test chained short spurious fragments at wrong lags with "high" confidence
  (accuracy 13 %, absent cases rejected 36 %).

## 3. What has not been tested on real clinic recordings

* The real seam / letterbox geometry was tested only on a video synthesised from one screenshot (static scene with
  injected per-panel motion) and on the rendered sample; real encoder noise, camera jitter or a third panel are
  untested. The GUI preview and `views.override` are the fallback.
* YOLOv8n-pose on clinic lighting, clothing, black sensor straps and de-identification blur (if faces are blurred,
  the face keypoints are irrelevant; the selection uses the pelvis). Partial bodies in the along-walkway view were
  tested on rendered meshes, not on real truncated people.
* The pelvis correlation on real video against a real IMU: the sample's pelvis motion is the recording's own
  double-integrated accelerometer imposed on a rendered body, so the match is consistent by construction; real
  footage was tested only with a simulated IMU made from the chosen person's own track. The decisive number, the
  subject's `r_fine` in real clinic video, comes out of the first real run and is printed in the QC summary.
* GVHMR on truncated bodies and on other GPUs/drivers; the Docker build; the tkinter GUI on Windows (built and
  scanned only on Linux under Xvfb).
* The Walk-label cross-check (tested on synthetic intervals), the annotation-sheet cross-check, and whether a sync
  light is visible in a combined video at all.
* Throughput on other hardware (ours: CPU tracking 4-20 frames/s per view on an Apple M-series laptop for a
  1420 x 600 file; GVHMR about 0.5 s per frame per view on a T4).

Two cautions for a first real run: (1) the acceptance thresholds were set on crowded public footage and a rendered
sample, and the co-walker case shows that a person walking in step 0.8 s behind the subject is the one configuration
where the margin gets thin (0.27-0.55 here): `lag_ambiguity` and the thumbnails are the check; (2) the Walk-label and
annotation-sheet cross-checks have only been exercised on synthetic inputs.

## 4. Questions to settle with the owners of a dataset before the first run

1. **File naming and layout.** Is `<subject>_<task>_<tags>` the convention for every session, and what do the tags
   mean (frame rate, rotation, export index)? Is the folder structure subject > task > one video, with the recording
   CSVs beside it, for every task, i.e. one video per CSV or one video per visit? (`wgvp_config.json` holds all the
   matching rules.)
2. **De-identification.** What was done to the frames (blurred faces, cropping)? Does a blur move with the person?
3. **Video format.** Container/codec, resolution of the combined frame, frame rate (constant or variable), duration
   per task, and whether the two panels are always in the same positions with the same sizes and letterboxing.
4. **Cameras and sync.** Positions of the two cameras, how they were synchronised before tiling, whether a sync light
   is visible in the combined video, and any known latency between the light and the IMU recording start.
5. **Annotation.** Were the GeneralEvent/ClinicalEvent labels made in the Video Labeler on this combined video and
   exported to the IMU timeline? Can the scene labels be exported (two MATLAB lines in the README)? Does any sheet
   store the video frame of the recording start?
6. **People in the room.** How many staff are typically visible, where do they sit, and does anyone walk alongside
   the participant (assistance, walking aid)? Are the sensor straps visible on the limbs?
7. **Compute and IRB.** OS, Python availability, an NVIDIA GPU with >= 8 GB and CUDA 12 for GVHMR (or Docker with the
   NVIDIA runtime), disk space (about 1 GB of intermediates per session), and whether the IRB accepts pose-level
   derivatives (SMPL parameters, 2D keypoints without face points, skeleton-only thumbnails, edge-map panel
   previews) leaving the site; whether `betas` (body shape) may be included.
8. **Participants partly out of frame.** How far does the walkway extend beyond the along-walkway frame, and how
   long are participants out of the side view at the ends? The pipeline reports both; typical values set
   expectations.
