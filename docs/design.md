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
   runner-up lag. Cross-view confirmation (2026-11-05): a view where no fragment reaches the pelvis match (the
   walker a hundred pixels tall in a room-wide side view: the bob is under a pixel, below the keypoint noise, at
   640 and at 1280 px of detector input alike) takes its subject from the confirmed view's by the correlation at
   zero lag, on the shared video clock, between progress-along-the-walk signals (the hip column; the log of the
   body's pixel width; the best of the four pairings, sign-free), |r| >= 0.8 over >= 5 s, accepted best first
   without temporal overlap; the margin against another person moving with the subject (a co-walker 0.8 s behind
   correlates nearly as well on a 20-s walk cycle) decides PASS or WARN (`selection.cross_view_selection`).
   Secondary: a walkway polygon drawn once per view by the operator. Fallback: a one-click pick on a
   skeleton-only frame. All three are reported with their agreement; a confident primary wins, else the pick, else
   the polygon (WARN).
4. **Sync.** Offset = video presentation time of IMU t = 0 from the joint cross-correlation (sub-frame).
   Cross-checks, each reported with its difference and a PASS/WARN: the light flash if a persistent local brightness
   step is visible (operator rectangle or automatic grid search, global exposure removed), the Walk-interval
   correlation against an exported Video Labeler label file (video time) when present, and an annotation-sheet start
   frame when present. Two sanity rules (2026-11-04, after the hc102 case in 2.8): the offset must leave at least
   60 % of the recording inside the video, and a strong (z >= 100) persistent light-like step more than 3 s from
   the offset is reported as a disagreement even though no flash was used; either makes the sync a WARN. Variable
   frame rate and dropped frames are handled by working on presentation timestamps (the `timing` QC lists gaps).
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
9. **Merge: the training target** (`wgvp.merge`, CPU, after the lift; `wgvp merge <root>` re-runs it alone). Both
   views come from one tiled video, so they share frame indices; the merge covers every frame some view lifted. Per
   view and frame the joints it can vouch for are those whose ViTPose keypoints are confident and inside the panel
   (`fusion.usable_joints_rt`). The view with more usable joints is the reference; the other view's GVHMR world is
   turned onto it by one yaw (circular median of the two roots' heading difference) and one offset (median
   translation difference), fitted on frames where both views see most of the body, with the residuals reported.
   Root: `fusion.fuse_root` (presence-ramped, usable-fraction-weighted; when the roots disagree by more than
   30 deg, a view turning faster than 150 deg/s since its previous frame and a view far from the previous fused
   root are discounted first (2026-11-05: a lift taking a 180-degree turn the wrong way round), then the weights
   are sharpened towards the heavier view);
   translation averaged with the same shares and smoothed over 5 frames inside runs. Body: per joint the view(s)
   that vouch for it, cross-faded over 0.5 s at segment edges, each view weighted by its root share, nothing where
   no view saw the joint, smoothed over 5 frames. Export: every leg joint and every torso joint (spine, neck, head) output, in runs of at
   least 0.5 s; an arm may be missing. For
   learning from body-worn sensors the root is also written heading-relative (yaw removed; velocity in the heading
   frame) and one body shape per recording is taken as the median of GVHMR's betas over whole-body frames. Loss
   weights in [0, 1]: per joint, the summed view weights (one confident view or two agreeing ones: 1) times an
   agreement factor 1 / (1 + ((d - 5 deg)+ / 20 deg)^2) on the two views' disagreement d, d scaled by twice the
   smaller view's share of the root, which is how much of the dissenting view the merged joint contains (until
   2026-11-05 the whole disagreement counted: on HC901 the along view's standing-phase fragment, set aside by the
   root rule with its knees 50 deg wrong on a legs-only crop, pulled the knee weight of that phase to 0.2 while the
   merged knee was the side view's; it is 1.0 now, and the export-mean weight 0.76 -> 0.92); for the root, the summed
   raw root weights; for the translation, the same discounted by the two views' pelvis-position disagreement
   (10 cm free, 0.5 at 40 cm). Mirrored view (2026-11-05): before the alignment, every other view is tested
   against the reference for being its mirror image (a camera's picture flipped by its software): on frames where
   the reference's knees differ in flexion by more than 15 deg (one leg swinging) the median geodesic distance of
   the knees as lifted and after `fusion.mirror_body` (left/right joints swapped, each local rotation conjugated by
   the sagittal reflection: rotvec (x, y, z) -> (x, -y, -z); exact for the template, and valid for any vertical
   mirror plane since two such reflections differ by a yaw and a shift, which the alignment absorbs); mirrored when
   the swapped distance is under 0.6 of the as-lifted one and 6 deg smaller; the view's body, root (rotvec sign
   flip), translation (x -> -x) and usable-joint mask are reflected back, `view_mirrored` is written and a flag
   raised. The walkway (2026-11-04/05): the recording CSV's `Walkway_X/Y` are, at every 10-ms sample, the lists of
   loaded cells of the pressure mat ('x1|x2|...', cell indices on the GAITRite's 384 x 48 grid of 1.27-cm cells),
   with per-cell footprint labels in `WalkwayFoot` and levels in `WalkwayPressureLevel`; `imu_ref.parse_walkway`
   keeps the pressure-weighted centre of each labelled footprint per sample (`walkway_t/x_m/y_m/foot`) and groups
   them into stances (`walkway_stance_*`; one open stance per foot, since the other foot's samples interleave).
   The merge then (a) fits per view a depth-scale factor of its path about the path's own footfall-time centre,
   with the mat's yaw and offset in that view's world, to the footfalls against the landing foot (forward
   kinematics of the mean SMPL-X body, `fusion.fk_joints`; a fit is accepted under 0.5 m of residual), aligns and
   fuses the scaled views and writes `transl_scaled` / `walkway_scale`; (b) fits the mat's pose on the fused feet
   and builds `transl_mat`, the pelvis on the mat: per stance, the footprint's position minus the pose's
   foot-to-pelvis offset turned into the mat frame, both feet during double support, each stance's say ramped over
   0.12 s, gaps under 0.5 s bridged (`merge.mat_pelvis_path`); `transl_mat_video` is the video path in the mat
   frame and `transl_from_walkway` the mat path back in the world, which also feeds the heading-frame velocity
   where it exists. A rigid fit of each view's camera-frame onto its world-frame translation is reported as a
   diagnostic (`camera_fit_rms_m`); its size (0.4-1.6 m) is why the scaling is not about the camera (section 2.9).
   The body IMUs are not used
   anywhere in this stage: the recording CSV serves the sync, the subject identity and (walkway columns) the
   position reference only, so the file can supervise a sensor-to-pose model without circularity. Output:
   `merged_smpl.npz` + `merge.json` beside `reconcile.json`; `merge_status` / `export_frac` in the QC summary.
10. **QC and privacy.** Every stage writes JSON; skeleton-only thumbnails (black background, green = subject) at the
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

On the sample as re-rendered with the floor and mat visible (2026-11-03) the same CPU dry run passes 11/11 in
89 s: offset 3.008 s (8 ms), light agrees within 8 ms; along view 99 % of 1452 frames on the true hip (median
18 px), r 0.77, margin 0.63; side view 100 % of 1303 frames (median 11 px), r 0.72; exits as gaps, 4/4 bouts
covered. One difference worth knowing: the runner-up lag is no longer the co-walker's but an alias one step
period away (3.57 s, 87 % of the winner's support), which the light cross-check settles; a session without a
light leans on the joint support alone there.

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

#### Re-run after the chunk-planner fix (2026-11-03, same T4)

Dry run with the GPU stage: **PASS 14/14** (789 s). Side view lifted as two chunks (885 + 504 frames) with nothing
lifted over the 3.3-s exit (IMU 29.5-31.8 s); 88 bridged frames remain (the 1-s exit and brief losses), 146 frames
flagged edge-truncated; along view 959 edge-truncated frames; no frame with a joint-rotation step over 30 deg in
either view. Two-view leg disagreement 8.4 deg median overall; 15.9 deg (p90 27.2) on the side view's bridged
frames of a true exit, 6.5 deg on the along view's bridged frames (the subject was in frame, the tracker had lost
it); 10.5 deg in the first second after the re-entry, 12.2 deg in seconds 1-3. Reconcile PASS, `betas` constant
within chunks, privacy check clean (37 files). The same run also showed that the CPU stages must run in the pinned
environment: run for convenience inside the GVHMR conda environment (ultralytics 8.3.80 on the GPU instead of the
pinned 8.4.45 on CPU), the tracker fragmented the along view's subject into 14 pieces instead of 1 on HC900 (still
all accepted) and, on the harder HC901 side view, into 8 pieces covering 54 % of the recording instead of 3
covering 92 %, after which the joint selection locked onto a wrong lag (7.22 s versus 3.00 planted) with r 0.76 and
margin 0.59. The light cross-check flagged it (`flash_disagrees -0.79 s`, WARN). The supported workflow is the
documented one: CPU stages in the pinned environment, then `wgvp gvhmr` alone on the GPU machine
(`tools/remote_gpu_stage.sh`).

#### Re-rendered sample (floor and mat visible), 2026-11-03, same T4

The first renders of both synthetic sessions had single-sided floor, tile and mat quads that pyrender culled, so
the walker stood on the background colour; fixed with `SKIP_CULL_FACES` and both sessions re-rendered. Dry run
with the GPU stage on the re-rendered HC900: **PASS 14/14** (823 s). Along view 1457 frames in one chunk (1367
visible, 90 bridged, 966 edge-truncated; joint step 1.9 deg median, 14.1 maximum, none over 30); side view 1352
frames in three chunks (885 / 340 / 127) at its two exits (IMU 28.6-32.0 and 42.3-44.4 s), 1293 visible, 59
bridged, 100 edge-truncated, two frames over 30 deg (39.0 at most: frame 1385, bridged, and 1494, 1.8 s before the
end), both flagged. Two-view leg disagreement 5.9 deg median (hips 4.1 / 8.3, knees 5.4 / 7.2, ankles 3.9 / 5.8),
8.1 deg over all 21 joints, knee-flexion residual lag -46 ms, pelvis -3 ms, reconcile PASS. Bridged frames: the
side view's 56 true-exit frames disagree at 16.5 deg (p90 37.9), its 3 in-frame losses at 9.2, the along view's 90
in-frame losses at 5.0 against 5.3 on visible frames. Re-entries: first second 10.4 and 12.9 deg, seconds 1-3
11.0 and 9.7, steady state 5.3. The along view's 777 visible head-out frames agree at 4.4 deg (p90 5.7) against 9.5
on its other visible frames (the walker far from the camera). Every agreement number improved on the first render
(legs 8.4 to 5.9 deg): GVHMR's camera and gravity estimates lean on a visible ground, so the floorless numbers
above were the pessimistic ones. The README figures are made from this run.

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

### 2.7 Second synthetic session: two compromised views (HC901)

`sample_v2/` renders the same walk from an along view 0.5 m beyond the near end of the mat at 0.6 m height (whole
body only at the far end; head and shoulders gone mid-way; legs only when close: hips in frame 40 % of the time)
and a side view 6 m back at 1 m height aimed at hip height (head in frame 23 %, shoulders 89 %, walker in frame
96 %). The seated operator sits beside the far end, outside the side view. The walker's ground-truth SMPL pose,
root trajectory, 3D joints and per-view in-frame flags are saved in `HC901_gt.npz`.

CPU stages (pinned environment, after the re-render with the floor and mat visible): panels exact; sync 2.995 s
versus 3.000 planted, validated by the light (flash minus cross-correlation 5 ms); side view selected
automatically (r 0.58, margin 0.40, confidence high, 100 % of its 1341 frames on the true hips); along view NOT
selected automatically (its genuine fragments score r 0.2 at the shared lag; of its 12 fragments 5 can be scored
at all, the rest rarely show the hips) because, close to a camera at the end of the walkway, the per-step
oscillation of the walker's forward speed moves the hip row more than the vertical bob does, with a different
phase; the scale reference was ruled out as the cause (a truncated box keeps the gait-band content, see the
regression test). The along view is taken from an operator pick by frame and point, shipped with the sample; the
picked tracks follow the true ankles in 93-94 % of their frames (the hips are out of this view most of the time)
and both views are lifted. The recording's QC reads WARN for exactly this reason (a view chosen by pick, not
confirmed by an independent method).

GPU stage on the T4 (`tools/remote_gpu_stage.sh`, 2026-11-03, 13 min including the instance start): along view
1438 frames lifted in two chunks (1378 visible, 60 bridged, 1303 edge-truncated, one flagged step over 30 deg at a
bridged frame 90, the moment the sync light comes on); side view 1410 frames in two chunks split at its exit (1341
visible, 69 bridged, 584 edge-truncated, no flagged step). Reconciliation PASS: 2D pelvis signals agree at a 37-ms
residual lag, knee flexion at -39 ms (the knee check first reported +0.53 s on an earlier lift, one step period,
an alias that a +-1 s search window admits when the views are noisy; the window is now +-0.3 s as for the pelvis
check). Leg-joint disagreement between the views 13.2 deg median (knees 7-13, ankles 12, hips 57-59: the along
view's hips compensate for its root, below), against 5.9 deg on HC900 whose views see the whole body.

`tools/two_view_fusion.py` against the truth, 1395 common frames (geodesic error per joint, medians over frames):

| estimate | legs | legs p90 | legs > 20 deg | arms | spine | all 21 | knee-flexion RMSE |
|---|---|---|---|---|---|---|---|
| along view | 14.9 | 56.3 | 34 % | 39.1 | 20.0 | 21.7 | 14.8 |
| side view | 10.3 | 20.3 | 11 % | 16.1 | 10.1 | 11.5 | 8.2 |
| mean | 12.7 | 25.7 | 27 % | 22.7 | 14.3 | 16.2 | 8.6 |
| confidence-weighted (ViTPose) | 12.7 | 25.0 | 26 % | 19.4 | 12.7 | 14.3 | 8.8 |
| visibility-weighted | 9.6 | 19.6 | 9 % | 16.5 | 11.1 | 11.9 | 10.3 |
| segment rule, truth visibility (91 % of cells emitted) | 10.1 | 19.4 | 9 % | 16.5 | 8.4 | 11.1 | 10.2 |
| segment rule, run-time visibility (98 % emitted) | 10.1 | 19.3 | 9 % | 21.0 | 11.2 | 13.1 | 8.6 |
| segment rule, cross-faded (98 % emitted) | 10.1 | 19.4 | 9 % | 20.4 | 10.7 | 12.7 | 9.9 |
| segment rule, cross-faded + smoothed 5 frames | 10.1 | 19.4 | 9 % | 20.3 | 10.7 | 12.7 | 9.9 |
| segment rule, cross-faded + smoothed 9 frames | 10.3 | 19.7 | 9 % | 20.3 | 10.7 | 12.7 | 10.1 |
| segment rule, cross-faded, x each view's root share, smoothed 5 frames (the demo) | 10.0 | 19.3 | 8 % | 15.7 | 10.3 | 11.4 | 7.4 |
| oracle pick | 9.4 | 17.5 | 5 % | 15.2 | 9.2 | 10.7 | 5.1 |

Continuity (largest per-joint rotation step between consecutive emitted frames; p99 / share over 20 deg): along
view 6.0 / 0.05 %, side view 7.2 / 0.02 %, plain mean 5.0 / 0.02 %, confidence-weighted 7.1 / 0.12 %,
visibility-weighted 6.7 / 0.19 %, segment rule 16.8 / 0.79 %, segment rule cross-faded 5.5 / 0.03 %, cross-faded
and smoothed over 5 frames 4.1 / 0.00 %, over 9 frames 3.0 / 0.00 %, with the root share 4.8 / 0.00 %, oracle
pick 13.0 / 0.45 %. Any rule that switches views per joint introduces steps of the size of the views' disagreement
unless the switch is faded; the cross-fade (weights ramped linearly over the first and last 15 frames of each
usable segment, zero outside it) removes them at no cost in error. The lifter's own jitter inside a segment is a
separate matter: a quaternion moving average within runs of emitted frames (gaps are boundaries, never averaged
across) takes the remaining steps down further at 5 frames with the error unchanged, while 9 frames starts to
attenuate the stride peaks (knee-flexion RMSE 9.9 to 10.1); 5 is the demo's default.

The segment rule emits a joint only from the view(s) in which the ViTPose keypoints that joint moves are confident
and inside the panel (margin 8 px), averaged when both views qualify, nothing when neither; its errors are over the
cells it emits. On the frames where the along camera has the hips out (798 of 1395, the standing start included)
it emits 98 % of the cells at 8.2 deg on the legs where the plain mean gives 11.6; on head-cut frames it is the
plain mean (both views qualify). Up to the cross-fade and the smoother it does not beat the better view on the
knee-flexion RMSE (9.9 against the side view's 8.2): wherever both views qualify for a leg joint, the along view's
poorer estimate is averaged in at equal weight.

The root (`fuse_root`). The along view's root pitches forward whenever the body is cut: signed median pitch
error +17 deg with the shoulders out of frame (n = 377), +50 deg with the hips out (798), against -5 / -3 deg for
the side view and -7 / -6 deg for both views on whole-body frames; its hip joints compensate (hip error 22.4 deg
against the side view's 15.3 on the shoulders-cut frames, 56.2 against 8.4 on the hips-cut frames), so that the
legs still land on the image while the pelvis is wrong in the world. The first demo took the root from the view
with more usable joints when the two disagreed by more than 60 deg and otherwise averaged them by usable count;
the along view's count toggling between 0 and 15 at IMU 24.7-25.2 s (measured on the previous lift of this sample)
moved the reconstruction's pitch by up to 17 deg each time, the rocking visible in that cut; frame-step statistics
missed it, because each swing takes 10-20 frames. The root is now the quaternion mean of the views' roots weighted
by presence (ramped over 15 frames at the edges of each lifted run) x (usable fraction, averaged over 15 frames)^2,
with the weights sharpened towards the heavier view (exponent rising from 0 at 30 deg of disagreement to 8 at
60 deg); each term is continuous. Fused root error 6.6 deg median against 49.6 (along) and 7.3 (side); frame step
p99 5.4 against 47.4 and 7.3; the along view's mean share 0.14. The view's share also scales its weights for every
body joint, on the argument that a lifter whose root is in the wrong gravity frame solved the whole body in it:
that row of the table lowers the knee-flexion RMSE from 9.9 to 7.4 deg (side view alone 8.2), the arm error from
20.3 to 15.7, the all-joint error from 12.7 to 11.4 and leg outliers from 9 % to 8 %, at a frame step p99 of
4.8 deg; it is the first fusion row below the better view in every column. Variants that did not move anything
measurable on the previous lift: a debounce dropping usable runs shorter than 5 or 10 frames before the ramps
(identical to three decimals).

What an export keeps (`export_mask`): frames in which every leg joint and every torso joint (spine, neck, head) is
output, in runs of at least 15 frames, with only the joints some view saw (an arm may be absent; the rule was legs
only until 2.12). On HC901 that is 91 % of the IMU span (1328 of 1458 frames; the along view alone has the legs for
35 % of it, the side view for 95 %). The demo draws it beside the reconstruction with the unseen parts cut out of
the mesh and the empty mat on dropped frames.

By what the along camera saw (leg / all-joint error of that view): whole body 14.2 / 15.7 (n = 70), head cut
14.3 / 14.8 (143), shoulders cut 17.1 / 20.0 (377), hips cut 14.4 / 33.9 (798), feet cut 43.2 / 31.9 (7). By what
the side camera saw: whole body 14.3 / 12.6 (328), head cut 8.5 / 10.7 (956), shoulders cut 11.9 / 13.2 (24),
feet cut 17.6 / 14.9 (84). Root orientation after a constant heading per view (+89 deg along, +5 deg side, the two
cameras being 90 deg apart): 49.6 / 7.3 deg median. Reading: truncation at the head is free; a view that keeps
only the legs still gets the legs (14 deg) but invents the rest of the body (34 deg) and pitches the root (+50
deg); a plain two-view fusion is bounded by the better view; cutting each view into the segments where it sees a
joint, emitting nothing elsewhere, fading the hand-overs and weighting each view by its share of the root beats
the better view in every column with run-time tests alone; the per-joint selector that would recover the oracle's
remaining margin (legs 9.4, knee-flexion RMSE 5.1) is the open question. Compared with the first lift of this
sample (rendered without a visible floor), every single-view number improved, the side view's leg error from 13.9
to 10.3 deg: GVHMR's camera and gravity estimates lean on the ground.

**The merge as a pipeline stage (2026-11-04).** The fusion library moved into the package (`wgvp.fusion`) and the
merge became stage 9 (`wgvp.merge`), with run-time inputs only: no truth-fitted heading per view, the second view
is turned onto the first from shared frames. On HC901 the stage covers 1,453 frames (every frame some view
lifted), exports 96.5 % of them (1,402 frames in 3 runs) and, against the rendered truth on the exported frames,
has legs at 10.0 deg median with 8.7 % over 20 deg, all 21 joints at 11.4 deg, knee-flexion RMSE 7.5 deg: the
same as the analysis tool's best row, so the truth-free alignment costs nothing measurable. The along view is
turned onto the side view by 80.4 deg where the truth-fitted yaws of the two views differ by 84 deg; the
translation offset is recovered to 2 cm; the roots still differ by 54 deg on the shared frames, which is the along
view's pitch (an `info` flag, since the merge gave that view a mean root share of 0.17). Fused root 6.6 deg
median; its frame step p99 8.1 deg (the tool had 5.4 on the both-lifted frames: the stage also covers the frames
only the along view lifts, and at IMU 37.5 s, with the side view out and the legs alone in frame, GVHMR flips
that view's root by 163 deg in one frame; the root weight there is 0.08). The weight does what it is for: median
leg error 9.7 deg on cells with weight > 0.5, 15.0 deg below; its correlation with the per-cell error is -0.46.
Body shape: each view's median betas lie 2.2-2.4 from the truth (the walker has the mean shape), 1.7 from each
other; GVHMR's shape from a single far view is not accurate, which is why one vector per recording is taken and
the two views' difference is reported rather than trusted. HC900 (the dry-run session): 1,458 frames, every one
exported, legs 8.7 deg, knee RMSE 5.7, root 4.8 (step p99 5.2), the side view turned onto the along view by
-115.3 deg with residuals 5.4 deg and 3 cm, mean weight 0.89. The dry run now checks the stage with the GPU: the
file exists and exports >= 70 % of the lifted frames, legs within 15 deg of the truth, root within 15 deg after
one constant heading with step p99 under 12 deg, no yaw left in the heading-relative root.

Translation is the weak field, and the two sessions show how. GVHMR's position is a monocular depth estimate
under an assumed focal length: median error after one yaw and one offset 6.5 cm (HC901) and 11.6 cm (HC900), but
the 90th percentile is 1.4 m in both. The error sits where a view sees the walker small: on HC901 the frames
where the along view carries the root alone have a median position error of 1.53 m (the side view's 5 cm), and
on HC900 the side view alone places the far ends of the walk 3.1 m off, its x-span 7.5 m against the truth's 4.9.
Where both views see the subject their disagreement exposes it (`transl_disagree_m`, discounted by
`transl_weight`); a single far view does not. The walkway export in the same CSV carries the mat position, is
independent of the body IMUs, and is the right scale reference; it is not used yet. The velocity target inherits
this: forward-speed RMSE against the truth 1.0-1.3 m/s over all frames, dominated by the ends of the walk.

### 2.8 Three subjects in a partner-style folder layout (2026-11-04)

The three synthetic sessions (HC900, HC901 and the new HC902: another walker, a co-walker on the far side of the
mat, the along camera at the far end and 1.5 m high so that the legs leave its frame when the walker is close, the
side camera on the other side of the room) were laid out as a partner might: `Sub 001/Video/SP_combo.mp4` with
`Sub 001/IMU/Sub001_selfpace.csv`, and the same for `Sub 002` and `Sub 003`. The default matching rules find all
three (the task from the `SP` token, the subject from the folder); HC901's along view needs its shipped operator
pick, renamed to the new recording id. The CPU stages ran locally, the GPU stage on the T4
(`tools/remote_gpu_stage.sh`), the merge and the three demos (`wgvp demo`, with each sample's truth) after.

The first HC902 was rendered with hc102 as the walker, and it failed in an instructive way. hc102's IK trajectory
covers 1.1 m of walkway (the stance-foot-pinned root barely advances) and its pelvis bob is 5 mm (hc100: 4.9 m,
10 mm), so the rendered walker paced within a metre, crossing in front of the co-walker. The side view's tracker
cut the subject into some forty fragments of a few seconds each, whose own best lags scattered from -0.8 to 45 s;
the along view's single long fragment peaked at r 0.39 with a 0.24 margin, which passed the "high" confidence
rule, and the joint lag landed at -27.4 s against a truth of 3.0 s. Nothing in the record contradicted it: the
side view simply chose nobody, the sync light's step at 3.0 s was not within 3 s of the offset and so was not
compared, and the sync read PASS. Two sanity rules now catch exactly this, and both fire on the case: the offset
leaves only 43 % of the recording inside the video (threshold 60 %), and a strong persistent light-like step
(z 640) sits 30.4 s from the offset. Each makes the sync a WARN with the reason in `sync_flags`
(`tests/test_sync_sanity.py`). The walker was then replaced by hc105 (6.3 m of walkway, 11 mm of pelvis bob, the
largest in the public set), with the side camera turned towards one end so that the walker leaves it at the other
for 45 % of the recording while the along camera sees the whole body there.

The second pass exposed a robustness bug in the GPU stage. Its three chunks lifted normally (299 s for the along
view, 87 s for each side-view chunk), but the assembly loop then reached the first two subjects, whose chunk
records had come back from the first pass marked done while their pose dumps had stayed on the box, and
`np.load` on the missing file took the whole stage down before the new subject was assembled. Assembly now
reports missing dumps per view (keeping an existing output when there is one) and a failure in one view no longer
stops the others (`tests/test_assemble.py`); the third subject was assembled from the dumps still on the box.

Results (each session against its own rendered truth, exported frames). Sub 001 (HC900): scan paired, sync 3.008 s
(truth 3.0, validated by the light), both views by cross-correlation (r 0.77 / 0.72); merge export 100 %, legs
8.7 deg, knee-flexion RMSE 5.9, root 4.7, the side view turned onto the along view by -116.5 deg with residuals
5.3 deg / 3 cm; 4 GVHMR frames flagged for a joint step over 30 deg in the side view. Sub 002 (HC901): sync
2.995 s, the along view by its shipped operator pick, the side view by cross-correlation (r 0.58); export 96.5 %,
legs 10.0, knee RMSE 7.5, root 6.6, the roots 54 deg apart on shared frames (the along view's pitch; info flag).
Sub 003 (HC902, hc105): sync 3.007 s validated by the light; the along view by cross-correlation (r 0.66, margin
0.51), the side view "weak, lag from the other view" with six fragments attached (WARN as designed: the walker is
out of it for 48 % of the recording); 1,205 + 654 frames lifted; export 100 %, legs 11.5 deg (14 % over 20),
knee RMSE 9.2, root 7.4 (step p99 5.7), the side view turned onto the along view by -128.7 deg with a root
residual of 8 deg and a pelvis-position residual of 0.52 m, which the merge flags: the two monocular depths
disagree by half a metre on the frames both views see (`transl_weight` mean 0.67 against pose weights of 0.92).
The reconciliation's two-view leg disagreement is 9.2 deg. Timing: CPU stages 65-85 s per session on the laptop,
the lift 8-10 min per session on the T4, the demo 2.5 min per session. The demos of all three sessions and the
README GIF were rendered by `wgvp demo` from these outputs. Lesson for re-runs on a remote GPU: a pull with
`rsync` must escape spaces in the remote path (`Sub\ 003`), and macOS's rsync has no `-s`.

### 2.9 Round two: other tasks, the mat with walkway footfalls, and a real clip without a truth (2026-11-04)

Three more synthetic sessions (TUG with a chair, hurried pace in the compromised views, a self-paced mat walk with
fabricated walkway footfalls) were added to the run folder as `Sub 004-006`, and the bundled example of our own
phone clip (`examples/bundled`, OWN01 single view and OWN02 two-view composite) was lifted on the T4 for the
no-truth demo. One thing broke on the way in and was fixed: a config written by the previous version keeps its own
token lists, so the mat video (`SPM_combo.mp4`) went unmatched until `SPM` was added to that folder's
`wgvp_config.json`. The walkway columns were misread in this round (the reader took the export's pipe-separated
cell lists for malformed numbers and concluded the columns were empty in all but 7 of 1,577 files; section 2.10
corrects this: they are filled in 1,574), so the calibration was tested on the rendered mat session with footfall
rows fabricated from the truth, one per footfall in cm. A first
attempt that fitted the scales jointly after the view alignment failed in the unit test: the rigid alignment had
already absorbed the scale error into the second view's offset and moved its camera centre with it; fitting each
view in its own world first, and aligning the scaled positions afterwards, recovers the injected factors (1/1.3
and 1/0.8) to within 6 %.

**What the walkway calibration does and does not do.** The first model scaled each view about its camera centre,
the geometry of a wrong focal length or body scale. It does not apply: GVHMR's world translation is not a rigid
image of its camera-frame translation (Kabsch residual 0.43 / 0.64 m on the mat session's views, 0.80 / 1.64 m on
HC900's), because the world path is integrated by the network, not projected. The shipped model is a per-view
similarity of the path about its own footfall-time centre, fitted on the landing foot (forward kinematics of the
mean body; foot labels from the contact flags, unlabelled rows re-associated with the nearer foot), accepted when
the residual stays under 0.5 m. On the mat session (HC905, 20 fabricated footfalls, all labelled): scales 0.87 and
0.83; path length +15 % too long before, 1 % short after; error vs truth median 0.19 -> 0.23 m, p90 0.85 -> 0.61 m,
far ends 0.25 -> 0.24 m; forward-speed RMSE while walking 0.26 -> 0.28 m/s. On HC900 (21 fabricated footfalls),
whose side view stretches the walk from 4.9 to 7.5 m: both fits rejected (scales 0.36 / 0.29, residuals 0.72 /
0.64 m), i.e. that path is distorted beyond a similarity. With the root instead of the feet (first attempt) the
mat-session scales came out at 0.76 / 0.75 and made the path 12 % short: half a step of foot-to-root scatter on 20
rows is too much. Conclusion: the walkway cannot repair GVHMR's path by rescaling; per-frame jitter dominates the
median and drift dominates the gross cases. If a partner's export carries footfalls, the position on the mat should
come from them directly, with the pose from the video.

**The other tasks.** Sub 005 (TUG, HC903): sync 3.005 s (truth 3.0, validated by the light), both views by
cross-correlation (r 0.82 facing the chair, 0.70 side); 1,499 + 1,357 frames lifted; export 97 %; knee-flexion RMSE
9.3 deg, root 8.4 deg, two-view leg agreement 9.3 deg (ankles 10 deg); per-joint error against the truth 26 deg on
the legs in EVERY phase (Walk 26.9, Sitting 25.1, SitToStand 26.6, TurnToSit 29.3) with the ankles at 92-99 deg,
and each view alone shows the same numbers: the IK truth of the TUG (the task with the lowest calibration pass
rate, 41 % of recordings) carries an ankle convention and a knee axis the lifter does not share, so this sample
measures the truth, not the lift. Sub 006 (hurried pace, HC904): sync 2.999 s; the side view holds the walker for
42 % of the recording in five pieces; 751 + 563 frames lifted, 130 shared; export 82 %; legs 14.0 deg median
(23 % over 20), knee RMSE 13.8, root 7.8; the two views' pelvis positions 1.65 m apart on the few shared frames
(flagged). Sub 004 (mat, HC905): legs 9.6, knee RMSE 6.7, root 6.2, export 100 %, views aligned to 7.4 deg /
0.40 m. The bundled clip (OWN01 single view: 393 frames, export 95 %; OWN02 mirrored composite: export 100 %, the
mirror "view" turned onto the original by -22 deg with a 14 deg root residual, as a mirror is not a second camera)
rendered through the no-truth demo. GPU: 36 min for the three sessions, 9 min for the clip.

### 2.10 Round three: the walkway's real form, the mat path, a mirrored view, and two fixes (2026-11-05)

**The walkway columns were never empty.** Re-reading whole files instead of their heads: 1,574 of the 1,577 public
recordings carry `Walkway_X/Y` rows, in every task (the mat records all through a session; a self-paced corridor
walk crosses it a few times), starting at 0 s in some files and 7-36 s into others. Each row is a pressure frame:
the lists of loaded cells (`'51|52|53|...'` in X, the matching Y), 50-120 cells when one or two feet are down,
integer indices 0-383 along and 0-47 across, every cell carrying a footprint label (`L`, `R`, `Other 1`) in
`WalkwayFoot` and a level 2-13 in `WalkwayPressureLevel`. 384 x 48 cells is the GAITRite's grid at half an inch
(4.88 x 0.61 m; a 1-cm pitch would make a 48-cm-wide mat, which nothing sells), and loaded footprints 13-17 cells
long by 9-10 wide fit a shoe at that pitch. The previous reader's `_safe_float` had turned every such list into NaN,
hence "7 of 1,577". The new reader keeps per sample and labelled foot the pressure-weighted centre and groups the
samples into stances (one open stance per foot; drift under 0.25 m; gaps under 0.12 s): a self-paced mat walk gives
32-36 stances of 0.59 s median, alternating feet 91 % of the time, a TUG 65 stances of 0.75 s; whole files parse
in 0.1-0.5 s. `Other 1` cells (3-20 % of cells; the chair's legs in a TUG, unassigned footprints) are counted and
ignored.

**The rendered walks do not match their recordings' own footprints.** The render walks the neutral SMPL body along
a stance-pinned path from the inertial pose; for hc105 that path spans 3.42 m where the mat's footprints of the
same walk span 4.45 m (a rigid fit leaves 0.40 m RMS, a similarity fit a scale of 0.78), i.e. the rendered walk is
22 % shorter than the real one, the neutral body's legs being shorter than the subject's. So the real walkway rows
would contradict the rendered video, and the samples' CSVs now carry empty walkway columns, except the mat
session's, whose cells are fabricated from the rendered truth in the export's own form (`tools/make_sample.py`:
an ellipse of 24 x 9 cm of cells under each foot for every sample its contact flag is on, heel- to toe-weighted
through the stance, on a 384 x 48 grid centred on the walk; 5,279 foot-samples, 32 stances). A `--csv-only` mode
rewrites a rendered sample's CSV without re-rendering it.

**The pelvis path on the mat.** With stances in hand the merge does what the round-two note asked for: per stance
the footprint's centre minus the video pose's foot-to-pelvis offset (forward kinematics of the mean body, turned
into the mat frame by the mat pose fitted on the fused feet), both feet during double support, each stance's say
ramped over 0.12 s, gaps under 0.5 s bridged. On the mat session against the rendered truth (exported frames):
the video's own path 0.283 m median / 0.651 m p90 horizontal error after a rigid yaw+offset, the scale-calibrated
path 0.289 / 0.494, the path from the walkway 0.034 / 0.089 m over 86 % of the frames (the rest are off the
mat); in the mat frame with no fitting at all, bias 0.1 cm along and 4 cm across, spread 4 cm. The fabricated
footprints come from the same truth as the video, so this measures the mechanism (stance detection, FK offsets,
hand-overs) and not the mat's own accuracy. Per-frame speed is not a useful metric here: the truth smoothed over
0.5 s misses its own 30-Hz speed by 0.12 m/s RMS (within-stride speed swings), and all three paths sit at
0.27-0.30 m/s against it. The scale fit stays in the file but the verdict of 2.9 stands (it does not repair the
path); the mat path is the useful product.

**A mirrored view.** The user's two-panel composite of the phone clip was re-made the way a careless second camera
would be: the clip zoomed in a little on the left (middle 84 x 86 %, 1.16x), the clip MIRRORED and zoomed in more
on the right (62 x 68 %, 1.32x, letterboxed), a 6-px separator (1298 x 1280). The CPU stages read the panels to 2 px
of the truth, sync at 0.817 s (planted 0.8), select the walker in both panels (r 0.84 / 0.81). A mirrored panel's
left leg is the walker's right, so the merge now tests every other view against the reference for being its mirror
image: `fusion.mirror_body` swaps the left/right joints and conjugates each local rotation by the sagittal
reflection (rotvec (x, y, z) -> (x, -y, -z); forward kinematics of the template agrees to 3 cm at the feet), the
root flips the same way and x -> -x, and the test compares the KNEES on frames where the reference's knee flexion
asymmetry exceeds 15 deg (one leg swinging), as lifted vs swapped. On the six rendered two-camera sessions the
knees read 11-16 deg as lifted vs 21-40 swapped (never flagged); on the previous composite's lift, 43 vs 7 (flagged,
reflected back: root residual 14 -> 7 deg, pelvis offset 0.33 -> 0.21 m, alignment yaw -22 -> -5 deg, knee curves
of the two "views" on top of each other). An all-legs median over all frames was too soft a test (as lifted 5.5 vs
swapped 8.7 on Sub 001): standing and turning frames look the same mirrored. The unit test plants a mirrored view
and expects the merge to recover the truth to 3 deg on the legs and to leave an honest pair unflagged.

**Two fixes from the user's viewing.** (1) Sub 005's rendered truth had a wrong left foot: hc100's TUG inertial fit
carries its left ankle at 163 deg (median; the right 21) where every other TUG fit sits at 7-20, so the sample was
re-rendered with hc142's TUG (ankles 10 deg, knees to 85 when seated; 1,745 frames, 53.7 s, whole body in both
views); its lift follows the next GPU run. (2) Sub 006's green skeleton at 0:14 of the demo draws both legs on one
foot in the side view: ViTPose put both ankles (and knees) on one leg, 0.3-1.7 px apart with confidence 0.8-0.9,
for 7-10 frames at a time. Measured on all six sessions against the truth: a collapsed pair (ankles within 8 % and
knees within 12 % of the shank length) occurs on 0.1-23 % of a view's frames (23 % in HC901's far side view, where
the lifter is nevertheless at its best, 7.8 deg), and the lifter's leg error on those frames is 8-19 deg against
9-25 elsewhere; ankle-only collapses are 0.7 % of frames at 15.9 deg vs 12.7. So the merge does not weight them
(the agreement factor already handles a view that is wrong) and the demo draws them in amber with a note; the
close-ups on real video no longer draw a mat (floor only; the mat belongs to the rendered rooms). The recording
reference's schema is now v2 (stances), and a cached reference is rebuilt when the schema or the cell size changes.

**On the renders themselves (a question asked while viewing).** The synthetic sessions are rendered from one 3D
scene with two perspective cameras placed where the aerial panel shows them; pyrender (OpenGL) draws each camera's
picture, the camera poses and intrinsics are saved with the truth, and the 2D truth joints are projections through
those cameras. Nothing about the views is composited or drawn by hand, so a camera's height, tilt and field of view
decide what is truncated. A game engine (Unity, Unreal) would add appearance, not geometry: textures, clothes,
lighting, lens distortion, motion blur, i.e. a harder test for the 2D detectors, which the real-footage examples
cover instead. Two layout corrections followed from the same viewing: the side camera of a real session sees the
entire room (the Video Labeler frame: elevated, wide, the whole mat with the operators' desk in the foreground, the
walker complete most of the time), so the hurried-pace sample's side camera was re-laid out that way (2.2 m high,
7 m back, 40 deg vertical, 66 deg across) and the first render's narrow side view (v2's, which lost the walker for
58 % of the recording) is kept only in HC901 as the deliberate stress test; and the user's two-panel composite
got a harder zoom on its mirrored panel (the middle 50 x 55 % of the clip, 1.64x) so that head and feet are cut
when the walker is close, as in the along-walkway view of a real session.

**The lifts after those corrections (T4, 29 + 15 min).** Sub 005 (TUG, hc142): sync 3.004 s, export 94 % of 1,609
frames in 4 runs, legs 14.3 deg median (24 % over 20), knee-flexion RMSE 8.1, root 12.8 (the seated and turning
phases), two-view leg agreement 7.6 deg on 865 shared frames: the 26-deg "error" of the first render was the
truth's. Sub 006 (hurried pace, room-wide side view): sync 2.992 s; the side view holds the walker for all 1,101
frames with every leg joint usable, the along view has its legs usable 52 % of the time; export 100 %, legs
12.0 deg (15 % over 20), knee RMSE 10.5, root 8.8, two-view legs 10.5 deg on 569 shared frames (82 % / 14.0 / 13.8
/ 7.8 with the narrow side view). The bundled composite (OWN02, zoomed 1.16x | mirrored 1.64x): panels within 2 px,
sync 0.817 s, both panels selected (r 0.85 / 0.77); the merge took the zoomed mirrored panel as the reference,
flagged the other as its mirror image (knees 44.2 deg as lifted vs 7.8 swapped on 216 swing frames), and the two
agree on the legs to 5.4 deg with 100 % exported; OWN01 95 %. One limit surfaced: on the frames where the walker's
feet lie below the hard-zoomed panel, ViTPose reports both ankles with confidence 0.9 about 200 px INSIDE the
panel (and the tracker's `edge_truncated` stays False), so the usable-joint rule passes a cut-off foot and the
lifter extrapolates it; the two views' agreement factor is the guard that remains, and a geometric truncation test
(the body's box against the panel edge, or the limb's expected extent) is the open item. Note also that
`pose_agreement_<a>_<b>.npz` from the reconcile stage compares the views AS LIFTED (before the merge's reflection),
so on a mirrored pair it reads 20-40 deg while the merge's own agreement is 5 deg; the demo's chart uses the
merge's reflected bodies.

### 2.11 Round four: the turn that snapped, and the room laid out after the real one for every sample (2026-11-05)

**The TUG's turn.** At 0:40 of Sub 005's demo the reconstruction flipped instead of turning. The per-view headings
tell why: through the 180-degree turn (1.5 s) the chair-facing view follows the truth (149, 180, -145, -124 deg at
quarter-second steps), the side view's lift takes the turn the other way round and snaps through the front/back
ambiguity (140, 18, -8, -27: 120 deg in 0.25 s), and with the two views at equal share the quaternion mean of two
opposite headings is meaningless, so the merge jumped from 144 to -77 in one frame. `fuse_root` now arbitrates
disagreements over 30 deg before sharpening: a view whose root turned faster than 150 deg/s since its previous
frame is discounted by 1 + ((rate - 150) / 150)^2 (a person turns at up to about 150 deg/s, a snapping lift at
500 or more), and a view far from the previous fused root by 1 + (angle / 30 deg)^2, which keeps the output with
the view it was following until the two agree again. Re-merged, Sub 005's heading stays within 5 deg of the truth
through the turn, 2.5 deg median and 14.5 max over the recording, with no step over 90 deg (`merge.json` now counts
them under `continuity.root_flips_over_90deg` and flags any). A first version that used only the distance to the
previous output failed its own test: before the snap the output had been the mean of the two diverging views, so
neither was nearer to it; the rate term is what breaks that tie. A frozen lift versus a turning one remains
undecidable by these terms, and the test plants a view that tracks until it snaps, as the real one did.

**The room, again.** Two more corrections from the user's viewing of the real Video Labeler frame: the side view
sees the entire room, elevated, very wide, the mat in the lower middle with the operators' desk in the foreground,
and aimed low enough that a walker's head sits at the top edge and leaves it when tall, bouncing or nearer the
camera; the along view sits at the mat's end about a metre high, so a walker two to three metres away shows from
the chest down (the first version's camera 0.6 m high and 0.5 m past the end gave shins only, and its 2D skeleton
jumped about when the walker was close). The mat is 16 x 2 ft (4.88 x 0.61 m, the GAITRite, exactly the 384 x 48
cell grid of section 2.10), centred on the walk: self-paced and hurried walks step off its ends to turn, the mat
tasks turn on it. Every sample now follows this layout (`side_cam`, `along_cam` in `tools/make_sample.py`: side
camera 2.3 m high, 43 deg vertical in a 2.4:1 strip, first 5.5 m from the mat's centre line aimed at -0.65 m on
the mat's line so the top edge at the mat was 1.65 m above the floor (a first pass at -0.42 m left 15 % of the
strip above the head), then, after the comparison with the real frame below, 4.0 m away aimed at -0.21 m (top
edge at the mat 1.55 m); along camera 1.0 m high, 1 m beyond the mat's end
and 1.2 m beyond the walker's turn point, 60 deg), with HC901 keeping the low, close along camera as the stress test.
The old HC901 operator pick belonged to the old camera and was removed; the user's reading of the layout is that
most of the information will come from the side view, the along view supplementing it.

**The room-wide side view defeats the pelvis match.** The first CPU dry run on the re-rendered HC900 failed its
side-view checks: the along view selected the walker as before (r 0.73), the side view tracked them in three
fragments (34, 10 and 4.5 s, keypoint confidence 0.95) but accepted none. Their box is 24 % of the panel height,
144 px, the size of the walker in the real Video Labeler frame, and at that scale the vertical pelvis bob (a
centimetre, under a pixel) sits below the keypoint noise: the 34-s fragment correlates with the IMU at r 0.17 at
the true lag, the chance level for that window; its own best lag is 21 s off. Running the detector at 1280 px
instead of 640 changed nothing for that view (and fragmented the along view, 7 pieces for 1). So a second cue was
added for the view the pelvis match leaves empty: the correlation, at zero lag on the shared video clock, of the
fragment's progress along the walk with the confirmed subject's in the other view (the hip column where the walk
runs across the picture, the log of the body's pixel width where it runs into it). On the dry run's side view the
walker's fragment reads |r| 0.86 against the along view's subject, the operator's 0.36, the bystander's 0.28, and
short pieces reach 0.9 only at lags of a second or more. What this cue cannot do is tell a person walking in step
from the subject (the correlation of a 0.8-s-delayed copy of a 20-s cycle is 0.97), which is what the pelvis match
did with its sub-frame lag; such a case is flagged by the margin and left to the thumbnails or a pick.

**The same cue on all six, and the co-walker's gait.** On the full run root the first version of the cross-view
cue (hip column against log body width) resolved only HC900's side view; the hurried-pace, HC902 and mat-walk side
views read 0.71, 0.35 and 0.59, because the along view's body width is a poor progress signal once the shoulders
leave the frame (each part's width was centred on its own frames). Two changes: one consistent body scale (the hip
width, the knee or shoulder width brought onto it by their median ratio on shared frames) and a third signal, the
feet's row, which in an along view is the distance itself. The feet-row-against-hip-column pairing then reads
0.98 (HC900), 0.90 (hurried pace) and 0.95 (mat walk) with margins of 0.27-0.81, and HC901's low along camera,
whose own pelvis match fails by design, takes its subject from the side view at 0.95 with a 295-frame identity
switch masked out (coverage 96 %). HC902 showed the other failure: its along view accepted the co-walker (hc100's
walk, 15 s) at r 0.78 at a lag of 7.3 s, support 6.1 against 2.2 for the true lag, which the strong-light rule
flagged. The sync light is now evidence as well as a check: when such a step contradicts the pelvis match's
winner, `joint_selection` is re-run preferring the candidate within 0.5 s of the light (adding the light's lag as
a candidate if no fragment proposed it), and HC902 comes back at 3.008 s with the walker's fragments (r 0.70, high
confidence) and the side view confirmed by cross-view. A cross-view reference now also counts when the pelvis
match's confidence is low but the light validated its lag (HC901's side view: r 0.32). The dry run's identity
check asks whether the truth hip lies inside the chosen box rather than within 36 px of the detector's hip
keypoint, since that keypoint sits about 30 px high while the head is cut at the top edge (placement, not
identity). Two detector input sizes were tried for the small walker: 1280 px does not rescue the pelvis bob and
fragments the along view (7 pieces for 1), so 640 stays the default.

**The turn that turned at a human rate.** With every sample re-rendered in that room and lifted (T4, 65 min for
the six), Sub 003's self-paced walk (HC902) still carried two root flips of 169 and 174 deg, at its two turns
(18.3 and 32.4 s). The per-view headings against the truth showed a different failure from the TUG's: the along
view turned with the truth (+134 deg over the 2-s turn, 100-150 deg/s), the side view turned the *other* way at
a similar rate (-88 deg) and then snapped 177 deg onto the true heading in one frame when the turn was over. Both
terms of round four's rule are causal: the rate term saw nothing wrong until the snap, and by then the continuity
term had locked the output onto the side view, the heavier one (21 usable joints against 14; its share was 1.0
through the turn, the along view's 0), so the output turned the wrong way and snapped with it. Nothing
distinguishes the two views while they turn; what does is the snap, and the merge runs offline, so it can read it
after the fact. `fuse_root` now runs a third rule over the whole recording before the causal pass: inside every
disagreement episode (two views more than 30 deg apart, gaps up to 15 frames bridged), a view whose own root turns
more than 90 deg within 0.25 s, during the episode or in the 0.25 s after it, is the one that took the turn the
wrong way (a person cannot turn 90 deg in a quarter second; the snap is the lift's own correction, so that lift was
wrong on one side of it, inside the episode), and its weight is zeroed over the episode, up to 4 s either side of
the snap; when no view snaps or more than one does, the rule is silent. The rule is covered by a test that plants
the HC902 pattern (the wrong-way view heavier, snapping at the end; the old code fails it, following that view
through its wrong turn and its snap) beside round four's test of the TUG pattern (the snap at the start), which still passes because the
rule and the causal terms agree there. Re-merged, HC902 has no flip, root error 8.6 to 8.1 deg and head 4.6 to
3.5; the other five recordings are unchanged to the first decimal (HC905's root 7.8 to 7.6).

**The side camera, measured against the real frame.** Checked against the Video Labeler frame once more, the
re-rendered side view was framed too far back: the mat spanned 44 % of the panel's width where the real one spans
63 % (the walker's size was right, 125 px of 375, against 140 in the real frame), and the walker's head sat on the
top edge rather than outside it, which is what the user had asked for. The camera moved from 5.5 to 4.0 m from
the mat's centre line (the mat at ~60 % of the width), aimed at -0.21 m on the mat's line so the top edge crosses
the mat 1.55 m above the floor: the top of the head is at or beyond the top edge along the mat while the face
keypoints stay inside on most frames (the head joint is usable on 92-100 % of the side view's frames, 66 % on
HC902), as in the real frame, where the head touches the top; the operators' desk moved with it, 1.2 m in front of
the lens. The
along view did not change; for the record, its 2D keypoints are steadier than the first version's low camera (the
"jumping skeleton" of Sub 006): on the hurried walk all 1,101 along-view frames have confident leg keypoints
against 662 of 881 before, and the high-frequency deviation of the leg keypoints is 0.35-0.46 % of the leg's
extent against 0.47-0.73 %. Every sample was rendered and lifted a third time with this camera; the numbers in
the README are from that run.

**The light's lag, refined half a step off.** The CPU pass on the third render synced Sub 002 (HC901, the
legs-only along camera) 0.54 s late. The pelvis match's own winner was 9.48 s, a bout later than the truth (the
self-paced walk repeats its bouts every ~6.5 s, and the subject's own fragments correlate at a bout's shift: 9.48,
-3.04 and 26.4 s at r 0.48-0.77, none at 3.0); the light's candidate, 3.0 s, was taken as designed, and then the
refinement step, which looks for the highest point of the combined correlation curve within 1 s of the chosen
lag, slid onto the neighbouring step's peak at 3.54 s: the pelvis bob repeats every step (0.45-0.65 s), so a
1-s window always holds the two neighbouring peaks, and at r 0.5 against 0.4 the wrong one won, with "high"
confidence and a flash-disagrees flag. Two changes in `joint_selection`: a lag taken from the light is refined
within 0.1 s of it only (`prefer_refine_s`; the per-view lag estimate behind the inter-view agreement check uses
the same window, or it reported the views 0.56 s apart), and, light or not, a chosen lag whose runner-up carries at
least 80 % of its support (`lag_ambiguity_max`) is reported with low confidence and `ambiguous_lag`: on the six
rendered recordings the right lags' runners-up carried 0-26 % (one light-chosen case 69 %), the wrong ones' 88 and
91 %, so a periodic protocol without a light now says so instead of syncing a bout off with confidence. Re-run,
Sub 002 syncs at 2.981 s, validated by the light, the along view confirmed by the cross-view cue, reconcile PASS.
Its lift from the chain's first pass (at the wrong lag) was redone.

**The turn that ran out of frame.** The third render's lifts, merged with the snap rule, left one flip, in the
hurried walk (HC904) at its last half second: the walker turns at the end of the mat, the along view turns with
the truth (4 deg off), the side view turns the wrong way (146 deg off at the end) and its lift simply ends at frame
1185, the walker gone from its frame, before any snap; the merge had given that heavier view the whole root share
(21 usable joints against 14), and at the hand-off the fused heading jumped 177 deg onto the along view. The snap
rule had nothing to read. A departure is the same evidence in another form: a view whose coverage ends, or begins,
inside a disagreement episode while another view continues through it never resolves the disagreement, and the
surviving view is where the output lands anyway, so `_drop_snapping_views` now treats the frame where a view's
coverage ends (or begins) inside an episode like a snap and zeroes that view over the episode. A test plants the
pattern (the wrong-way view heavier and ending mid-turn; the old code is 170 deg off at the hand-off). Re-merged,
the rule touched 21 frames of one recording and nothing else: HC904 has no flip (root 7.8 to 7.7 deg), the other
five are bit-identical. The arbiter that would not need the turn to resolve is the pelvis IMU's own yaw rate, whose
sign says which view turned the right way; it is not used, because the pipeline reads the recording for the sync
and the identity only, and a third use would want its own argument.

**The side camera's gravity frame, and the shipped HC901.** With the camera 2.3 m up looking down at the mat,
GVHMR's root for the side view is pitched 6-15 deg from the truth on every rendered session (median over the
recording after a yaw fit; the along views 3-9 deg), the alignment between views is a yaw and an offset, so the
pitch stays as the views' root residual (12-23 deg, against 5-8 with the earlier side cameras), and the merged root
carries the views' tilts in proportion to their shares of it: 14.5 (HC901) and 15.8 (the TUG) where the side view
holds most of the root, 7.0-7.7 on the other walks where it holds a quarter to a half (HC900 read 5.6 while its
side view covered half the span; with the standing walker kept, that view holds 0.39 of the root and the merged
root took on more of its tilt). The same camera costs the TUG's seated and rising phases, which it sees obliquely and
close: the side view's own leg error there went from 17 and 21 to 19 and 24 deg (the walking frames unchanged at
14), and the TUG's legs over 20 deg from 16 to 29 % of the exported frames, all of it in the chair. A
full-rotation alignment of the views onto the more upright one would remove the residual and, choosing the
reference by uprightness, most of the tilt; it is the open item of this round. The shipped HC901 itself, the
reconstruction section's sample, now reads: along view alone 22.2 deg on the legs (65 % over 20, its standing-phase
fragment included; 15.5 and 11 % where its legs are in frame), side view 11.5 and 0 %, merge 11.3 and 0 % at 94 %
export, below both views on the 511 frames both have the legs (13.0 against 13.6 and 15.5), knee-flexion RMSE 6.9
(the side view 5.9: eleven frames at the side view's return from its gap at 3.8 s, where the cross-fade hands the
along view's 50-deg-wrong standing knees the weight; the along view 46 over all of its frames); the experiment's
table in the README is from the first render and says so.

**What the room-wide side view's track does not hold.** Checked against the truth's projected pelvis, the side
view's subject track covers the walker on 100 % of its frames and never another person, but it misses the walker
while the walker is in its panel on four of the six recordings: 51 % of those frames on HC900, 30 % on HC902, 33 %
on HC904, 14 % on HC905 (5 % on HC901, 0 on the TUG), nearly all of them the walker standing still before the
first bout or between bouts at the near end of the mat (inside the annotated walk bouts the misses are 0-3 %, 10 %
on HC900). Two mechanisms: on HC902 and HC904 the standing walker is a tracker fragment of its own, with no motion
along the mat for the cross-view cue to correlate, so it is never attached; on HC900 and HC905 the attached
fragment does span the standing phase, and the cue's window test (5-s windows whose progress correlation with the
other view's subject falls under 0.5 are dropped as an identity switch) drops it, because its "too little motion
to judge" guard asks for a standard deviation under 1e-6 pixels, which keypoint noise on a standing person never
is, and the correlation of two noise signals is nothing. The along view has every one of those frames and the
export loses none, so the training target is whole; what is lost is the side view's lift of the standing phases.
Two rules in `cross_view_selection` answer it: the window test judges a window only where the other view's subject
moves at least a tenth of its range (the 5th-95th percentile span inside the window), and a fragment that ends
within 2 s of an attached fragment's start, or starts within 2 s of its end, with its box centre within half a box
height of the junction and no other fragment fitting it, is attached by continuity (`attached_by_continuity`),
whatever its motion, a frame or two in common with the attached fragment allowed, since that is how a tracker
hands an id over (the first version demanded no overlap and missed the sample's 81-frame turn fragment for one
shared frame); all with tests (a standing-then-walking scene: the standing frames are kept, a standing fragment
split off by the tracker is attached, with or without the shared frames, two fragments at one junction attach
neither). On the CPU stage rerun, HC900's side view went from missing 51 % of the walker's in-panel frames to
13 %, HC905's from 14 to 0 %, HC901's along view gained its standing-phase fragment (priced in the README's HC901
section: eleven frames of a 50-deg knee at one hand-over); what remains missing is where the side view
has no fragment at all, the walker standing half out of the panel's edge at the near end of the mat before the
first bout (HC902, HC904; HC900's first five seconds, in pieces) and a detector gap at a turn there (HC900, 36.2-36.9 s), which is the
detector's limit, not the selection's. The dry run's coverage check, which compared the chosen track with the
scene (and failed once the room-wide camera never lost the walker), now judges the selection on what the tracker
produced: the chosen track must hold at least 90 % of the frames in which some fragment's box holds the subject's
hip, and the detector's own recall is reported beside it. The continuity rule lives in the cross-view path only:
HC902's side view, taken by the IMU match at a weak margin, would gain 41 frames from it, and the TUG's along view,
taken by the IMU match, lacks the seated phases (33 % of the walker's frames in that view, 71 % of what its tracker
found: a seated subject's fragments never correlate with the pelvis acceleration, and nothing attaches them; the
side view has every one of those frames). Every recording was re-lifted on that selection; the README's numbers are
from that run.

### 2.12 Round five: the turns, the torso in the export, the demo's room, and where the error sits (2026-11-10)

**The turn that was too quick.** In Sub 003's self-paced demo the reconstruction went round its second turn (24-26 s) in
0.60 s where the truth takes 1.63, peaking at 5,290 deg/s, 178 deg off at the worst frame. Neither view turned that
fast. Each view's lift had a root glitch inside the turn (two frames each: the root flips by more than 60 deg and
is back a frame or two later), the offline snap rule read the glitch as that view snapping and set it aside for the
window, and the view set aside was the one that had the turn right. `repair_root_glitches` now runs on each view's
root before alignment, fusion or anything else reads it: an excursion of more than 60 deg from both neighbours that
returns within 0.2 s is interpolated across (2 + 2 frames on HC902, 16 on HC901's along view, none on the other
four; counted per view in `merge.json` and reported as an `info:` flag). The same turn now takes 1.43 s, peaks at
230 deg/s and is 16 deg off at worst.

**Jitter in the turns.** What remained in the turns of Sub 001 and Sub 002 was the arbitration itself: where the
views disagree by more than 30 deg the rule's per-frame discounts moved the root between them frame by frame. The
arbitration weights are now averaged over the cross-fade time before they are applied (`arb_smooth`) and the fused
root is smoothed over 5 frames. Measured on the 24 turns of the six sessions (a turn = the truth's heading moving
faster than 30 deg/s for a net change of more than 60 deg, measured with 0.7 s either side; duration = 10-90 % of
the change; jitter = SD of the heading's second difference, merged over
truth): duration within 0.11 s of the truth's on average (0.47 at worst; before 0.14 and 1.03), peak rate 1.17
times the truth's (1.51 at worst; before 2.88 and 29.4), heading error at the worst frame of a turn 18 deg on
average and 43 at worst (before 27 and 178), 7.5 deg at the median frame (7.7), jitter 0.39 of the truth's on
average and 0.52 at worst (before 5.5 and 72.5; under 1 because the 5-frame smoothing also takes out what the
truth's own heading carries). The largest remaining errors are HC901's two turns at 29.0 and 42.5 s (43 deg at the
worst frame, the side view holding 88 % of the root) and HC905's at 14.9 and 23.6 s (36 and 32). `merge.json`
gained `root_turns_over_90deg_in_quarter_s`, a count of turns no person makes; it is 0 on all six and on both real
clips.

**Torso or no export.** The export rule was legs only; a frame in which no view vouched for the spine went out with
a NaN torso. It is now legs and torso (spine 1-3, neck, head; `TORSO_LEGS`), arms optional, and a frame without
either is not exported at all. The cost is where the along camera, 1 m high at the end of the mat, has the walker
close: it shows the chest down, cannot vouch for the upper spine, and when the side view has no subject on those
frames nothing does. Exports: HC900 100 to 90 %, HC901 94 to 91, HC902 100 to 90, HC903 100 (unchanged), HC904 99
to 62, HC905 100 to 99. HC904's 37 % is its standing start (the first 11.8 s: see the next paragraph); its walk and
both turns are kept. The two real clips are unchanged (95 and 100 %). The merge reports the share as an `info:`
flag, and `info:` flags no longer turn a PASS into a WARN.

**Why the side view did not register a standing walker (HC904, HC902).** The tracker has no detection of the
subject there, not a lost selection: the sample stands the walker beyond the near end of the mat, at the panel's
top edge and half out of it, and the avatars are pale, unclothed and close in tone to the wall; the person detector
finds the walker once the first step brings the body into the panel. That is a property of the sample's scene
rather than of a clinic recording, and it is to be fixed in the generator (start position inside the panel,
clothing-like colours) together with the truth's defects below. Nothing in the pipeline was changed for it.

**The demo's close-ups now stand in the room.** The three close-up panels used to draw each body at the origin on a
strip standing in for the mat, the truth in the room's frame and the reconstruction in the reference view's GVHMR
world, whose heading is arbitrary (about a quarter turn off when the along view is the reference). So the
reconstruction looked turned against the truth, and the mat under it had nothing to do with the mat in the views.
When the truth carries its room (`transl` in the truth file, the mat in `expected.json`), the close-ups now draw
all three bodies at the truth's position on the room's floor and mat (a cross line every 2 ft), seen from the side
camera's side by a camera that follows the walker; the reconstruction and the export are turned into the room by
one yaw fitted over the whole recording, the same one the scores use, and the panel prints the root-orientation
error of the frame. The merged translation is not used to place them: after one yaw and one offset it is 0.09-0.65
m from the truth at the median of a session (0.44-1.02 m at the 90th percentile), which would read as the walker
leaving the mat, and `transl_weight` already says so. Without a truth or a room the old layout is kept.

**What in the videos is the sample's own (to fix in `tools/make_sample.py`; needs a new render and a new lift).**
Three things seen in the demos are in the rendered truth, so in the videos the pipeline is given, and the
reconstruction follows them. (1) The walker drifts off the mat as he walks down it (HC900): the truth's path is
integrated from the inertial fit's heading by pinning the stance foot, and that heading drifts about 7 deg from the
mat's axis over the recording. (2) A backward slide of about 4 cm at 25.93-26.07 s of HC900: the stance foot is
chosen frame by frame as the lower one, with no hysteresis, and flickers in double support. (3) The TUG's chair
(HC903) is placed under the first frame's pelvis, and by the third repetition the same drift has the seated pelvis
(-0.75, -0.30) m from it, beside the chair at 0:36. The fixes are a heading correction per straight bout before
pinning, hysteresis on the stance foot, and closing each repetition's path on the chair. They change the rendered
videos, so every number in the README would have to come from a new lift; they are held until the GPU stage can run
again.

**The lifter's camera.** The lift hands GVHMR a default intrinsic matrix with the focal length equal to the image
diagonal. Against the samples' real cameras that is 1.53 times too long for the along view and 2.04 for the side
view. The consequence shows in the heading: with the lens taken for longer than it is, a walker off the optical
axis is read as turned by about his bearing. Over the six sessions the side view's heading error regresses on the
bearing of the viewing ray at +0.96 deg per degree (r = +0.69), its spread is 14 deg against the along view's 4.4,
and removing the bearing term (an oracle, from the known camera) takes that view's median heading error from 7.6 to
3.9 deg. Applied in the merge (a yaw per frame on the side view's root before fusion) it did not help the output,
HC901's legs got worse, the body having been solved under the same wrong camera; it is not shipped. The place for
it is the lift (the K handed to GVHMR, per view, from a field of view given in the config or measured on the mat)
and it needs a new lift.

**Where the remaining error sits.** On frames where both views have the legs, the merged legs are below both views
on five sessions and 0.5 deg above the better view on HC905; the merged knee RMSE is below both or within 0.7 deg
of the better one. The lift itself under-reads amplitude: the slope of estimated on true flexion is 0.63-0.71 for
the knee on the walks (0.86 in the TUG) and 0.47-0.59 for the hip, in each view as in the merge (side view: hip
0.64-0.69, along view 0.42-0.55), with no bias, and that term is 35-50 % of the knee's squared error on the walks
(bias 0-1 %; in the TUG bias and amplitude are 22 % each, the seated knee read 5 deg short). A weight on each
view's leg joints for how face-on it sees the sagittal plane (the squared cosine between the walker's lateral axis
and the viewing ray, from the camera-frame lift; 0.05-0.12 for the along views, 0.5-0.9 for the side views) moved
the knee RMSE by -0.3 deg on average (HC900 6.6 to 5.8; the TUG and the hurried walk worse by 0.2-0.4), the hip
slope by +0.03, and the three-axis leg median by +0.6 deg; not shipped. Scoring against the truth one frame later
changes the knee RMSE by -0.7 (HC900), -0.4 (HC901), -0.3 (HC902) and 0.1 or less elsewhere: about half a frame of
timing. In order of what they should be worth: the real field of view at the lift; one fit to both views' keypoints
in place of two single-view lifts (the cameras are near right angles; needs their relative pose, which the mat
gives); a full-rotation alignment of the views for the root's tilt; a per-axis rather than per-joint weighting of
the views.

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
* The merge's alignment of the two views' GVHMR worlds (one yaw and one offset from shared frames) and GVHMR's
  translation scale under an unknown focal length, both checked on rendered cameras only; the demo renderer
  (pyrender offscreen) on the partner's operating system.
* The pelvis path on the mat under a real video: the walkway reader runs on the public recordings (section 2.10),
  but the path it yields has only been checked against a rendered truth whose footprints were fabricated from that
  truth; a real mat under a real video is the test that remains. The mirror check has fired correctly on a mirrored
  panel of a real clip and stayed quiet on six rendered two-camera sessions, never yet on a real two-camera pair.

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
9. **Walkway cells.** The public export's `Walkway_X/Y` are per-sample lists of loaded cell indices on a 384 x 48
   grid with `WalkwayFoot` labels and pressure levels (section 2.10); the reader assumes the GAITRite's 1.27-cm
   cell (`params.walkway_cell_cm`) from the grid's size and the footprints' extents. Please confirm the mat model
   and cell pitch, which way X runs relative to the cameras, and whether `Other 1` cells are unassigned footprints
   or walking aids; with that the merge writes the pelvis path on the mat (`transl_mat`).
