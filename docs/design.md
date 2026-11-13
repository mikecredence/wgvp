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
   vertical by the better score; otherwise a single view. A seam that moving content crosses is vetoed (pixels off
   their temporal median on both sides of the boundary at the same positions, in at least two sampled frames and a
   fifth of the frames with motion at the boundary): nothing moves from one tiled view into the other, so such a
   line is a wall corner or a post inside one view. Separator bands are exempt. Each panel's content box trims edge-adjacent all-black
   lines only, so a black walkway mat inside a view is not taken for a border. The boxes are written to the QC and
   drawn on a two-level edge-map preview; `views.override` in the config forces boxes per file.
2. **Tracking per view.** YOLOv8n-pose + ByteTrack (one tracker per view), frames downscaled to 960 px for the
   detector, keypoints reported in full-frame pixels, with per detection a torso colour histogram (12 x 4 H-S bins,
   for re-identification only, kept in `work/`) and a visibility class (full: shoulders, hips and ankles seen;
   partial; none). A body whose torso is outside the picture gets no histogram. Tracker ids are split into fragments
   at box-centre jumps, at time gaps, and at persistent appearance changes where a second tracked person overlaps the
   box.
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
   subject and not the longest track. Rest hold (2026-11-10, 2.13; `session.rest_hold`, `params.rest_hold`): inside
   an IMU "Standing" or "Sitting" interval that holds a detection of the subject, frames without one keep the
   subject's box (interpolated between two detections whose boxes overlap at IoU >= 0.8, the first or last
   detection's box outside them), are lifted without a chunk split and carry `rest_held`.
7. **GPU stage** (`wgvp.gvhmr_stage`, runs in the GVHMR environment): GVHMR demo per chunk, static camera, no QC
   renders (`GVHMR_RENDER=0`), with GVHMR's default intrinsics (focal length = image diagonal) unless
   `gvhmr.focal_35mm` gives the view's 35 mm equivalent focal length (2026-11-10, 2.13; passed as `--f_mm`, recorded
   in `gvhmr_meta.json`, the matrix used saved as `K_fullimg_cut` in `kp2d_vitpose.npz`); `hmr4d_results.pt`
   converted to `smpl_global.npz` (gravity-aligned world frame) and
   `smpl_incam.npz` (camera frame) with `frame_idx`, `pts_ms`, `imu_t_s`, `visible`; ViTPose 2D keypoints mapped
   back to full-frame pixels; GVHMR's input copies and overlays deleted. Chunks are merged at the overlap midpoint;
   every frame carries `visible`, `edge_truncated` (the subject box touches a view edge), `pose_step_deg` (largest
   joint-rotation step from the previous frame) and `chunk_id`.
8. **Reconciliation between the views** on the IMU timeline: correlation and residual lag of the 2D pelvis signals
   (always) and, after GVHMR, per-joint geodesic disagreement of the SMPL-X body pose, knee/hip flexion correlations
   and the knee-signal residual lag (tolerance 3 frames); the per-frame disagreement is written to
   `pose_agreement_<a>_<b>.npz`.
9. **Merge, and the training target** (`wgvp.merge`, CPU, after the lift; `wgvp merge <root>` re-runs it alone). Both
   views come from one tiled video, so they share frame indices; the merge covers every frame some view lifted. Per
   view and frame the joints it can vouch for are those whose ViTPose keypoints are confident and inside the panel
   (`fusion.usable_joints_rt`; 8 px inside, and since 2026-11-10 4 px for the shoulders at the top edge, 2 px to
   stay in a run that passed at 4: 2.13). A rest-held frame counts as seen only when those keypoints confirm the
   body is still there (`fusion.seen_frames`). The view with more usable joints is the reference; the other view's
   GVHMR world is
   turned onto it by one yaw (circular median of the two roots' heading difference) and one offset (median
   translation difference), fitted on frames where both views see most of the body, with the residuals reported.
   Root: `fusion.fuse_root` (presence-ramped, usable-fraction-weighted; when the roots disagree by more than
   30 deg, a view turning faster than 150 deg/s since its previous frame and a view far from the previous fused
   root are discounted first (2026-11-05: a lift taking a 180-degree turn the wrong way round), then the weights
   are sharpened towards the heavier view);
   pelvis path fused from the views' per-frame steps with the same shares and summed (2026-11-10, 2.13;
   `fusion.fuse_path`: a view's horizontal step counts in full across its camera's line of sight and 0.05 of that
   along it; until then the views' positions were averaged, which slid whenever the shares changed), smoothed over
   5 frames inside runs. Body: per joint the view(s)
   that vouch for it, cross-faded over 0.5 s at segment edges, each view weighted by its root share, nothing where
   no view saw the joint, smoothed over 5 frames. Hips' and knees' flexion in two bands (2026-11-10, 2.13;
   `fusion.flexion_two_band`): the posture is that mean's, the stride's swing comes from a second mean that also
   weights each view by how side-on it sees the walker (`fusion.side_on_score`, from the camera-frame lift; written
   as `side_on`). Export: every leg joint and every torso joint (spine, neck, head) output, in runs of at
   least 0.5 s; an arm may be missing. For
   learning from body-worn sensors the root is also written heading-relative (yaw removed; velocity in the heading
   frame) and one body shape per recording is taken as the median of GVHMR's betas over whole-body frames. Loss
   weights in [0, 1]: per joint, the summed view weights (one confident view or two agreeing ones: 1) times an
   agreement factor 1 / (1 + ((d - 5 deg)+ / 20 deg)^2) on the two views' disagreement d, d scaled by twice the
   smaller view's share of the root, which is how much of the dissenting view the merged joint contains (until
   2026-11-05 the whole disagreement counted: on HC901 the along view's standing-phase fragment, set aside by the
   root rule with its knees 50 deg wrong on a legs-only crop, pulled the knee weight of that phase to 0.2 while the
   merged knee was the side view's; it is 1.0 now, and the export-mean weight 0.76 -> 0.92); for the root, the summed
   raw root weights; for the translation, the same where the path exists (until 2026-11-10 discounted by the two
   views' pelvis-position disagreement, 10 cm free, 0.5 at 40 cm; 2.13: against the truth that distance says nothing
   about the velocity's error and little about the path's, and it is recorded as `transl_disagree_m`). Mirrored view
   (2026-11-05): before the alignment, every other view is tested
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
   frame and `transl_from_walkway` the mat path back in the world. Where it exists the heading-frame velocity's
   mean over 1.5 s is the mat path's and what is faster the fused video path's (2026-11-10, 2.13; until then the
   velocity was the mat path's own derivative, which carries the footprints' scatter frame to frame and stepped
   where the mat path begins and ends). A rigid fit of each view's camera-frame onto its world-frame translation is
   reported as a
   diagnostic (`camera_fit_rms_m`); its size (0.4-1.6 m) is why the scaling is not about the camera (section 2.9).
   The body IMUs are not used
   anywhere in this stage: the recording CSV serves the sync, the subject identity and (walkway columns) the
   position reference only, so the file can supervise a sensor-to-pose model without circularity. Output:
   `merged_smpl.npz` + `merge.json` beside `reconcile.json`; `merge_status` / `export_frac` in the QC summary.
   Since 2026-11-13 (2.18) the merge is followed by the joint two-view fit (`wgvp.jointfit`, CPU, 2.15), which
   fits one body to both views' 2D keypoints, and `target.json` names the recording's training target: the fit
   where there are two views and its checks pass, the merge everywhere else. The merge's file is always there.
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
and side bars, a single letterboxed view, a single view with a wall corner mid-frame that a walker crosses, and a
view that fills only a strip of the frame beside a full-height view;
the fabricated example recording read back through the IMU reader.

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

**Why the side view did not register a standing walker (HC904, HC902).** The tracker has no detection of the subject
there, not a lost selection: the sample stands the walker beyond the near end of the mat, at the panel's top edge and
half out of it, and the avatars are pale, unclothed and close in tone to the wall; the person detector finds the
walker once the first step brings the body into the panel. That is a property of the sample's scene rather than of a
clinic recording. The generator now dresses the avatars (below), and on the new renders the chosen side-view track
covers 84.8 % of HC904 (66.6 on the bundled video; first tracked frame 96 against 427), 79.0 % of HC902 (70.0) and
99.7 % of HC900 (86.9). What is left is the detector's margin, and no tracker setting recovers it. YOLOv8n-pose gives
the standing walker of the new HC904 a confidence of 0.25 or more on 48 % of the standing frames. ByteTrack's second
association is meant to carry a track through fainter detections, but with `fuse_score: True` (the default) the IoU is
multiplied by the detection's score, so a detection under 0.25 sits at a matching distance of 0.75 or more against a
threshold of 0.5 and never matches; this is why lowering the pose confidence alone changed nothing. With the score
left unfused and the low threshold at 0.1 or 0.05, a direct tracker test covers 63 and 67 % of the 350 standing frames
instead of 43. In the pipeline the same settings break the side view up (27 fragments become 41), the selection falls
from the inertial correlation to the cross-view rule (r 0.0), and the chosen track covers 72.4 % instead of 84.8 and
starts at frame 394 instead of 96: faint detections carry erratic keypoints and descriptors. The settings were
reverted. One idea was left untested here because it changes the frames the lifter is given: where the subject's track
ends and resumes at the same place with the body at rest, hold the box over the gap. It is the rest hold of 2.13.

**The demo's close-ups now stand in the room.** The three close-up panels used to draw each body at the origin on a
strip standing in for the mat, the truth in the room's frame and the reconstruction in the reference view's GVHMR
world, whose heading is arbitrary (about a quarter turn off when the along view is the reference). So the
reconstruction looked turned against the truth, and the mat under it had nothing to do with the mat in the views. When
the truth carries its room (`transl` in the truth file, the mat in `expected.json`), the close-ups now draw all three
bodies at the truth's position on the room's floor and mat (a cross line every 2 ft), seen from the side camera's side
by a camera that follows the walker; the reconstruction and the export are turned into the room by one yaw fitted over
the whole recording, the same one the scores use, and the panel prints the root-orientation error of the frame. The
merged translation is not used to place them: after one yaw and one offset it is 0.09-0.65 m from the truth at the
median of a session (0.44-1.02 m at the 90th percentile), which would read as the walker leaving the mat (2.13
rebuilds the path from the views' steps; it is 0.16-0.34 m off at the median since). Without a truth or a room the old
layout is kept.

**What in the videos was the sample's own, and its fix in `tools/make_sample.py`.** Three things seen in the demos are
in the rendered truth, so in the videos the pipeline is given, and the reconstruction follows them. (1) The walker
drifts off the mat as he walks down it (HC900): the truth's path is integrated from the inertial fit's heading by
pinning the stance foot, and that heading drifts about 7 deg from the mat's axis over the recording. (2) A backward
slide of about 4 cm at 25.93-26.07 s of HC900: the stance foot was chosen frame by frame as the lower one, and the
inertial fit has no floor, so a swinging toe that dips below the planted one for a few frames was pinned and threw the
root backwards. (3) The TUG's chair (HC903) is placed under the first frame's pelvis, and by the third repetition the
same drift had the seated pelvis (-0.75, -0.30) m from it, beside the chair at 0:36.

The generator's path (`walker_path`) is now built in four steps. The stance foot is chosen over the whole recording by
dynamic programming (`stance_sequence`: the lower foot and the foot moving backwards under the body are preferred, a
change of foot costs, and the root velocity the pinned foot implies has to be continuous). Every straight bout
(`straight_bouts`) is turned onto the room's axis by a yaw of its own, eased through the turn between two bouts, and
the path is pinned again. Every bout is brought back to one line. Every seated run (`seated_runs`: the pelvis under 75
% of its standing height above the lower foot, and the root at rest) is brought back to the first seat's position. The
line and chair shifts are spread between their anchors in proportion to the distance the root travels, a slightly
longer or shorter step, so no foot slides while the walker stands. Measured on the new truth: HC900's four straight
bouts lie within 0.1 deg of the mat's axis, no foot joint leaves the mat in their 490 frames (the pelvis stays within
0.108 m of the centre line of a mat that reaches 0.305 m to each side), no frame inside a bout moves backwards, and
the stance changes 45 times. In HC903 all four seated runs return to the chair within 1 mm and face it within 0.1 deg,
no bout frame moves backwards, and 5 of 521 bout frames have a foot joint off the mat. In the rendered frames the
walker of HC900 stands on the mat at 25-26 s, and the walker of HC903 sits down onto the seat at 35-38 s.

The avatars are dressed (`clothed_colours`: a top, trousers, shoes, skin and hair per vertex, the top and trousers
different for each person). A uniformly pale body is missed by the person detector while it stands small and still in
the side view, and the pipeline's appearance descriptor is a torso histogram, which a room of identical bodies does
not exercise. HC901's along camera is now low and close (`close_along_cam`: 0.6 m high, the standing walker 1 m away
shown from the thighs down), a stress view for bodies cut by the top of the frame.

The new videos were held back until the GPU stage had run on them, every number in the README coming from a lift;
since 2.13 they are the bundled samples. The six new renders pass the stages before the lift (QC status PASS on all
six).

**Tracking revision 3: what clothed, cut-off bodies exposed.** With the avatars dressed and one camera close, the
colour descriptor started to matter, and four weaknesses of tracking and selection showed, all in the handling of a
body that is partly outside the picture. They are fixed with the sample (`TRACK_REV = 3`, part of the tracking cache
key, so caches of the earlier revision are not reused).

(1) A descriptor from the wrong part of the body. With the torso above the picture the descriptor's region fell on the
trousers, and the fragment's distance to the subject's own descriptor went to the far end of the range: in HC901's
close view the chosen track covered 43.4 % of the recording and began at frame 875. `torso_histogram` now returns no
descriptor when the torso is not in the picture: a box cut by the top of the frame without both hips, or with a column
over the hips shorter than 0.75 thigh lengths (measured: the column's distance to the full-torso descriptor is under
0.2 from 0.8 thigh lengths up, and over 1.4 under 0.35).

(2) Keypoints clamped to the edge. For a body part above the picture the pose model reports a point on the top edge,
with a confidence that passes in some frames: shoulders at y = 0.0 with 0.4-0.6 on a legs-only box, against 10 px and
more for shoulders that are in the picture. A keypoint now counts only when it is confident and off the top edge. Two
simpler rules were tried and rejected on the regression set. Taking the hip column whenever no head is seen lost a
19-frame standing fragment whose shoulders were in the picture (distance to the subject 0.808 instead of 0.329,
against a threshold of 0.35). Following the shoulders without the edge rule cut the bundled HC900's side view in three
places (86.9 % to 61.2).

(3) Fragments without a descriptor. Appearance cannot place them, so the IMU-match path now has a continuity rule of
its own for them, beside the cross-view path's (`_chain_blind`, reported under the same name,
`attached_by_continuity`): at each end, the subject's detection nearest in time within 2 s has to be the same box (IoU
above 0.4). The box centre alone was not enough; it also took a small static false detection beside the place where
the walker stood. A tracker that loses a faint standing walker opens a second id and then alternates between the two,
so a fragment may share frames with the subject as a second box on the same body (median IoU above 0.4), and those
frames are masked out of it. Fragments that do carry a descriptor are still not attached by continuity in this path.

(4) Appearance cuts with nobody to switch with. A fragment is cut where the descriptor changes abruptly and for good
(the distance between the medians of the 8 detections before and the 8 after is above 1.1, of 2). One person's
descriptor also changes on its own: a chair's backrest behind a seated walker in the TUG's side view gives 1.04-1.13,
and so does a body leaving the top of the frame. On the real clip, with one person in it, revision 2 cut the track at
frame 192. An appearance change now cuts only where a detection of another tracker id overlaps the box within 2 s: an
identity switch needs a second person to switch with. On that clip each view is now one fragment of 13.1 s instead of
two, the offset is 0.8205 s against 0.8212, and the selection margins are 0.56 and 0.61 against 0.45 and 0.44.

The regression set is thirteen run roots: the six new renders, the six bundled samples and the real clip. Coverage of
the chosen track, revision 2 against revision 3:

| view | revision 2 | revision 3 |
|---|---|---|
| new HC901, along (the close camera) | 43.4 %, from frame 875 | 99.7 %, from frame 91 |
| new HC905, side | 98.6 % | 99.3 % |
| bundled HC902, side | 70.0 %, from frame 413 | 73.4 %, from frame 218 |
| bundled HC903, along | 66.4 % | 79.4 % |
| the other 20 sample views, and both views of the real clip | | unchanged |

The QC status of every root is the same under both revisions, and `tests/test_tracking.py` holds the three rules. Two
of the bundled samples' chosen tracks change, so revision 3 and the new samples went to `main` together with the lift
of 2.13.

**The lifter's camera.** The lift hands GVHMR a default intrinsic matrix with the focal length equal to the image
diagonal. Against the samples' real cameras that is 1.53 times too long for the along view and 2.04 for the side view.
The consequence shows in the heading: with the lens taken for longer than it is, a walker off the optical axis is read
as turned by about his bearing. Over the six sessions the side view's heading error regresses on the bearing of the
viewing ray at +0.96 deg per degree (r = +0.69), its spread is 14 deg against the along view's 4.4, and removing the
bearing term (an oracle, from the known camera) takes that view's median heading error from 7.6 to 3.9 deg. Applied in
the merge (a yaw per frame on the side view's root before fusion) it did not help the output, HC901's legs got worse,
the body having been solved under the same wrong camera; it is not shipped. The place for it is the lift (the K handed
to GVHMR, per view, from a field of view given in the config or measured on the mat), which needed a new lift: 2.13.

**Where the remaining error sits.** On frames where both views have the legs, the merged legs are below both views on
five sessions and 0.5 deg above the better view on HC905; the merged knee RMSE is below both or within 0.7 deg of the
better one. The lift itself under-reads amplitude: the slope of estimated on true flexion is 0.63-0.71 for the knee on
the walks (0.86 in the TUG) and 0.47-0.59 for the hip, in each view as in the merge (side view: hip 0.64-0.69, along
view 0.42-0.55), with no bias, and that term is 35-50 % of the knee's squared error on the walks (bias 0-1 %; in the
TUG bias and amplitude are 22 % each, the seated knee read 5 deg short). A weight on each view's leg joints for how
face-on it sees the sagittal plane (the squared cosine between the walker's lateral axis and the viewing ray, from the
camera-frame lift; 0.05-0.12 for the along views, 0.5-0.9 for the side views) moved the knee RMSE by -0.3 deg on
average (HC900 6.6 to 5.8; the TUG and the hurried walk worse by 0.2-0.4), the hip slope by +0.03, and the three-axis
leg median by +0.6 deg; not shipped. Scoring against the truth one frame later changes the knee RMSE by -0.7 (HC900),
-0.4 (HC901), -0.3 (HC902) and 0.1 or less elsewhere: about half a frame of timing. In order of what they should be
worth: the real field of view at the lift (2.13: root 13.3 to 10.3 deg, heading 5.9 to 3.1); one fit to both views'
keypoints in place of two single-view lifts (the cameras are near right angles; needs their relative pose, which the
mat gives); a full-rotation alignment of the views for the root's tilt; a per-axis rather than per-joint weighting of
the views (2.13 ships it for the flexion axis of the hips and knees).

### 2.13 Round six: the new samples lifted four ways, a walker at rest, the cameras' field of view, and the pelvis path (2026-11-10)

2.12 left three things to a GPU run: the new renders, the box held over a walker at rest, and the cameras' field of
view at the lift. The six new renders were lifted four times on the T4, with and without each of the last two:

| lift | the lifter's camera | rest hold |
|---|---|---|
| 1 | GVHMR's default (focal length = image diagonal) | off |
| 2 | each camera's field of view | off |
| 3 | default | on |
| 4 | each camera's field of view | on |

All four are merged by the same code, the one shipped with this section, so two lifts differ in the options alone.
Lift 4 is what the bundled samples, the README's numbers and the figures now are; lift 3 is the same recordings for a
site that does not know its cameras. Against the rendered truth (medians over a recording; heading = the root's yaw;
slope = estimated on true flexion over the export frames with both views' legs, 1 = full amplitude; path = distance to
the truth's pelvis after one rigid fit; speed = RMSE of the horizontal root speed while the walker moves):

| | lift | HC900 | HC903 (TUG) | HC901 | HC902 | HC904 | HC905 | mean |
|---|---|---|---|---|---|---|---|---|
| export, % of frames | 3 | 100.0 | 99.9 | 100.0 | 100.0 | 99.8 | 99.8 | 99.9 |
| | 4 | 100.0 | 99.9 | 100.0 | 100.0 | 99.8 | 99.8 | 99.9 |
| legs, deg | 3 | 9.6 | 16.2 | 11.4 | 11.5 | 12.3 | 9.8 | 11.8 |
| | 4 | 9.5 | 16.4 | 10.7 | 11.1 | 12.0 | 9.2 | 11.5 |
| knee flexion RMSE, deg | 3 | 5.2 | 11.1 | 5.6 | 6.9 | 7.3 | 6.2 | 7.0 |
| | 4 | 5.0 | 11.9 | 5.3 | 6.9 | 7.0 | 5.7 | 7.0 |
| knee slope | 3 | 0.78 | 0.82 | 0.85 | 0.80 | 0.79 | 0.78 | 0.80 |
| | 4 | 0.78 | 0.81 | 0.84 | 0.80 | 0.79 | 0.79 | 0.80 |
| hip slope | 3 | 0.84 | 0.60 | 0.75 | 0.73 | 0.81 | 0.67 | 0.73 |
| | 4 | 0.84 | 0.59 | 0.74 | 0.73 | 0.79 | 0.69 | 0.73 |
| root, deg | 3 | 10.6 | 16.2 | 17.8 | 10.9 | 15.1 | 9.0 | 13.3 |
| | 4 | 8.1 | 15.1 | 12.5 | 7.4 | 10.5 | 8.4 | 10.3 |
| heading, deg | 3 | 3.6 | 3.6 | 8.5 | 7.0 | 9.2 | 3.3 | 5.9 |
| | 4 | 1.7 | 3.9 | 4.1 | 2.7 | 2.7 | 3.3 | 3.1 |
| spine, deg | 3 | 9.4 | 10.0 | 9.3 | 7.9 | 8.8 | 9.2 | 9.1 |
| | 4 | 9.7 | 10.4 | 9.1 | 8.4 | 9.2 | 10.2 | 9.5 |
| arms, deg | 3 | 16.4 | 16.7 | 17.0 | 15.8 | 15.1 | 17.8 | 16.5 |
| | 4 | 15.6 | 16.8 | 17.0 | 15.0 | 15.8 | 16.3 | 16.1 |
| pelvis path, m | 3 | 0.48 | 0.34 | 0.48 | 0.16 | 0.30 | 0.52 | 0.38 |
| | 4 | 0.16 | 0.19 | 0.33 | 0.18 | 0.34 | 0.21 | 0.24 |
| speed, m/s | 3 | 0.166 | 0.156 | 0.173 | 0.201 | 0.243 | 0.219 | 0.193 |
| | 4 | 0.134 | 0.153 | 0.148 | 0.216 | 0.221 | 0.185 | 0.176 |

The means of all four lifts, with the turn metrics of 2.12 over the 24 turns (mean / worst):

| | lift 1 | lift 2 | lift 3 | lift 4 |
|---|---|---|---|---|
| export, % | 96.7 | 96.7 | 99.9 | 99.9 |
| legs / knee RMSE, deg | 11.8 / 7.1 | 11.5 / 7.1 | 11.8 / 7.0 | 11.5 / 7.0 |
| root / heading, deg | 12.8 / 5.6 | 10.2 / 3.0 | 13.3 / 5.9 | 10.3 / 3.1 |
| spine / arms, deg | 9.1 / 16.4 | 9.6 / 16.1 | 9.1 / 16.5 | 9.5 / 16.1 |
| pelvis path, median / 90th percentile, m | 0.38 / 0.63 | 0.23 / 0.43 | 0.38 / 0.64 | 0.24 / 0.44 |
| speed RMSE, moving / standing, m/s | 0.192 / 0.060 | 0.177 / 0.058 | 0.193 / 0.054 | 0.176 / 0.056 |
| turn: duration off by, s | 0.15 / 0.37 | 0.12 / 0.27 | 0.18 / 0.66 | 0.12 / 0.27 |
| turn: peak rate over the truth's | 1.07 / 1.32 | 1.08 / 1.29 | 1.06 / 1.32 | 1.08 / 1.29 |
| turn: heading error at the worst frame, deg | 16.8 / 40 | 10.7 / 18 | 18.0 / 41 | 10.5 / 18 |
| turn: median heading error inside it, deg | 8.6 | 3.8 | 8.6 | 3.8 |
| turn: jitter over the truth's | 0.32 / 0.54 | 0.32 / 0.59 | 0.33 / 0.52 | 0.31 / 0.58 |

The QC status of lift 4 is PASS on two sessions and WARN on two, both for a view's lift and neither for the merge: two
and three GVHMR frames with a joint rotation jumping more than 30 deg in a frame (Sub 001's side view, Sub 002's along
view), and in the latter a fragment dropped at an identity switch. Every view is selected by the inertial correlation,
the offsets are 3.006-3.013 s and validated by the flash, and the walk bouts reconcile on all six.

**The new samples against the old.** Lift 1 is the counterpart of 2.12's numbers, which were the old renders under the
default camera: the export goes from 90.2 / 99.8 / 91.1 / 89.9 / 62.5 / 99.1 % to 100 / 100 / 99.2 / 100 / 81.4 /
99.8, with the walker on the mat throughout and the shoulder margin below. The legs are where they were (8.7 / 15.7 /
11.3 / 11.3 / 12.1 / 10.7 deg, now 9.6 / 16.2 / 11.5 / 11.5 / 12.4 / 9.8), the knee's RMSE is lower through the
stride-swing rule below (6.6 / 10.1 / 6.6 / 9.7 / 8.6 / 7.2, now 5.1 / 11.1 / 5.6 / 7.0 / 7.6 / 6.2), and the root is
higher on every recording (7.7 / 15.4 / 14.8 / 6.4 / 9.1 / 7.0, now 10.6 / 16.2 / 18.2 / 10.1 / 12.6 / 9.0; pooled
over the six, 8.3 to 11.9 deg at rest, 10.0 to 11.8 walking, 11.2 to 13.0 in the turns). The field of view takes the
root back to where it was (lift 2: 9.3, 9.5 and 10.5 in the three phases). What in the new renders raises the root
under the default camera was not isolated.

**The trunk at the top edge.** With the walker on the mat for the whole recording, the new renders bring the head and
shoulders closer to the top of both panels: the truth's shoulder joints pass 9 px below the side panel's top and 13 px
below the along panel's, and ViTPose marks the top of the shoulder 4-6 px above the joint. The usable-joint test asked
every keypoint to sit more than 8 px inside the panel, so the trunk failed it on a third of the frames in which a view
sees the subject, and under the legs-and-torso export rule of 2.12 the export of the first merge fell to 45 % of HC900
and 41 % of HC901 on a lift that was no worse. Against the truth the spine does not care about those 8 px: by the
shoulder keypoints' distance below the top edge (lift 4, per view frame), the three spine joints' median error is 4.9
deg at 5-8 px and 7.2 at 4-5, against 8.7-9.1 further inside, and rises only under 4 px (9.5 at 3-4, 13.1 at 2-3). The
shoulders' margin at the top edge is now 4 px (`shoulder_top_margin` in `wgvp.fusion.usable_joints_rt`); every other
keypoint and edge keeps 8. A fixed threshold then cut runs in which the shoulders ride the line: in HC902 39
consecutive frames went in and out of the export one or two at a time. So the margin has a hold (`shoulder_top_hold`,
2 px): a frame whose shoulders are 2-4 px inside counts when it is joined to a run that passes at 4. On lift 4, of
15,855 view frames in which the subject is seen, the trunk passes on 10,875 under the 8-px rule (spine median 9.0 deg,
90th percentile 14.4), 13,807 at 4 px (8.2, 13.9) and 14,344 with the hold (8.3, 14.1); the frames the hold adds have
a spine error of 10.2 deg (90th percentile 17.2, 5 of 520 over 20 deg). What the margin does not solve: the neck and
head are 17-22 deg off whenever the shoulders are within 5 px of the edge, the head being out of the picture, at any
margin; and a hip keypoint at the top edge is a body cut at the waist, where the legs are 21-66 deg off (about 190
view frames, all in HC901's close view), so the hips keep the full margin.

**A walker at rest the detector loses.** 2.12 ended on an untested idea for the side view's standing walker: hold the
box where the track stops and resumes with the body at rest. It is `session.rest_hold` (`params.rest_hold`, default
on). Inside an IMU "Standing" or "Sitting" interval that holds a detection of the subject, a frame without one keeps
the subject's box: interpolated between two detections whose boxes overlap at IoU 0.8 or more, the first or last
detection's box before and after them, the interval trimmed by 0.3 s where it borders another event. The frames are
lifted in the same chunk and carry `rest_held`. Nothing is taken on trust: the merge uses a held frame only if
ViTPose, run inside the held box, finds the shoulders, hips, knees and ankles where the nearest detected frame has
them (median confidence 0.5 or more, median displacement at most 5 % of the box height; `fusion.rest_held_seen`), and
a held frame that fails is a bridged frame like any other. Lifts 3 and 4 hold 232 frames of HC902's side view, 164 of
HC904's, 11 of HC901's and 2 of HC900's, all accepted, and one frame in each view of the TUG, both rejected. Against
the truth the 409 accepted frames are as good as the frames the detector did find at rest: legs 11.8 deg at the median
(90th percentile 13.1) and spine 3.5 (5.1) on lift 4, against 11.4 and 5.6 on the 2,677 view frames the detector did
find at rest. The export of HC904 rises from 81.4 to 99.8 % and HC901's from 99.2 to 100; the six average 99.9 %. The
frames gained are the standing start, and they show the default camera at its worst, the walker standing at the end of
the mat and far off the side camera's axis: under the default camera HC904's root median goes from 12.6 to 15.1 deg
and its heading from 7.0 to 9.2 with them (lift 1 to 3), with the field of view from 10.1 to 10.5 and from 2.7 to 2.7
(lift 2 to 4). The rule rests on the IMU labels and a static camera. A subject labelled at rest who is carried out of
the box fails the keypoint test; a second person stepping into a held box in front of a hidden subject is the case it
cannot exclude, and no sample has it.

**The cameras' field of view at the lift.** 2.12 left this as the first lever and as something only a new lift could
test. `gvhmr.focal_35mm` in `wgvp_config.json` now gives the lifter a view's focal length as a 35 mm equivalent on the
picture's diagonal, one number for all views or one per view; the GPU stage passes it to GVHMR (`--f_mm`), records it
in `gvhmr_meta.json` and saves the intrinsic matrix the lifter used as `K_fullimg_cut` in `kp2d_vitpose.npz`. For a
panel of W by H pixels and a horizontal field of view h, the focal length in pixels is (W / 2) / tan(h / 2) and the 35
mm equivalent is that times 43.27 / sqrt(W^2 + H^2); a panel that is a crop of the camera's picture has the crop's
field of view. The samples' cameras come to 28 mm (along view: 60 deg vertical on a 519 x 600 panel) and 21 mm (side
view: 43 deg vertical on 901 x 376), and the lifter received focal lengths of 513 and 473 px where the default had
given it 793 and 975. `tools/make_partner_layout.py` writes those two numbers into the layout's config;
`--default-camera` leaves them out. Lifts 3 and 4 differ in that option alone, as do lifts 1 and 2, and the two pairs
agree: the root's median falls from 13.3 to 10.3 deg (12.8 to 10.2 without the rest hold) and the heading's from 5.9
to 3.1 (5.6 to 3.0): the root on all six recordings, the heading on four, the TUG's and the walkway recording's
staying at 3.3-3.9. The side view's heading no longer follows the walker's bearing: its error regresses on the bearing
of the viewing ray at +0.11 deg per degree (r = +0.44) where the default camera gave +0.86 (r = +0.86), its median
heading error goes from 7.8 / 5.3 / 8.2 / 14.3 / 10.9 / 9.4 deg to 3.2 / 4.2 / 3.9 / 5.9 / 3.8 / 7.5, and that error's
standard deviation, pooled over the six, from 14.6 to 7.1 deg. The turns gain most. Over the 24 of them the heading
error at a turn's worst frame falls from 18.0 deg on average (41 at the worst) to 10.5 (18), the median error inside a
turn from 8.6 to 3.8, and the turn's duration is within 0.12 s of the truth's (0.27 at worst) where it was 0.18
(0.66). HC902's turn at 31.3 s, 1.37 s in the truth, was read as 2.03 s under the default camera and is 1.40 s, its
worst frame 38 deg off and now 8; the worst frames of HC901's three turns go from 40 / 19 / 41 to 16 / 11 / 18 deg.
The translation gains with it: the side view's own path is 0.19-0.32 m from the truth at the median where it was
0.34-0.46, the along view's 0.19-0.50 where it was 0.31-0.77, and the fit of each camera's centre from its own lift
(rms) falls from 0.79-2.12 m to 0.28-1.22 for the along view and from 0.36-0.67 to 0.19-0.36 for the side view. The
position residual between the views after alignment falls on the four recordings on which it is flagged (1.84 to 1.70
m on Sub 001's walk, 0.62 to 0.35 on the TUG, 1.10 to 0.76 on Sub 002, 1.27 to 0.44 on the hurried walk). Three things
do not gain. The legs barely move (11.8 to 11.5 deg). The spine is 0.4 deg worse (9.1 to 9.5). And the seated knee of
the TUG is not helped: over the seated frames the knee is read 14.3 deg short where it was 12.6 and the legs are at
26.5 deg (25.6), while the TUG's walks and turns improve like the others' (root 10.9 to 9.7 and 16.6 to 14.2 deg). All
of this is measured on rendered pinhole cameras whose field of view is known exactly (section 3).

**The stride's swing of the hips and knees, from the view that sees it.** The lift under-reads amplitude (2.12), and
the two views do not do so alike. On lift 4, over the moving frames, the slope of estimated on true knee flexion is
0.77-0.90 in the side view and 0.38-0.83 in the along view (0.38 on HC901 and 0.48 on HC900, whose walkers move
straight at that camera for the whole walk); for the hip, 0.80-0.91 against 0.16-0.48 on the five walks (in the TUG
0.53 and 0.65). The merge's weighted mean lands between them: knee 0.63-0.83, hip 0.58-0.69. A swing along the line of
sight is the thing one camera reads worst, and nothing in the mean's weights says so: the keypoints are confident in
both views, and the along view holds about half of the root on five of the six recordings.

The measure of it is `side_on`: per view and frame, the squared cosine between the pelvis's left-right axis and the
ray from the camera to the pelvis, both in that camera's frame (`global_orient` and `transl` of `smpl_incam.npz`). It
is 1 where the camera looks along that axis and the legs flex in the image plane, 0 where the walker comes straight at
the camera, and the same for a mirrored picture. Its recording means are 0.05-0.11 for the along views and 0.60-0.80
for the side views.

Weighting each view's whole hip and knee rotation by it was tried in 2.12 and again here, per joint and per axis, and
fails the same way each time: the legs' three-axis median rises from 11.65 deg (mean of the six, lift 2) to 12.0-12.3.
The side view is the worse one about the hips' other two axes, and where the along view has the legs it carries their
posture better. So the rule takes from the side-on weighting only what that weighting is right about. For the two hips
and the two knees, the fused rotation is split into its flexion about the joint's own x axis and the rest (a
swing-twist split). The rest stays the weighted mean's, and so does the slow part of the flexion angle, the posture.
The fast part, the stride's swing, comes from a second mean of the views whose weights are multiplied by (0.05 +
`side_on`):

    angle = base + (d - lowpass(d)),   d = alt - base

with `base` and `alt` the two means' flexion angles and the low-pass a Gaussian of sigma 0.75 s, a normalised
convolution over the frames on which both exist. It applies when two or more views have their camera-frame pose;
otherwise the merge is the weighted mean as before. `flexion` in `merge.json` says which was used,
`per_view.<view>.side_on_mean` gives the means above, and `side_on` (V, T) is in `merged_smpl.npz`. The code is
`wgvp.fusion.side_on_score` and `flexion_two_band`, called from `wgvp.merge`.

Lift 4 without and with the rule, recomputed outside the merge from the same view outputs (RMSE and median over the
export frames, slopes over the export frames outside the rest intervals, so a few hundredths from the first table's):

| | HC900 | HC903 (TUG) | HC901 | HC902 | HC904 | HC905 |
|---|---|---|---|---|---|---|
| knee-flexion RMSE, weighted mean (deg) | 6.6 | 12.1 | 6.3 | 7.4 | 6.9 | 6.6 |
| with the rule | 4.9 | 12.0 | 5.3 | 7.1 | 7.2 | 5.6 |
| knee slope, weighted mean | 0.63 | 0.83 | 0.70 | 0.69 | 0.70 | 0.70 |
| with the rule | 0.83 | 0.87 | 0.84 | 0.80 | 0.79 | 0.78 |
| hip slope, weighted mean | 0.58 | 0.69 | 0.62 | 0.65 | 0.60 | 0.60 |
| with the rule | 0.84 | 0.71 | 0.78 | 0.75 | 0.73 | 0.73 |
| legs, three-axis median, weighted mean (deg) | 9.6 | 16.6 | 10.8 | 11.1 | 12.2 | 9.3 |
| with the rule | 9.5 | 16.5 | 10.7 | 11.1 | 12.0 | 9.2 |

The export, the root, the spine, the arms and the turn metrics do not move. Averaged over the six, the knee RMSE goes
from 7.65 to 7.01 deg (on walking frames 8.3 to 7.1, at rest 5.5 and 5.5), the hip-flexion RMSE from 12.58 to 12.11,
the share of leg-joint errors over 20 deg from 12.7 to 11.7 %, the knee's slope from 0.71 to 0.82 and the hip's from
0.62 to 0.76. Two recordings do not gain: the hurried walk's knee RMSE is 0.3 deg worse although its slope rises, and
the TUG moves by 0.1. The constants matter little (measured on lift 2): the mean knee RMSE is 7.16-7.26 for a floor of
0.02 to 0.3 and the cosine to the power 1, 2 or 4, and 7.10-7.16 for sigma between 0.4 and 1.0 s. The shape of the
filter does matter: a 1.5-s boxcar has a negative side lobe at stride frequency (-0.21 at 0.9 Hz), so one minus it
passes the swing at 1.21 and over-reads it. What the rule adds is motion and not noise: the ratio of the estimate's
second difference to the truth's rises from 0.52-0.54 to 0.61-0.69 for the knee and from 0.44 to 0.54-0.58 for the
hip, and stays under 1. On the round-7 lift of the old renders the gain is smaller (knee RMSE 8.14 to 7.94).

**The lifter runs half a frame early.** Scored against the truth shifted in time, each view's knee and hip flexion
fits best a quarter to half a frame (8-17 ms) ahead of it, in every recording: on lift 4 the side view's knee is at
-0.50 frames on five recordings and -0.25 on HC905, the along view's at -0.25 to -0.75 (and -1.75 on HC900 and HC901,
where that view reads little of the swing and the lag is poorly determined), the merge's at -0.50 (-0.25 on HC905).
ViTPose's 2D keypoints against the truth's projected joints show no lead (within 0.25 frames either way), and the
sample generator is frame-exact, so the lead arises inside GVHMR, between its keypoints and its pose. It is worth
0.3-0.9 deg of the knee's error. It is recorded here and not corrected: a constant fitted to six rendered recordings
is a property of one lifter's weights, and a shift of the export against the IMU timeline is the kind of error the
pipeline exists to prevent.

**The pelvis path, fused from steps.** The merged `transl` was the share-weighted mean of the views' aligned
positions. The views' paths do not agree after alignment (at the median they are 0.04 / 0.36 / 2.49 / 0.72 / 0.57 /
0.39 m apart on lift 4), so whenever the shares changed, a view losing the legs or a cross-fade between arbitration
states, the mean travelled from one view's path toward the other's: the merged pelvis slid over the floor while the
walker stood, and steps were stretched or shortened in the walk. In the demo's room panel, which draws the
reconstruction at its own `transl`, that is a body gliding with its feet planted. The speed shows it: under the old
rule the horizontal speed of the root on lift 4 has an RMSE of 0.323 m/s while the walker moves and 0.152 while they
stand, and reads 0.70 of the truth's speed.

The path is now built from steps (`wgvp.fusion.fuse_path`). Each view's frame-to-frame step is taken in the reference
view's world; its horizontal part counts in full across that view's line of sight and at 0.05 (`PATH_ALONG_SIGHT`)
along it, a camera measuring a movement across its picture well and one in depth badly; the weighted steps are fused
with the views' shares and summed; the vertical step is the share-weighted mean. A sum of steps does not know where it
is, so every run is set on the share-weighted mean position by the median offset over the run. A change of shares now
changes whose steps are counted from that frame on and moves nothing. Lift 4 merged with the old rule and with this
one (speed = horizontal root speed against the truth's; slope = estimated on true):

| | rule | HC900 | HC903 (TUG) | HC901 | HC902 | HC904 | HC905 | mean |
|---|---|---|---|---|---|---|---|---|
| speed RMSE, moving, m/s | old | 0.470 | 0.132 | 0.536 | 0.247 | 0.296 | 0.254 | 0.323 |
| | steps | 0.134 | 0.153 | 0.148 | 0.216 | 0.221 | 0.185 | 0.176 |
| speed slope | old | 0.31 | 0.97 | 0.88 | 0.76 | 0.56 | 0.73 | 0.70 |
| | steps | 0.87 | 1.26 | 0.84 | 0.81 | 0.70 | 0.83 | 0.88 |
| speed error, 99th percentile, m/s | old | 1.17 | 0.34 | 2.02 | 0.60 | 0.82 | 0.66 | 0.93 |
| | steps | 0.40 | 0.34 | 0.47 | 0.63 | 0.62 | 0.47 | 0.49 |
| speed RMSE, standing, m/s | old | 0.249 | 0.086 | 0.136 | 0.273 | 0.071 | 0.097 | 0.152 |
| | steps | 0.039 | 0.061 | 0.038 | 0.063 | 0.065 | 0.068 | 0.056 |
| path error, median, m | old | 0.13 | 0.15 | 0.36 | 0.26 | 0.17 | 0.09 | 0.19 |
| | steps | 0.16 | 0.19 | 0.33 | 0.18 | 0.34 | 0.21 | 0.24 |
| path error, 90th percentile, m | old | 1.23 | 0.21 | 0.78 | 0.38 | 0.36 | 0.35 | 0.55 |
| | steps | 0.34 | 0.52 | 0.60 | 0.37 | 0.55 | 0.28 | 0.44 |

The speed error halves and the slide while standing goes to a third; the other lifts give the same (moving-speed RMSE
0.324 to 0.177 on lift 2, 0.399 to 0.193 on lift 3). It is a trade and not a pure gain. The median position error is
higher on three recordings (the hurried walk 0.17 to 0.34 m, the walkway recording 0.09 to 0.21, the TUG 0.15 to
0.19), a sum of steps drifting where a mean of positions cannot, and the TUG is the one recording the old rule served
better: its speed is over-read by a quarter (slope 1.26; 1.27-1.28 on the other lifts) and its 90th-percentile
position error goes from 0.21 to 0.52 m. The export keeps the steps because what a sensor model is trained on is the
root's velocity in the heading frame, and a position in a room that no IMU can know is the lesser loss.

**The velocity on the mat.** Where a session has walkway data, the pelvis path on the mat (`transl_from_walkway`) is a
measured position, 0.03 m from the truth at the median on HC905 (0.11 at the 90th percentile, 1,016 frames), against
0.21 m for the video path. Its derivative is not a good velocity: built from footprints, it steps at each stance, and
its own forward velocity has an RMSE of 0.356 m/s where the video path's has 0.196. The exported velocity used to be
the derivative of the mat path where it exists and of the scaled video path elsewhere, which adds a jump at each
change of source: recomputed on lift 4 that rule gives 0.348 m/s over the moving frames and 0.736 within 4 frames of a
switch (0.552 and 2.47 on lift 3). The velocity is now the fused video path's everywhere, and inside a mat run a
moving average over 1.5 s (`MAT_VEL_S`, a stride and a half) of the difference between the mat path's velocity and the
video's is added to it: the mat sets the mean speed and the video the shape inside the stride. On lift 4 the exported
forward velocity has an RMSE of 0.185 m/s (bias +0.009; 0.097 at the switches), 0.188 on lift 3. The mat's part in
that is small, 0.196 to 0.185: what the walkway gives the export is the position, and the velocity no longer pays for
it.

**`transl_weight` and the position residual.** `transl_weight` carried a factor that fell with the distance between
the views' positions, and the QC flag for a large position residual between the views was a WARN. Against the truth
that distance says nothing about the velocity (the old factor against the velocity's error: Spearman rho -0.03 on lift
4, the RMSE 0.25-0.28 m/s in every bin of the factor) and little about the path: frame by frame the path's error
follows it at rho +0.41 pooled over lift 4 (0.17 m at the median where the views are under 0.4 m apart, 0.30 m above)
and +0.18 to +0.22 under the default camera, where the median error is no higher above 0.4 m than below (0.42 against
0.46-0.47 m); within a recording the sign is not even stable (-0.24 to +0.47 on lift 4, -0.19 to +0.65 under the
default camera). With the path built from steps the factor also discounted the wrong thing, since two views can sit a
metre apart and agree on every step. `transl_weight` is now `root_weight` wherever the path exists; the distance is
kept as `transl_disagree_m` and its median as `translation.views_disagree_median_m` in `merge.json`, for whoever wants
to threshold on it. The residual is computed on the moving frames only (`transl_residual_frames`) and above 0.3 m it
is an `info:` flag, not a WARN: a monocular depth disagreement that the step rule is built to live with does not make
a session's merge suspect.

**Where the remaining error sits now.** On lift 4, the medians over the six are legs 11.5 deg, root 10.3, spine 9.5,
arms 16.1 and heading 3.1, with a knee-flexion RMSE of 7.0 deg; by phase the root is 10.1 deg at rest, 9.4 walking and
10.4 in the turns, the legs 9.0, 12.4 and 13.7. The largest single piece is the TUG's seated phase: root 21.5 deg,
legs 26.5, the knee read 14 deg short, and no option of this round touches it. The stride's amplitude is at 0.80 for
the knee and 0.73 for the hip, the lifter's half-frame lead is uncorrected, the speed is read at 0.88 of the truth's
(1.26 in the TUG) and the path is 0.24 m off at the median. In order of what they should be worth: one body fitted to
both views' keypoints at once, which would replace the mean of two monocular lifts and is the only route to the depth
each view lacks; the seated posture, where both views see a body folded toward one of them; the arms, at 16 deg the
least accurate part of the export and outside the export rule by design; and, for real cameras, a measured field of
view, which this round shows to be worth 3 deg of root and half of the heading error in the turns and which no
recording from a clinic has yet had.

### 2.14 Round seven: a camera's field of view read from the room, and a level floor on the real clip (2026-11-10)

Round six measured what the field of view is worth on rendered cameras and left every real camera without one. The
bundled phone clip is the one real recording in the repository and its file does not name the lens, so this round
reads the field of view from the picture, lifts the clip with it, and looks for a check that needs no truth.

**The estimator.** `tools/estimate_focal.py` uses the room. Floor and ceiling edges along it, the same across it, and
door frames and wall corners up it are three perpendicular families of lines; each family aims at one vanishing point
in the picture, and two perpendicular directions with vanishing points v1 and v2 satisfy (v1 - c) . (v2 - c) = -f^2
with c the picture centre. Per frame it takes line segments (OpenCV's LSD, a second pass on a contrast-stretched
picture for faint edges), finds the focal length and room orientation that put the most segment length on the three
points (RANSAC over pairs of points, then least squares), and reports the median over the frames in millimetres of 35
mm equivalent on the diagonal, the unit of `gvhmr.focal_35mm`. A frame counts only if two directions each carry 6
segments and 5 % of the segment length, the second direction leans 6 deg or more out of the picture plane, and the
focal length lies in 9-110 mm; when fewer than half the frames count the tool refuses and says the room's lines do not
fix the focal length from this view.

**Its checks.** On the rendered side camera (21.15 mm) it reads 21.1, 21.1, 20.9 and 21.4 mm on Sub 001-004. On the
rendered along camera (28.3 mm), which stands square to the floor grid, it refuses in all four: two of the three
families are parallel in the picture there, which is the case the rule is for. On the phone clip it reads 17.3 mm
(quartiles 17.2 to 17.5 over 12 frames, pairs of directions on their own 16.0 to 17.8; a 103-deg diagonal), the same
on the bundled 720 x 1280 clip as on the 4K original, and 20.2 and 32.2 mm on the two zoomed panels of OWN02, where
17.3 times each panel's zoom (1.171 and 1.858) predicts 20.2 and 32.1. Those are consistency checks: nothing here
knows the phone's lens.

**A level floor.** The check without a truth is the floor. `merge.path_slope_deg` fits the pelvis height against
distance along the walk's main horizontal direction over the moving frames (none for a walk that reaches under 2 m),
`merge.json` records it per view as `floor_slope_deg`, and the no-truth demo prints it on one line. The clip was
lifted nine times on the T4, the full picture at 43.3 (the lifter's default), 33, 30, 27, 20, 18, 17, 16 and 14 mm and
each panel at that value times its zoom (the default row gives both panels 43.3):

| focal (mm) | slope full (deg) | slope left | slope right | pelvis height range (m) | panels' pelvis positions apart (m) | root residual (deg) | legs apart after mirroring (deg) |
|---|---|---|---|---|---|---|---|
| 43.3 | 21.9 | 17.9 | 5.2 | 2.83 | 0.54 | 8.1 | 5.3 |
| 33 | 14.7 | 15.2 | 14.9 | 1.90 | 0.10 | 7.2 | 5.3 |
| 30 | 12.2 | 12.3 | 12.3 | 1.61 | 0.08 | 7.2 | 5.2 |
| 27 | 9.7 | 10.1 | 9.0 | 1.31 | 0.08 | 7.1 | 5.2 |
| 20 | 3.1 | 2.8 | 1.6 | 0.58 | 0.10 | 7.1 | 5.1 |
| 18 | 1.3 | 1.1 | 0.0 | 0.41 | 0.11 | 7.2 | 5.2 |
| 17 | 0.5 | 0.5 | 0.4 | 0.34 | 0.10 | 7.2 | 5.1 |
| 16 | 0.2 | 0.1 | 0.9 | 0.26 | 0.10 | 7.3 | 5.1 |
| 14 | 0.6 | 0.4 | 0.9 | 0.17 | 0.10 | 7.3 | 5.0 |

The signed slope crosses zero at about 15.6 mm for the full view, 15.9 mm (in full-picture terms) for the left panel
and 18.0 mm for the right, and the floor is level to 1.3 deg or better from 14 to 18 mm: the floor brackets the lens and the
lines' 17.3 sits inside the bracket. A second reading that needs no truth agrees. The speed of the body in the
camera's frame is 2.83 and 3.05 m/s on the two legs of the walk under the default against 1.25 and 1.26 in the world
frame (camera depth 3.2 to 21.4 m); at 17 mm it is 1.16 and 1.24 against 1.10 and 1.13 (depth 1.3 to 8.6 m, reach 6.84
m against 7.79), at 16 mm 1.10 and 1.18 against 1.08 and 1.12, and at 14 mm it falls below the world speed (0.98 and
1.05 against 1.06 and 1.10). The two speeds meet between 16 and 17 mm.

Two things follow for the composite. The 0.54 m between its two panels' pelvis positions, which the README had put
down to the depth two zoom levels disagree about, was one lens handed to two zoom levels: with each panel's own focal
length it is 0.08-0.11 m at every rung, 33 mm included, so it tests the ratio between the panels and not the lens. And
the body does not move: export 94.7 % (two runs) for OWN01 and 100 % for OWN02 at every rung, the mirror check 44.2
deg as lifted against 7.5 swapped over 224 frames at 17 mm, forward lean 5.3 and 5.8 deg against 5.7 and 5.8, the knee
curves on top of each other. The demo's close-ups therefore look as they did, and the change is in the exported path.
The bundled config now carries 17.3, 20.2 and 32.2 (the lifter takes whole millimetres: 17, 20 and 32), and the
README's figure is from that lift.

**How far the floor check can be trusted.** On the six rendered sessions of round six the same slope, read from the
lifts under the default camera and under the true field of view, has a median over the twelve views of 2.6 and 0.9
deg, with 8 of 12 views more level (per view, default then true, left and right: Sub 001 self-paced 7.0 to 0.4 and 2.3
to 1.4; Sub 001 TUG 3.3 to 4.1 and 0.2 to 0.6; Sub 002 11.3 to 18.4 and 2.4 to 1.6; Sub 003 hurried 2.8 to 0.1 and 2.8
to 2.2; Sub 003 self-paced 1.1 to 1.3 and 3.2 to 0.2; Sub 004 on the mat 0.4 to 0.1 and 1.1 to 0.4). One view gets
worse by 7 deg with the right lens, so a slope is an indicator and a flag is not raised on it. It also senses the
field of view only through the camera's tilt: the phone looks about 14 deg down, and a level camera would read zero
whatever the lens.

**What stays open on this clip.** The 0.34 m of pelvis height range left at 17 mm is several times a stride's rise and
fall, so the lifted floor is level on average and not flat. The clip is hand-held, with about 26 deg of pan in its
first 3 s and a little in its last second, and the lift runs with a static camera. The lens itself has not been read
from the phone. None of the three has a truth to be scored against.

**Small things.** The demo's slope readout is one line because two collided with the export panel's label.
`tools/remote_gpu_stage.sh` and `remote_gpu_e2e.sh` no longer copy the local Python environment to the GPU box. The
suite is at 82 tests (five for the estimator on synthetic room edges, one for the slope on a ramp and a level floor).

### 2.15 Round eight: one body fitted to both views' keypoints, compared with the merge (2026-11-11)

Round six named the largest remaining lever: one body fitted to both views' 2D keypoints at once, in place of a mean
of two single-view lifts. This round builds it as an optional stage (`wgvp jointfit`, `wgvp/jointfit.py`), runs it on
the six rendered sessions and scores it beside the merge on the same frames (`tools/compare_jointfit.py`; every
number below is in `docs/jointfit.json`). The merge stays the default training target and `jointfit_smpl.npz` is
written beside it (until round eleven, 2.18). Two things could not be checked. There is no real two-camera recording
with a truth, so every number here comes from rendered rooms. And the fit's settings were chosen on these six
sessions, with none held out.

**Why it can help.** The merge takes each joint from the view that can vouch for it, but each view was lifted on its
own. A limb's depth along a camera's line of sight is that lift's guess, and the pelvis path is two monocular depth
estimates fused. Two views of one body fix both by triangulation once the pose of one camera against the other is
known. No calibration comes with the videos, so the fit finds it from the walker.

**The fit.** Per recording, in the frame of camera a, which is the merge's reference view.

1. Seeds. The pose of camera b against camera a is the rigid fit of the two lifts' in-camera joints on the frames
   where both views detected the subject. The body starts at the merge's pose, the pelvis at the lifts' in-camera
   pelvis, and the world at the rotation that takes the merge's root onto view a's in-camera root.
2. Unknowns. Per frame a correction of the root and of each body joint and the pelvis position; per recording the
   body shape, the pose of camera b, the world rotation, and six left/right-mirrored offsets between ViTPose's
   keypoints and the model's joints (a keypoint marks the surface of a shoulder or a hip and not the joint centre).
3. Loss. The reprojection of the twelve body keypoints in both views, in metres at the body's depth so that a near
   and a far camera count alike, under a Geman-McClure loss with a 3 cm scale that sets a keypoint aside once it is
   about 9 cm off; a prior that holds every correction near zero (root 0.05 rad, limbs 0.20, spine and collars 0.10);
   smoothness of the corrections and of the pelvis's acceleration; the shape near the lifts' mean.
4. The up direction, below.
5. Output. The merge's root with the fitted correction, and the fitted pelvis carried into the merge's world and set
   on the merge's path by one constant offset. Joints and frames the merge does not output are not output here
   either: the fit refines what the merge exports and never adds to it. Adam, 150 + 450 steps, 4-7 s per recording
   on a laptop CPU for 1100-1600 frames.

**The up direction.** Two views cannot see the pelvis's pitch from keypoints, and a world tilted by a few degrees
costs the reprojection nothing, so the up direction has to be pinned from elsewhere. The lifts give it, in part: each
lift knows its camera's roll to 1-2 deg and not its pitch (the high side camera looks 31 deg down and its lift reads
about 10 deg less). The image direction that is level in the world needs the roll only, and the two cameras' level
directions, 75-85 deg apart, fix the up direction as their cross product. With it the fitted world's up is 0.8-4.3 deg
off the truth on five recordings and 8.5 on Sub 002, and 0.9-2.7 deg when the true camera pose is given. The world's
yaw stays free and is held by the root prior, so the output's heading is the merge's.

**The measure.** The truth is an SMPL body and both estimates are SMPL-X, and one posture has different joint
rotations in the two models. An SMPL-X body fitted to the truth's joint positions to 0.9 cm still differs from the
truth's rotations by 2.6 deg at the root, 4.8 at the legs, 19.1 at the spine and 6.0 at the arms. The three-axis
rotation errors of the earlier rounds therefore stand on a floor no estimate can go under, and they are kept in the
table but are not the measure of this round. The primary measures come from joint positions: the direction of each
limb segment after one yaw fit per recording; the knee's angle and the thigh's sagittal angle (RMSE, and gain as the
slope of estimate on truth); the pelvis frame from the two hips and the first spine joint, its constant offset
removed; joint positions about the pelvis; the path after one rigid fit; the speed. Frames are the merge's export,
and a joint counts where the merge outputs it and every joint above it in the chain, for all three columns alike.

**Result.** Medians over the six recordings. "True cameras" is the same fit with the renderer's camera pose passed in
and held: the bound that a calibration would reach.

| | merge | joint fit | joint fit, true cameras |
|---|---|---|---|
| thigh direction (deg) | 6.05 | 3.81 | 4.27 |
| shank direction (deg) | 5.48 | 4.70 | 3.78 |
| upper arm direction (deg) | 8.95 | 5.98 | 5.42 |
| forearm direction (deg) | 6.87 | 5.84 | 5.23 |
| trunk direction (deg) | 4.03 | 3.54 | 3.37 |
| knee angle RMSE (deg) / gain | 6.23 / 0.74 | 5.61 / 0.77 | 5.65 / 0.77 |
| thigh sagittal RMSE (deg) / gain | 5.68 / 0.76 | 4.37 / 0.83 | 4.47 / 0.85 |
| ankle position (cm) | 8.56 | 6.79 | 6.80 |
| limb position (cm) | 7.93 | 7.24 | 7.03 |
| path, median / 90th percentile (m) | 0.202 / 0.443 | 0.046 / 0.123 | 0.016 / 0.033 |
| speed, estimate over truth | 0.974 | 1.038 | 1.003 |
| speed RMSE (m/s) | 0.169 | 0.067 | 0.044 |
| pelvis frame (deg) | 5.06 | 4.78 | 4.66 |
| heading (deg) | 2.98 | 3.16 | 3.11 |
| leg directions at rest / walking / turning (deg) | 4.15 / 6.66 / 6.89 | 3.22 / 4.84 / 4.78 | 3.31 / 4.86 / 4.37 |
| TUG seated: root rotation (deg) | 21.53 | 22.55 | 22.71 |
| TUG seated: thigh / shank direction (deg) | 17.74 / 6.21 | 15.18 / 5.01 | 14.05 / 3.75 |
| root rotation, three-axis (deg) | 9.44 | 9.77 | 9.53 |
| leg / spine / arm rotations, three-axis (deg) | 10.87 / 9.41 / 16.06 | 10.13 / 9.18 / 15.90 | 10.34 / 9.16 / 15.88 |

Per recording, merge then joint fit then true cameras:

| | Sub 001 self-paced | Sub 001 TUG | Sub 002 self-paced | Sub 003 self-paced | Sub 003 hurried | Sub 004 on the mat |
|---|---|---|---|---|---|---|
| thigh direction (deg) | 4.2 / 3.7 / 4.1 | 9.6 / 6.5 / 6.2 | 4.4 / 3.2 / 4.3 | 6.6 / 3.9 / 4.2 | 6.8 / 5.2 / 4.8 | 5.5 / 3.4 / 3.7 |
| shank direction (deg) | 5.0 / 3.8 / 2.9 | 5.8 / 4.6 / 4.1 | 6.3 / 4.9 / 3.4 | 5.2 / 5.0 / 4.7 | 7.4 / 4.8 / 4.2 | 4.0 / 2.6 / 2.5 |
| ankle position (cm) | 11.0 / 7.8 / 7.6 | 11.3 / 9.9 / 7.3 | 8.1 / 6.6 / 6.4 | 8.7 / 6.9 / 6.6 | 8.4 / 6.7 / 7.0 | 6.4 / 5.7 / 6.1 |
| path, median (m) | 0.162 / 0.050 / 0.008 | 0.194 / 0.103 / 0.037 | 0.329 / 0.042 / 0.015 | 0.181 / 0.042 / 0.016 | 0.342 / 0.064 / 0.017 | 0.210 / 0.019 / 0.016 |
| pelvis frame (deg) | 3.6 / 3.2 / 3.1 | 8.4 / 8.5 / 8.5 | 6.3 / 6.1 / 5.6 | 4.7 / 4.4 / 4.3 | 5.1 / 4.6 / 4.3 | 5.0 / 5.0 / 5.0 |
| root rotation (deg) | 8.1 / 8.1 / 7.8 | 15.1 / 15.1 / 15.2 | 12.5 / 12.5 / 12.0 | 7.4 / 7.7 / 7.6 | 10.5 / 10.6 / 10.1 | 8.4 / 9.0 / 8.9 |
| camera rotation / position error | 2.4 deg / 0.12 m | 3.4 / 0.32 | 7.2 / 0.57 | 1.6 / 0.05 | 3.4 / 0.32 | 3.7 / 0.05 |
| fitted over true camera distance | 1.016 | 1.057 | 1.003 | 1.005 | 1.010 | 0.999 |
| reprojection median, camera a / b (px) | 2.1 / 1.5 | 2.6 / 2.2 | 1.2 / 4.2 | 1.4 / 2.2 | 1.9 / 2.3 | 2.0 / 2.4 |

**What improves.** The thigh, the shank, the ankle's position and the path are better than the merge's on every one of
the six recordings. The path is the largest step: 0.20 to 0.05 m at the median and 0.44 to 0.12 m at the 90th
percentile, with the speed's RMSE from 0.17 to 0.07 m/s. The thigh's direction goes from 6.0 to 3.8 deg, the upper
arm's from 9.0 to 6.0, and the thigh's sagittal gain from 0.76 to 0.83: the depth each view lacks is where the merge
lost amplitude, and a second view supplies it. The step grows with the motion: 0.9 deg on the leg directions at rest,
1.8 walking and 2.1 turning.

**What does not.** The root's rotation is unchanged (9.4 and 9.8 deg), the heading is unchanged (3.0 and 3.2), and the
seated root of the TUG is unchanged (21.5 and 22.6). The first two follow from the design: the root is held to the
merge's by a tight prior, since twelve keypoints fix where the pelvis is and say little about how it is turned. The
seated root is the lifter's reading of a sitting body in both views, which no second view of the same keypoints
corrects; the seated thigh improves by 2.6 deg and stays at 15. A looser prior on the root was tried and did not
improve it. The knee's gain moves from 0.74 to 0.77 and no
further. The three-axis rotation medians move by less than 1 deg in either direction, as the floor predicts.

**Self-calibration.** The fitted camera pose is 1.6-3.7 deg and 0.05-0.32 m off the true one on five recordings, and
it moved 3.7-7.9 deg from its seed. The gap between the second and third columns is what that error costs: the path
(0.046 against 0.016 m), the shank's direction (4.7 against 3.8 deg) and, on the TUG, the ankle (9.9 against 7.3 cm).
Scale is the other part. Reprojection fixes the scene up to one factor, which the body's size as the lifts saw it
sets: the fitted distance between the cameras is 0-6 % long, and the speed reads 1.04 times the truth where the true
cameras give 1.00. A measured camera pose, or the subject's height, would remove both.

On Sub 002 the along camera stands so close that both views show the whole body in 30 % of the frames. There the
calibration is the worst of the six (7.2 deg, 0.57 m, up 8.5 deg off), the robust loss sets aside 28 % of that view's
keypoints (0.1-5 % elsewhere), and the limbs still come out better than the merge's (thigh 4.4 to 3.2 deg, path 0.33
to 0.04 m). The stage notes the case in `jointfit.json` (`views_complete_frac` below 0.5).

The choice of camera a matters through the calibration only. In an earlier setting of the fit, with the left view as
camera a on every recording, Sub 002's thigh direction was 7.1 deg, and with the right view 3.2; on the other five the
two orders differed by 0.4 deg or less in that measure, and with the true cameras they agreed to 0.1 deg at the
median. The rule in the code, camera a = the merge's reference view, picks the right view on Sub 002 and on the
hurried walk. It was set after seeing these six.

**A body at rest.** The fit follows the keypoints frame by frame, and a standing body shimmered: the limbs' second
difference at rest was 0.22 cm per frame squared, against 0.08 for the truth's own motion and 0.06 for the merge. The
output is therefore a 3-frame moving average inside runs of emitted frames (the merge uses 5). That brings the rest
figure to 0.09 and costs the knee 0.14 deg of RMSE and 0.01 of gain; 5 frames would cost 0.45 deg and 0.04. While
walking both estimates are smoother than the truth (0.56 and 0.60 against 0.90).

**Three defects found on the way, all fixed.** They concern the merge and the demo, not only the new stage.

- The moving average cut its window on the outer side only at the end of a run, which pulls a moving value toward the
  run's inside. The merged pelvis path lost half of its last two steps when the walker was moving at the cut (Sub 001
  self-paced: 2.5, 2.4, then 1.2 and 1.1 cm; now 2.3 and 2.1), and a joint's rotation lagged on the two frames at
  either end of each of its runs. The position and rotation smoothers now shrink the window on both sides
  (`moving_average_in_runs(..., centred=True)`, `smooth_rotations`). In the path four frames per recording changed, by
  up to 2.1 cm. In the pose 4 to 44 frames per recording changed (614 joint-frames in all, median 0.1 deg, 90th
  percentile 1.1; the largest, 37 deg, is an elbow on the last frame before Sub 002's arm is dropped), and the legs by
  4 deg at most. No median of the merge in the table above moves at its second decimal. The slow mat-velocity trend
  keeps the one-sided rule, which is right for it.
- The demo drew its "reconstruction" panel from the stored pose after a second 5-frame smoothing, so that panel
  looked steadier than the file it stood for. It now draws the stored pose; only the red fill for joints no view
  vouched for is smoothed. Demo videos rendered before this round are not comparable with later ones in that panel.
- `wgvp merge` counted a recording whose merge had raised an error as merged, and the file of the earlier run stayed
  in place. It now says how many failed and returns a non-zero exit code.

**The demo.** `wgvp demo --target jointfit` draws the same video with `jointfit_smpl.npz` in place of the merge
(`<recording>_demo_jointfit.mp4`), so the two can be watched side by side.

**The fit on the bundled phone clip, and what one camera knows about distance.** The composite of our own clip
(OWN02) is two panels cut from one camera, so the fit is a test of what it does when the second view adds nothing.
It ran in 3 s, met the keypoints to 1.1 and 1.5 px, and placed the second camera 2.4 cm from the first and 0.16 deg
turned against it. The level axes of the two views are 0.24 deg apart, so the up direction fell back to the first
view's gravity estimate. `jointfit.json` now flags a baseline under 0.3 m (`min_baseline_m`; the six rendered rooms
have 5.2 to 6.6 m): the views are one viewpoint, and the distance to the walker rests on the body's size in the
picture and on the lens. The pose stays within 0.5 deg (root), 1.3 (legs) and 0.9 (arms) of the merge's at the median.
The path does not: 0.48 m apart at the median and 0.69 at the 90th percentile, a reach of 7.7 m against 6.9, a median
walking speed of 1.29 m/s against 1.09, a pelvis slope of 0.5 deg against 0.1.

Which path is right cannot be measured on the clip, so the question was put to the rendered sessions
(`tools/one_camera_speed.py`, `docs/one_camera.json`): each view alone, its pelvis speed read from the lifter's own
world path and from the fit handed that view twice, on the frames where the truth walks faster than 0.25 m/s and the
view's legs are usable. Walking along the camera's line of sight (the along camera, 459 to 891 frames a recording),
the lifter reads 0.51, 0.95, 0.55, 0.87, 0.83 and 0.80 of the true speed and the fit 1.00, 1.06, 0.94, 1.02, 1.04 and
0.99, with a speed RMSE of 0.22 to 0.63 m/s against 0.13 to 0.22. Walking across it (the side camera, 409 to 786
frames) the two agree: 1.01 to 1.12 and 1.07 to 1.12. This is the reason for `PATH_ALONG_SIGHT` seen from the other
side: a lifted step along the sight line is short as well as noisy. The clip's merge has only such steps, and its
speed is 0.85 of the fit's, so the fit's length is the likelier one. The rendered lenses are known and the clip's is
an estimate, which the fit's distance scales with.

The fit has two faults on the clip. The camera is hand-held and the fit holds it still, so in the seconds of pan (the
first 3 s and around second 13) the body slides sideways at 0.3 to 0.5 m/s. And the slower ankle moves at 0.40 m/s
while walking, 0.32 of the pelvis's speed, against 0.04 m/s in the merge: the path grew by 11 % while the legs'
swing along the sight line, which no panel sees, stayed as lifted. That measure is not a test of a path by itself. On
the six rendered recordings it reads 0.18 to 0.24 m/s for the merge (0.20 to 0.30 of the pelvis's speed) and 0.13 to
0.18 for the joint fit (0.15 to 0.22), where the fit's path is the better one on both path scores against the truth; the
merge's 0.04 on the clip is the lifter's foot lock applied to a path that is too short. A one-view mode of the fit,
with the camera's rotation per frame among the unknowns, is what these two findings point to. It is not built.

**What stays open.** A real two-camera recording with a truth. A held-out session for the settings. The camera pose:
the true-camera column is the case for measuring it once per room, or for carrying one fitted pose across a session's
recordings instead of fitting each alone. The scale, from the subject's height. The seated root. The walkway path
(`transl_mat`) is not refitted and is taken from the merge. A mirrored view is handled in the code and tested on
fabricated data only. The body IMUs are not used here either. The suite is at 94 tests (ten for the fit on a
synthetic two-camera walker, one for the centred moving average).

### 2.16 Round nine: a real two-camera recording with a chalk truth (2026-11-12)

*The lines were measured again after this round and are 13.5 ft apart, not 13. This section stays as it was
scored; 2.19 restates every number that rests on the tape.*

Round eight ended on the missing piece: a real two-camera recording with a truth. This round has one. OWN03 is our
own recording: one person walks back and forth between two chalk lines drawn 13 ft (3.962 m) apart on a paved lot,
filmed by a tablet that stands to the side and looks across the walk and by a phone on the ground at one end that
looks along it. Both cameras are still. A hand clap gives the offset between the two clips (5.270 s), and they are
tiled into one two-panel frame as a clinic recording would be (31 s, 930 frames at 30 fps). The clip is not bundled.
`tools/chalk_walk.py` reads the truth from the side view and scores every estimate against it; every number below is
in `docs/chalk_walk.json`. Three things could not be checked. The subject's height is not on record, and the scene's
scale turns on it (below). Neither lens is measured: the tablet was run at 26.2 and 31.0 mm and the phone at 30.4 and
27.1 mm (35 mm equivalent). And the truth is a position along the walk: there is none for the joint angles, the
heading or the path off the walk's axis.

**The pipeline on it.** Nothing was changed for the recording. The merge passes its checks and exports 100 % of the
930 frames. The joint fit ends with a median reprojection of 4.1 and 7.8 px in the two panels, sets the phone 5.79 m
from the tablet with the two optical axes 90.4 deg apart, and has both views on the walker in every frame.

**The truth.** The two chalk lines are parallel on the ground, so their images meet at a vanishing point on the
ground's horizon. A ground point's position across the lines is linear in the slope of the image line that joins it
to that point, X = h (x - x_vp) / (y - y_vp), and the known distance between the lines sets h. No lens, camera height
or subject height enters; the input is four clicked points, two on each line. The pelvis is placed over the ground at
the walker's feet, the bottom of the tracked box smoothed over a step. On level ground the horizon is level in the
picture. This pavement is not level: with a level horizon the walker's size (box height over the feet row's distance
below the horizon) changes along the walk with an SD of 0.110 m. Turning the horizon about the vanishing point until
that size is constant gives ground that rises 2.66 deg to the picture's right and an SD of 0.017 m, and that is the
truth used. It puts the tablet 0.77 m above the pavement and the lines 2.05 m to one side of it and 1.91 m to the
other. The walk has five turns between seconds 3.0 and 28.5, so four passes: 4.41, 4.20, 4.12 and 3.85 m (the walker
steps past each line before turning). The first is a lower bound, because at the first turn the pelvis reaches the
panel's side. Clicks that are 2 px off move the travel by 0.4 % and the stature below by 0.4 % (300 draws); at 3 px
both stay under 1 %.

**The scale check, and what it found.** The same geometry gives the height of the walker's box: 1.529 m in
mid-stride with shoes on, if the lines are 13 ft apart. Said without any distance, the lines are 2.59 box heights
apart. A fitted body's vertical extent in mid-stride is 0.990 of its standing height at the median (0.977 and 0.999
at the 10th and 90th percentiles of 200 frames, the same for the merge's body and the fit's), and shoes add about
0.03 m, so the picture says the subject stands about 1.51 m barefoot. The lifter's body stands 1.780 m and the joint fit's 1.736 m. One of two things is true:
the subject is about 1.5 m tall and both bodies are 15 to 18 % too tall, or the subject is nearer the bodies' height
and the lines are further apart than 13 ft (a 1.78 m subject puts them about 15 ft apart). The recording cannot say
which. Every estimate here takes its scale from the size of its body, so the scores are given two ways: in metres
with the lines at 13 ft, and in body heights (slope x 1.529 / the estimate's own standing height), which does not
depend on the distance or on the subject's height. The body-height column carries the gauge's own error, 2 to 3 %:
with the stride and shoe corrections applied it reads 0.946 for the merge and 1.033 for the joint fit.

**Results** (lenses 26.2 and 30.4 mm; 765 frames; slope = estimate on truth along the walk; passes in metres against
a truth of 4.41 or more, 4.20, 4.12 and 3.85):

| estimate | slope | spread about that line (m) | off the walk's axis (m) | body (m) | slope in body heights | passes (m) |
|---|---|---|---|---|---|---|
| merge | 1.112 | 0.041 | 0.118 | 1.780 | 0.955 | 4.91, 4.74, 4.67, 4.32 |
| joint fit | 1.185 | 0.035 | 0.042 | 1.736 | 1.043 | 5.21, 5.03, 4.89, 4.59 |
| side view alone, lifter's path | 1.098 | 0.063 | 0.119 | 1.780 | 0.942 | 4.74, 4.71, 4.64, 4.32 |
| side view alone, keypoint fit | 1.171 | 0.022 | 0.046 | 1.759 | 1.017 | 5.15, 4.98, 4.88, 4.55 |
| end view alone, lifter's path | 1.187 | 0.267 | 0.121 | 1.784 | 1.017 | 5.19, 4.65, 4.99, 4.34 |
| end view alone, keypoint fit | 1.479 | 0.105 | 0.043 | 1.760 | 1.285 | 6.35, 6.13, 6.09, 5.59 |

With the tablet at 31.0 mm the merge reads 1.103 and the joint fit 1.172 (0.952 and 1.041 in body heights). With the
phone at 27.1 mm they read 1.109 and 1.181 (0.952 and 1.041).

**What it shows.**

1. The two targets differ by a constant factor. The joint fit's path is 6.6 % longer than the merge's, in every pass
   (estimate over truth 1.11 to 1.13 for the merge, 1.18 to 1.20 for the fit) and under every lens tried (6.3 to
   6.6 %). In body heights the merge is about 5 % short and the fit about 4 % long. That is the rendered rooms'
   result over again (speed 0.974 and 1.038 of the truth, section 2.15), on a real walker, and the two errors are of
   one size: the gauge's 2 to 3 % does not separate them.
2. The fit's path is the straighter one. A walk between two lines has little motion off its axis, and the fit's path
   has 0.04 m of it against the merge's 0.12 m. There is no truth for that direction, so this is a plausibility
   reading and not a score.
3. The lens matters little here. Moving either lens across its bracket moves either slope by at most 1.1 %, while
   the fitted distance between the cameras moves from 5.47 to 6.29 m. The path's scale comes from the body and not
   from the camera pose.
4. The second camera is not what straightens or lengthens the path. The side view fitted by itself reads 1.171 with
   0.05 m off the axis, almost the joint fit's numbers. What changes the path is the keypoint fit, which holds one
   body size for the recording and moves the pelvis to keep the keypoints on it, where the lifter integrates a
   velocity. What the second camera adds is the limbs' depth, and this recording has no truth for that.
5. A view along the walk did not carry the path alone. The phone's lift reads the four passes as 5.19, 4.65, 4.99 and
   4.34 m (spread 0.27 m). Its keypoint fit is steadier (0.105 m) and reads 28 % long in body heights with the lens
   at 30.4 mm and 16 % long at 27.1 mm: depth along the line of sight comes from the body's size in the picture, and
   the lens scales it. In the rendered rooms, where the lens is known, the same fit read 0.94 to 1.06 of the true
   speed (section 2.15). An along view needs its lens measured before the distance from its fit is used.

**What it means for the choice.** On the one thing this recording can score, neither target is the more accurate: 5 %
short against 4 % long. The fit's path has the better shape. The merge stays the default and the joint fit the
optional second estimate, as before (2.18 changes that); the recording gives no reason to change either, and it
confirms on real video that the fit runs unattended on two still cameras that were placed by hand and never
calibrated. The larger finding is the scale. If the lines are 13 ft apart, both targets read the walk 11 and 18 %
long, because both bodies are too tall, and no choice between them repairs that. Scaling the body to the subject's
measured height would: it is one number per subject, it removes the common factor, and it leaves the 5 % and 4 %
above. It is not built.

**What stays open.** The subject's height for OWN03, which decides between the two readings of the scale check, and
how the 13 ft was measured. A scale taken from a measured height. A truth for the joint angles on real video, which
needs markers or body IMUs. A recording with the cameras where a clinic has them (one high to the side, one along
the walkway), with the lenses read. The suite is at 98 tests (four for the chalk truth on a synthetic camera over
level and rising ground). Section 2.17 has the subject's height and what it changes.

### 2.17 Round ten: the walker's height, the side view cut below the head, and a wider panel detector (2026-11-12)

*The lines were measured again after this round and are 13.5 ft apart, not 13. This section stays as it was
scored; 2.19 restates every number that rests on the tape.*

Round nine left OWN03's height open. It is now on record: the subject stands 1.651 m (65 in). They also report that
the chalk lines were "probably just over 13 feet" apart, from the middle of one line to the middle of the other, and
that the tablet stood on a box and looked down a little. This round reads the recording again with the height, then
repeats it with the side view cut below the head, as a clinic's side camera frames a walker, to see what the end view
then has to carry. Three things could not be checked. The height and the 13 ft disagree by 8.8 %, and the recording
cannot say which is off. The cut run has no truth of its own: it is scored against the truth read from the full side
view of the same frames. And there is still no truth for the neck, head or arms on real video, so what the cut does to
them is measured as a change and not as an error.

**The height against the tape.** The picture puts the lines 2.592 walker boxes apart (2.16). A walker of 1.651 m has a
box of 1.664 m in mid-stride with shoes on (0.990 of the standing height, plus 0.03 m). That puts the lines 4.313 m
apart (14.15 ft) and not 3.962 m, and the tablet 0.84 m over the pavement and not 0.77 m. Read the other way, lines 13
ft apart need a walker of 1.52 m. These were checked and do not close the gap: the tablet's tilt (the lines' vanishing
point is 65 px above the picture's centre, about 2.7 deg of downward tilt, which agrees with the subject's account and
moves the gauge by 1.2 % the wrong way); where on the shoe the box ends (1.1 %); the clicks (0.4 %); slack in the
tracked box; a lens that bends lines; the width of the chalk. The height is the firmer of the two numbers, so
`tools/chalk_walk.py` now prints the reading that trusts it beside the other (`--subject-height-m`), each as delivered
and with the body rescaled to the subject's height (slope = estimate on truth, 765 frames):

| estimate | 13 ft trusted, as delivered | 13 ft, body rescaled | height trusted, as delivered | height, body rescaled |
|---|---|---|---|---|
| merge | 1.112 | 1.031 | 1.022 | 0.947 |
| joint fit | 1.185 | 1.127 | 1.088 | 1.035 |
| side view alone, lifter's path | 1.098 | 1.018 | 1.008 | 0.935 |
| side view alone, keypoint fit | 1.171 | 1.098 | 1.075 | 1.009 |
| end view alone, lifter's path | 1.187 | 1.099 | 1.091 | 1.009 |
| end view alone, keypoint fit | 1.479 | 1.388 | 1.359 | 1.275 |

With the height trusted the bodies are 7.8 % (the lifter's, 1.780 m) and 5.1 % (the joint fit's, 1.736 m) too tall,
not 15 to 18 %. As delivered, the merge then reads the walk 2 % long and the joint fit 9 % long: the merge's short
path and tall body nearly cancel, and the fit's two errors add. Rescaled to the subject they read 5 % short and 4 %
long, which is the body-height column of 2.16 again, as it has to be. The choice made in 2.16 stands (until 2.18).
Its "larger finding" depends on which number is right: in metres the two targets are 2 and 9 % long if the height is,
11 and 18 % if the tape is. Measuring the lines again would settle it.

**The side view cut below the head.** Rows 300 to 880 of the side picture's 1080 were kept and tiled as a strip (1920
x 580) beside the full-height end view, which is untouched. In the full run the top of the walker's head sits at row
157 at the median (120 and 198 at the 10th and 90th percentiles) and the shoulder keypoints at row 278 (246 and 325).
So the head is out of the picture in every frame, and the cut passes through the shoulders or the upper chest: the
full run's shoulder keypoints are above it in 73 % of the frames, by 22 px at the median, a tenth of the length from
shoulders to hips. A fixed crop cannot do better, because the walker's shoulders move through about 80 rows along the
walk. The strip's lens was set to keep the picture's scale; the stage rounds a lens to whole millimetres, which leaves
the focal length 1.5 % apart (1344 against 1324 px). The strip's centre is 50 px below the camera's, 2.2 deg of tilt
that the lifter is not told about. Nothing in the pipeline was changed for the run, but the panel detector had to be
fixed before it would start (below).

**What the merge took from each view.**

| | full side view | side view cut |
|---|---|---|
| reference view | side | end |
| frames exported | 100 % | 100 % |
| root: share taken from the side / end view | 0.51 / 0.49 | 0.47 / 0.53 |
| legs: from both views / the end view alone | 99.6 / 0.4 % | 99.3 / 0.6 % |
| spine: both / end alone / side alone | 96.3 / 0 / 3.7 % | 79.6 / 17.1 / 3.3 % |
| neck and head: both / end alone / side alone | 96.6 / 0 / 3.4 % | 79.9 / 17.1 / 3.0 % |
| arms: both / end alone / side alone | 95.7 / 0.9 / 3.4 % | 89.8 / 7.0 / 3.2 % |
| the two views' paths apart, median (m) | 0.33 | 0.21 |

The end view gained less than the cut suggests. The reference passed to it (the merge takes the view with more usable
joints), and the trunk, neck and head came from it alone in 17 % of the frames. In the other 80 % the side view's neck
and head were still mixed in, although that view never shows the head (below).

**Scores against the chalk truth** (765 frames, lines at 13 ft, as delivered; with the height trusted every slope is
0.919 of the one shown):

| estimate | slope, full / cut | spread about that line (m) | off the walk's axis (m) | body (m) |
|---|---|---|---|---|
| merge | 1.112 / 1.192 | 0.041 / 0.140 | 0.118 / 0.130 | 1.780 / 1.785 |
| joint fit | 1.185 / 1.264 | 0.035 / 0.046 | 0.042 / 0.045 | 1.736 / 1.807 |
| side view alone, lifter's path | 1.098 / 1.186 | 0.063 / 0.128 | 0.119 / 0.200 | 1.780 / 1.814 |
| side view alone, keypoint fit | 1.171 / 1.263 | 0.022 / 0.040 | 0.046 / 0.078 | 1.759 / 1.831 |
| end view alone, lifter's path | 1.187 / 1.184 | 0.267 / 0.247 | 0.121 / 0.128 | 1.784 / 1.785 |
| end view alone, keypoint fit | 1.479 / 1.483 | 0.105 / 0.104 | 0.043 / 0.043 | 1.760 / 1.763 |

How far each pose moved between the two runs (deg, median with the 90th percentile in brackets; the side view alone is
the lift of that view before any merging):

| joints | merge | joint fit | side view alone |
|---|---|---|---|
| legs | 1.6 (3.6) | 1.5 (3.3) | 4.2 (7.5) |
| spine | 1.3 (2.6) | 1.4 (2.8) | 2.7 (6.1) |
| neck and head | 6.0 (11.3) | 6.0 (11.3) | neck 16.6 (22.9), head 10.8 (16.3) |
| arms | 3.8 (7.7) | 3.6 (6.9) | shoulders 7.7 (12.7), elbows 10.0 (31.3), wrists 8.2 (16.6) |
| root, after one yaw between the runs' worlds | 2.3 (3.8) | 2.1 (3.5) | |

**What the cut did.**

1. Both targets' paths grew by about 7 % (the merge's by 7.2 %, the joint fit's by 6.7 %), and the side view is the
   cause. Its lifter puts the cut walker 9.6 % further from the camera (median depth 3.80 to 4.16 m; 8 % once the 1.5
   % in the focal length is taken out) and gives the body another shape, and its path grows by 8.0 %. The end view's
   lift is the same within 0.3 %. The walker's box is 23 % shorter without the head, and the lifter reads depth from
   the box: that it allows for most of a cut body and not all of it is the likely reason, and it was not isolated. The
   side view's keypoint fit grows by the same 7.9 %, with a body 4 % taller. Rescaling to the subject's height removes
   little of this from the merge (0.947 to 1.013, still 7.0 %) and more than half from the joint fit (1.035 to 1.061,
   2.5 %). So on this recording a side camera that cuts the head reads the walk 7 to 8 % longer than one that shows
   the whole body.
2. The merge's path lost its shape and the fit's did not. The merge's spread about its line rose from 0.041 to 0.140
   m: its four passes read 5.38, 4.89, 4.89 and 4.46 m where the full view gave 4.91, 4.74, 4.67 and 4.32, so the
   first pass grew by 0.47 m and the others by 0.14 to 0.22 m. Its reference is now the end view, whose own lifted
   path has a spread of 0.25 m, and the cut side view's spread doubled. The joint fit's spread stayed under 0.05 m and
   its path stayed straight. In metres the merge is still the nearer of the two in every reading (1.192 against 1.264
   with the tape trusted, 1.095 against 1.161 with the height).
3. The legs held. The merge's leg joints moved by 1.6 deg at the median; its knee flexion differs from the full run's
   by 1.8 deg rms (correlation 0.993) and its hip flexion by 2.2 deg (0.991), and the joint fit's by 1.8 and 2.0 deg.
   The side view by itself moved its legs by 4.2 deg, so the second view takes up more than half of what the cut does
   to them.
4. The neck, head and arms moved, by 6.0 and 3.8 deg at the median in the merge and the same in the joint fit. In the
   side view alone the neck moved by 16.6 deg and an elbow by 10.0 (31.3 at the 90th percentile). There is no truth
   for these joints here.
5. The joint fit found the same pair of cameras. The two runs' self-calibrated poses agree within 1.3 deg in rotation
   and 1.3 deg in the direction between the cameras; the distance between them went from 5.79 to 6.26 m, 8 % more,
   like the side view's depth. The median reprojection is 4.5 px in the strip and 6.5 px in the end view (4.1 and 7.8
   px in the full run), and both views have the walker complete in 93 % of the frames (98 %).

**A neck taken from a view that does not show it.** The usable-joint rule judges the neck and head by the shoulder
keypoints (face keypoints are dropped by design), with a margin of 4 px at the top of the picture (2.13). In the cut
run the keypoint model still marks shoulders, 7 px below the top edge at the median (4 and 25 at the 10th and 90th
percentiles) and with a confidence of 0.62 where the full view has 0.97, although the true shoulders are above the
edge in 73 % of the frames. So the trunk passes the rule in 83 % of the side view's frames, and the neck and head that
go into the merge from that view are the lifter's guess. Section 2.13 found the same on rendered views. It was
measured again on the six rendered sessions (15,446 view frames), by how far the shoulder keypoints are below the top
edge in trunk lengths (r; median error against the truth, deg):

| r | view frames | neck | head | upper spine |
|---|---|---|---|---|
| under 0.05 | 1658 | 27.2 | 17.5 | 6.4 |
| 0.05 to 0.15 | 2322 | 21.2 | 10.0 | 5.0 |
| 0.15 to 0.25 | 841 | 17.1 | 8.5 | 5.4 |
| 0.25 to 0.35 | 1201 | 14.2 | 11.7 | 8.3 |
| 0.35 to 0.65 | 3902 | 11.1 to 11.5 | 8.6 to 10.4 | 5.2 to 5.5 |
| 0.65 to 1.0 | 814 | 10.7 to 14.6 | 10.9 to 11.3 | 6.8 to 7.8 |
| 1.0 to 1.5 | 1026 | 17.2 | 22.8 | 10.1 |
| 1.5 and over | 3661 | 11.6 | 7.8 | 7.5 |

A neck whose shoulders are within 0.15 trunk lengths of the edge is 21 to 27 deg off, twice what it is with the head
in the picture, and the spine does not care. (The row at 1.0 to 1.5 is out of line; half of its frames fail the usable
rule already, and it was not looked into.) OWN03's cut side view has a median r of 0.05. The plain remedy is to take
the neck and head from the view that shows the head when only one view does. It was tested on the same sessions and
not adopted. With the head counted as shown from r = 0.35 (1972 frames with it in one view only), the merge's neck is
11.6 deg off and its head 9.1; the view with the head alone gives 10.8 and 12.2, and the other view 14.8 and 10.3.
From r = 0.55 (4037 frames) the merge gives 10.6 and 8.2 and the view with the head 9.7 and 7.9. On the frames that
look most like OWN03's cut (one view at 0.35 or over, the other under 0.15; 306 frames) the merge gives 12.2 and 7.5
and the view with the head 9.3 and 9.7. Choosing the view gains 1 to 3 deg on the neck and loses up to 3 on the head,
so the merge is unchanged. The larger error is elsewhere: where neither view shows the head (1375 frames at 0.35) the
merge's neck is 22.7 deg off, there is nothing to choose from, and the export does not say so.

**Fixes made on the way.**

* The panel detector read the cut composite as a single view. A view that fills only a strip of the frame has black
  above and below it, so the black separator next to it failed the test that asks for picture on both sides. A band
  that is black over its whole length now counts as a separator when the picture beside it covers at least a quarter
  of the frame's other dimension. The cut composite reads as `[0, 721, 1923, 1299]` and `[1952, 0, 3032, 1920]` (tiled
  at `[0, 720, 1920, 1300]` and `[1952, 0, 3032, 1920]`), and the full composite and every earlier layout read as
  before.
* `wgvp demo <root>` without `--session` looked for its sessions one folder too high and found none. It now finds
  every session that holds a merged recording.
* `tools/chalk_walk.py` takes `--subject-height-m` (the second reading above) and `--truth-root` (the truth from
  another run of the same recording, for a view that cuts the body).

**What stays open.** Which of the height and the 13 ft is off; measuring the lines again answers it. The longer path
from a head-cut side view rests on one recording and one crop: whether it holds for a clinic's camera, which stands
higher and looks down, is not known, and no correction is built. The neck and head where no view shows the head, which
are a guess in the export and are not flagged. A truth for the upper body on real video. The suite is at 100 tests (a
known height against a wrong distance, a strip view beside a full-height view, and the demo finding its sessions).

### 2.18 Round eleven: the joint fit becomes the training target where it passes (2026-11-13)

*The lines were measured again after this round and are 13.5 ft apart, not 13. This section stays as it was
scored; 2.19 restates every number that rests on the tape.*

Rounds eight to ten scored the joint two-view fit beside the merge and left the merge as the default. The question
put after round ten was whether that evidence is enough to train on the fit. The answer taken here is yes, for a
recording with two still cameras whose fit passes its own checks, with the merge as the fallback everywhere else and
the merge's file in every delivery. Nothing new was recorded or lifted for this round: the decision rests on the numbers of
2.15 to 2.17 and on one new reading of the mat session. Not checked: the GPU end-to-end run on the box was not
repeated, so the dry run's new check has met only a copy of an earlier GPU run of the sample.

**The evidence in one place.** Each cell is merge / joint fit.

| six rendered recordings, median (2.15) | merge / joint fit | better on |
|---|---|---|
| path after one rigid fit (m) | 0.20 / 0.05 | 6 of 6 |
| speed RMSE while moving (m/s) | 0.169 / 0.067 | 6 of 6 |
| thigh direction (deg) | 6.0 / 3.8 | 6 of 6 |
| shank direction (deg) | 5.5 / 4.7 | 6 of 6 |
| ankle about the pelvis (cm) | 8.6 / 6.8 | 6 of 6 |
| upper arm direction (deg) | 9.0 / 6.0 | 6 of 6 |
| knee angle RMSE (deg) | 6.2 / 5.6 | 6 of 6 |
| root, three-axis (deg) | 9.4 / 9.8 | no gain |
| heading (deg) | 3.0 / 3.2 | no gain |
| root while seated, TUG (deg) | 21.5 / 22.6 | no gain |
| speed over the truth's | 0.97 / 1.04 | neither |
| limb jitter at rest, walking (cm per frame) | 0.06, 0.56 / 0.09, 0.60 | the merge, by a little |

| one real recording, OWN03 (2.16, 2.17) | full side view | side view cut below the head |
|---|---|---|
| distance over the truth, subject's height trusted | 1.022 / 1.088 | 1.095 / 1.161 |
| the same, 13 ft tape trusted | 1.112 / 1.185 | 1.192 / 1.264 |
| body rescaled to the subject, height trusted | 0.947 / 1.035 | 1.013 / 1.061 |
| body rescaled to the subject, tape trusted | 1.031 / 1.127 | 1.103 / 1.155 |
| spread about the fitted line (m) | 0.041 / 0.035 | 0.140 / 0.046 |
| off the walk's axis, rms (m) | 0.118 / 0.042 | 0.130 / 0.045 |
| the three complete passes (m; truth 4.20, 4.12, 3.85) | 4.74, 4.67, 4.32 / 5.03, 4.89, 4.59 | 4.89, 4.89, 4.46 / 5.38, 5.22, 4.90 |

**What speaks for the fit.** Where there is a truth for the body it is the better body: every limb row of the rendered
table, on every recording. Its path has the better shape on rendered and on real video. And it held under the framing
a clinic's side camera gives: with the head cut the merge's spread about its line tripled and its passes stopped
tracking the truth's (the second and third read the same 4.89 m), while the fit's passes kept their order and its
spread stayed at 0.05 m.

**What speaks against it.** Distance. As delivered the fit reads the walk longer than the merge in every reading of
OWN03, and longer than the truth in every one: on the full view 9 % with the subject's height trusted and 18 % with
the tape. Of the eight readings of mean distance (two truths, as delivered or rescaled, two framings) the merge is
the nearer in seven. On the rendered rooms the two sit on either side of the truth, 3 % short and 4 % long. The
fit's limbs also move a little less smoothly from frame to frame: 0.03 cm more at rest, 0.04 cm more walking.

**Why the fit all the same.** A sensor-to-pose model is trained on joint rotations and on the velocity in the heading
frame. The scale error is one factor per recording, it is in both targets, and it comes from the lifted body's
height: a known height removes about 5 points of it from the fit (1.09 to 1.04) and nothing else in the pipeline
can. The merge's near-correct distance on the full view is two errors that cancel, a short path and a tall body
(2.17), and the cancelling did not survive the head cut, where the merge reads 10 to 19 % long. The merge's errors
of shape are not one factor: a path whose passes do not track the truth, and limb directions 1 to 3 deg worse at the
median, cannot be corrected afterwards. So the fit is the target, the over-read is stated wherever the target is
described, and `body_height_m` is written beside the target for a consumer who has the subject's height.

**The rule** (`settle_target` in `wgvp/jointfit.py`, written as `target.json` in each recording's folder). In order:

1. The merge failed: `target` is null and the recording has no target.
2. `jointfit.enabled` is false in `wgvp_config.json`: the merge.
3. No fit is on file for this merge: the merge. The fit records a stamp of the merge it was made from (a hash of
   `frame_idx`, `body_pose`, `global_orient`, `transl` and `export_ok`), so a fit left from an earlier merge counts as
   not run. The usual reason is a machine without the SMPL-X model file.
4. The fit's status is SKIP (it does not apply: one view) or FAIL: the merge.
5. The fit's status is PASS: `jointfit_smpl.npz`, with the merge as `second_estimate`.
6. The fit's status is WARN: the merge, with the fit as `second_estimate` and its first warning as the reason. Notes
   that begin `info:` do not make a WARN.

The warnings that exist today are two cameras whose level axes are nearly parallel, a baseline so short that the
views are one viewpoint, and a mirrored view. `target.json` holds `target`, `method`, `why`, `second_estimate`, both
statuses, the merge's stamp and the fitted body's height.

**What changed in the pipeline.** The fit runs after every merge, inside `run`, `gvhmr` and `merge`, and
`wgvp jointfit` runs it alone; each of them settles `target.json` again. The SMPL-X model file is looked for in
`--body-models`, then `jointfit.body_models` in the config, then the GVHMR checkout. A missing file, or one that
does not load, costs the fit and never the merge. The session summary has two more items per recording,
`<recording>/jointfit` and `<recording>/target`, and the QC summary two more columns. A fit that WARNs or FAILs on a
two-view recording makes the session WARN: it is worth a look, and the merge is still there. `wgvp demo` shows the
target unless told otherwise (`--target merge` or `jointfit`). The dry run on a GPU checks that a target is named
and its file exists, and, where the target is the fit, that it holds the merge's frames and stands within 15 deg of
the rendered truth at the legs.

**What changed in the fit's file.** It now stands alone as a target. Besides the fit's own pose, path and body shape
it carries the heading, the heading-relative root and the heading-relative velocity, computed from the fit's pose
and path by the same three functions the merge uses (`heading_frame`, `heading_rel`, `rate_in_runs` in
`wgvp/fusion.py`; the merge was moved onto them and its output is unchanged to the byte). The export mask, the loss
weights and `transl_mat` are the merge's, copied. Fields that describe how the merge was made (`usable`, `source`,
`share`, the alignment, the other mat paths) stay in the merge's file.

**The velocity without the walkway.** The merge ties its velocity on the mat to the mat's measured progress (2.13).
The fit's has no such term, so the one rendered session with a mat (HC905) was read again. The path is scored
after one rigid fit, the speed against the truth's while the walker moves:

| HC905 | path (m) | path scale | speed ratio | speed RMSE (m/s) |
|---|---|---|---|---|
| merge, `transl` | 0.210 | 1.056 | 0.968 | 0.185 |
| merge, `transl_scaled` | 0.180 | 0.977 | 0.898 | 0.197 |
| merge, `transl_from_walkway` | 0.032 | 1.018 | 1.031 | 0.301 |
| joint fit, `transl` | 0.019 | 1.026 | 1.025 | 0.050 |
| merge, `root_vel_heading_rel_m_s` | | | 0.960 | 0.179 |
| joint fit, `root_vel_heading_rel_m_s` | | | 1.028 | 0.051 |

Without the mat the fit's path is nearer the truth than the merge's path on the mat, and its velocity is more than three
times nearer than the merge's with the mat's help. Over the six recordings the fit's velocity field reads 1.039 of the
truth's speed at 0.069 m/s RMSE, the merge's 0.973 at 0.168. The fit's path scale runs from 1.016 to 1.087 across
the six and the merge's from 1.029 to 1.129. One mat session, with fabricated footfalls: tying the fit's velocity to
a real mat is untested and not built.

**Verification.** `wgvp merge` was run again over five data roots that hold eleven recordings: the dry-run sample
from an earlier GPU run, the six rendered sessions, OWN03 in both framings, and the two bundled clips. No array of
any `merged_smpl.npz` changed. The fit's pose, root, path and shape came out identical to round ten's files, so the
fit is deterministic. Session statuses are as before. The targets:

| recording | target | reason given |
|---|---|---|
| the dry-run sample, the six rendered sessions | joint fit | checks passed |
| OWN03, full side view and head cut | joint fit | checks passed |
| OWN02 (one camera, tiled twice) | merge | the two cameras' level axes are nearly parallel |
| OWN01 (one view) | merge | the fit takes two views |

On the dry-run sample the fit stands 8.5 deg from the rendered truth at the legs over 1447 frames, the merge 8.5.
Fitted body heights: 1.74 to 1.79 m on the rendered sessions (the dry-run sample 1.82), 1.736 m on OWN03 and 1.807 m
with its head cut, against the subject's 1.651 m. The fit takes 3 to 7 s per recording. The CPU dry run passes (10
of 10), the bundled example verifies, and the suite is at 107 tests: the rule's every branch, a fit file read alone,
the fallback without a model file and with one that does not load, the summary's rows, and the demo's choice.

**What stays open.**

* Distance. The fit reads long and nothing rescales it. A rescale by a recorded subject height is the next step, and
  on OWN03 it would leave 4 % (13 % if the tape is right, which is still not settled).
* A moving camera. The fit solves one camera pose per recording and assumes both cameras stand still. Nothing
  detects one that moved. OWN02, hand-held with a pan, was kept out by the parallel-axes warning and not by any test
  of motion: none of the checks would stop a hand-held camera at right angles to a still one.
* The weights. The fit's file carries the merge's, which measure how far the two lifts agree and not how well the
  fit met the keypoints. A weight from the fit's own reprojection is not built.
* The root's rotation, the heading and the seated phase of the TUG, which the fit leaves where the merge has them.
* The evidence. The fit's settings were chosen on the six rendered sessions it is scored on. One real recording has
  a truth, for distance alone, of one person, outdoors. There is still no truth for the body on real video.
* The GPU end-to-end run with the fit in the pipeline, on the box.

### 2.19 Round twelve: the chalk lines measured again, the recording published, and demos in the README (2026-11-13)

Three things changed after round eleven. The subject measured the chalk lines of OWN03 again and found 13.5 ft, not
13. The recording itself is now in the repository as an example that runs offline. And the README shows the demo
videos of the six rendered sessions and of that recording, with their scores. One fault in the cross-view subject
match came out of the published clip and is fixed. Not checked: the full-size runs of rounds nine to eleven were not
lifted again (only their scoring was repeated, at the new distance), the published clip was lifted once, and the
neck, head and arms on real video still have no truth.

**The lines measured again.** Centre to centre the two lines are 13.5 ft apart (4.115 m). The distance enters only
the truth, so no lift was repeated: `tools/chalk_walk.py` was run again over the four lifts with `--distance-m
4.1148`, and `docs/chalk_walk.json` holds the new numbers. Sections 2.16 to 2.18 stay as they were scored, at 13 ft.
Each cell is merge / joint fit.

| OWN03, lenses 26.2 and 30.4 mm | lines at 13 ft (2.16 to 2.18) | lines at 13.5 ft |
|---|---|---|
| the lines by the tape (m) | 3.962 | 4.115 |
| the lines by the subject's height of 1.651 m (m) | 4.313 | 4.313 |
| the two apart by | 8.8 % | 4.8 % |
| the walker's stature read by the geometry (m) | 1.529 | 1.588 |
| camera height read by the geometry (m) | 0.77 | 0.80 |
| distance over the truth, in metres as delivered | 1.112 / 1.185 | 1.071 / 1.141 |
| the same with the body rescaled to 1.651 m | 1.031 / 1.127 | 0.993 / 1.085 |
| distance over the truth, in body heights | 0.955 / 1.043 | 0.955 / 1.043 |
| subject's height trusted, as delivered | 1.022 / 1.088 | 1.022 / 1.088 |
| subject's height trusted, body rescaled | 0.947 / 1.035 | 0.947 / 1.035 |
| complete passes, truth (m) | 4.20, 4.12, 3.85 | 4.36, 4.28, 4.00 |

| other lifts, in metres as delivered | lines at 13 ft | lines at 13.5 ft |
|---|---|---|
| side lens 31.0 mm | 1.103 / 1.172 | 1.062 / 1.129 |
| front lens 27.1 mm | 1.109 / 1.181 | 1.067 / 1.137 |
| side view cut below the head | 1.192 / 1.264 | 1.148 / 1.217 |
| side view cut below the head, body rescaled | 1.103 / 1.155 | 1.062 / 1.112 |

Every row that rests on the tape moved by the same 3.7 %, and nothing else moved. The rows in body heights and the
rows with the height trusted do not depend on the tape, and neither do the residuals about each fitted line
(0.041 / 0.035 m rms), the path's spread off the walk's axis (0.118 / 0.042 m) or the head-cut run's ratios to the
full view. With the lines at 13.5 ft the delivered walk reads 7 % long in the merge and 14 % long in the joint fit,
where round nine had 11 and 18 %.

**What the remeasurement settles and what it does not.** Half of the disagreement between the subject's height and
the tape is gone: 8.8 % became 4.8 %. The rest is still there, and the recording cannot say which of the two
carries it. A walker of 1.651 m puts the lines 14.2 ft apart, and lines 13.5 ft apart suit a walker of about 1.59 m,
read mid-stride and in shoes. The two readings that do not need the tape are unchanged: counted in body heights the
merge is 4.5 % short and the fit 4.3 % long, and the lifted bodies stand 1.78 and 1.74 m against the subject's
1.651 m. The choice of 2.18 does not rest on the tape. It rests on the six rendered sessions, on the body-height
reading and on the straighter path, and in metres the order of the two estimates is the same at either distance,
the merge the nearer one. The open item of 2.18 on distance now reads: a rescale by a recorded subject height would
leave the fit 3.5 % long if the height is right and 8.5 % long if the tape is.

**The recording as an example.** `examples/two_camera/` holds OWN03 as a data root: the two-view file at half size
(1516 x 960, 930 frames at 30 fps, 6.0 MB), a fabricated recording CSV with IMU t = 0 planted 0.8 s into the clip,
the known answers (`expected.json`, with the chalk marks, the distance and the subject's height) and a config that
sets the detector's image size to 960 and the two lenses. Three things were done to the footage before it was
published. The source files' metadata was stripped. Two signs in the background are covered in every frame. And a
passer-by who crosses the far road in the side view is painted out: the camera is still, so the far band is replaced
by its own clean plate (the median of the frames after they have left) everywhere but on the walker, whose
silhouette comes from the tracker's keypoints.

On the CPU the example runs in about 70 s. Both panels are found within 2 px of the true boxes, the sync reads
0.801 s against the planted 0.8, and the chosen track holds the walker in 100 % of 930 frames
(`examples/verify_examples.py --data examples/two_camera`). The side view takes its subject by the accelerometer
(`r_fine` 0.85, margin 0.73). The end view's tracker holds the walker as two fragments of 9.7 and 21.3 s, neither of
which matches the accelerometer alone, and its subject comes from the cross-view match (r 0.83). The session's status
is WARN from the sync-light cross-check (`flash_disagrees`, +1.532 s): the clip has no light, as on OWN02.

**The published clip, lifted.** The half-size clip was lifted on the T4 by `wgvp gvhmr`, which also ran the merge,
the joint fit and the target rule on the box: the first run of the GPU stage with the fit in the pipeline. The dry
run's own GPU pass, which 2.18 left open, was still not repeated. Both views lifted for all 930 frames, the merge
and the fit passed, 100 % of the frames are exported, and `target.json` names the fit. Against the chalk truth (765
frames, lines at 13.5 ft):

| | full size (above) | published half size |
|---|---|---|
| distance over the truth, in metres as delivered | 1.071 / 1.141 | 1.072 / 1.131 |
| distance over the truth, in body heights | 0.955 / 1.043 | 0.960 / 1.042 |
| residual about the fitted line (m rms) | 0.041 / 0.035 | 0.045 / 0.033 |
| path off the walk's axis (m rms) | 0.118 / 0.042 | 0.167 / 0.042 |
| body height (m) | 1.780 / 1.736 | 1.776 / 1.727 |
| the lines by the subject's height (m) | 4.313 | 4.305 |

Half the pixels and a painted far band leave the result where it was. The fit's numbers hold to 0.01; the merge's
path is less straight at half size (0.17 against 0.12 m off the axis) and the fit's is not.

**A fault in the cross-view match, found on the published clip.** At half size the end view's tracker changes the
walker's id once, and the first fragment is 291 frames long. The cross-view match tests each attached fragment in 5-s
windows on a 2.5-s grid and drops a frame only when every window that holds it fails. On that grid the last frames of
a fragment whose length is not a multiple of the hop sit in one window only. Here frames 150 to 224 sat in the window
75 to 224 alone, that window failed on the turn at the far end of the walk, and the view lost the walker for 2.5 s
that the tracker had. The fix adds one window flush with the fragment's end, so that its tail is judged twice like the
rest (`cross_view_selection` in `wgvp/selection.py`, with a test in `tests/test_cross_view.py`). The published clip
now keeps all 930 frames in the end view. The fix can only keep frames: the two full-size OWN03 runs, whose end view
is one unbroken track, select the same frames before and after, and the rendered sessions take both views by the
accelerometer and do not reach this code. The pelvis match's own window test (`consistency_mask`) walks the same grid
and was left as it is: its windows are scored against the accelerometer at a lag, nothing has shown the same loss
there, and a change would touch every recording's selection.

**Demos in the README.** `docs/demos/` holds seven demo videos, one per rendered session and one of the published
clip, each rendered by `wgvp demo` with the recording's target and encoded again at 1184 px wide (1.0 to 1.8 MB
each). GitHub does not play a video file kept in a repository, so each has a 14-s excerpt beside it as a looping
WebP picture (592 px wide, 10 frames a second, 0.6 to 0.8 MB), taken where the top half of the demo moves most. The
README shows the excerpt of each recording, linked to its video, with the fit's scores per session against the
rendered truth and the published clip's readings above. The folder is 15 MB. The demo of OWN03 is made from the
published clip, so the covered signs and the painted band are in it.

**Verification.** The suite is at 108 tests. `examples/verify_examples.py` passes on `examples/two_camera`. The six
rendered sessions take both views' subjects from the accelerometer and do not reach the changed code. The two
full-size runs of OWN03 do reach it and keep the same frames before and after.

**What stays open.** Which of the height and the tape carries the remaining 4.8 %; a third measurement would be of
the walker in shoes. A scale from a recorded height, which is not built. The pelvis match's tail window, if a
recording ever shows the loss there. A truth for the body on real video.

### 2.20 Round thirteen: the package as a first user meets it (2026-11-13)

This round changes no method. It follows the README from a fresh clone, makes the mistakes a first user makes, and
fixes the cheap ones.

**The path, timed.** On an Apple M-series laptop, in the pinned CPU environment: the clone takes 18 s (145 MB of
history, 210 MB on disk); `dry-run --cpu-only` passes 10 of 10 checks in about 70 s; `run examples/bundled
--cpu-only` takes 53 s and `run examples/two_camera --cpu-only` 71 s, and `examples/verify_examples.py` passes on
both. A second run of a folder reuses its tracks and takes about 10 s. The README said 3 to 6 minutes for the dry
run; it now says 1 to 6.

**Mistakes, and what now answers them.** Each of these gave a traceback or nothing at all, and now gives a
sentence. A data folder that does not exist: `data folder not found`, exit 2, for every command that takes one. A
folder with no subject folders in it: the scan says what layout it expected and where the matching rules are.
`gvhmr` with no GVHMR configured: the reason, exit 1 (it used to print only the summary line). `summary` before any
run: what to run first. `python -m wgvp` on a Python without tkinter: the install command per platform and the
three commands that do what the buttons do. After a `--cpu-only` run the last lines say that the poses come from
the GPU stage and give the command. Measured numbers in the run log and the dry-run checks are rounded to three
decimals. `tests/test_cli_messages.py` holds these.

**The GUI.** The pinned environment is a Homebrew Python, which is built without tkinter, so the window had never
opened on the development machine. `brew install python-tk@3.12` is the fix and is now in the README. For this
round the real `wgvp/gui.py` was driven under another Python that has Tk, with its pipeline calls relayed to the
pinned environment, and every widget's text, state and geometry was read back after each step. That shows what
the window says and when; it does not show the window as pixels, and it is not a test on Windows. What it found
and what changed:

| found | now |
|---|---|
| the panels preview of every session was written under the last session's folder (a loop variable captured late) | each session's own folder |
| after a run the list still read "1 recording(s), 1 video(s)"; the outcome was a Python dict on the status line | each session's row shows PASS, WARN or FAIL with up to two reasons; the status line reads `finished: PASS 1, WARN 1, FAIL 0 of 2 session(s)` |
| the bar kept the last run's full state through a dry run | reset at the start of a run; a moving bar during the dry run, which has no steps to count |
| **Open QC summary** could not reach a dry run's page, which is in a temporary folder | it opens the page of whatever just finished |
| a folder that does not exist gave "choose the top-level data folder first" | "this folder does not exist" with the path |
| an empty folder left an empty list and "idle" | the log says what layout was expected |

**The QC page.** `flags`, the reason for a WARN or FAIL, was column 31 of 32, two screens to the right of
`status` at 1440 px. On the page it now sits beside `status` (the CSV keeps its column order). A line above the
table says what the three statuses mean. The page sets its own white background, since it set a dark text colour
and was unreadable in a browser with a dark default canvas. A GPU stage that has not run is a sentence, not
`{"ran": false}`.

**The bundled example's table.** `examples/README.md` still carried the readings of the composite as it was before
its right panel was mirrored and zoomed (section 2.10): panel boxes, `r_fine`, `margin` and the size of the flash
disagreement. They are measured again; the session passes its check as before.

**Verification.** The suite is at 114 tests. The dry run passes, both shipped examples verify, and the driven GUI
finishes every step with no error in its log.

**After the fix.** With `python-tk@3.12` installed, `python -m wgvp` opens the window in the pinned environment
under Tk 9.1. The install also moved the Homebrew Python from 3.12.13 to 3.12.16; the environment followed it
without a rebuild, because its interpreter link points at Homebrew's stable path, and the suite still passes. The
same script then drove the window in process, with the real pipeline and no relay: panels detected 2.4 s after
**Scan**, the bundled example run in 50 s and run again in 9 s, the dry run in 71 s, the two-camera example in
68 s, every session row ending with its status and reason, and no error.

**What stays open.** The window has not been seen on a screen by us, on any platform, and has not been run on
Windows at all. It has no stop button, no single-session run (the command line has `--only`), no step inside a
session for the bar to show (a session of the two-camera example sits still for a minute), and **Check GPU** holds
the window while it works. The QC table is still 32 columns and 2900 px wide. The README is 1270 lines with the
quick start at line 108.

### 2.21 Round fourteen: a PC, a machine with no GPU, a bar that moves, and the window in the README (2026-11-13)

Four questions from a first look at the window: does it run on a PC, can the lift run without a GPU, why does the
progress bar not move, and why is the window not shown anywhere.

**A PC.** Until this round only the unit tests ran on Windows. The tests now run the dry run, the bundled example
and the window itself on Windows, macOS and Linux at every change:

| system (the test machine) | dry run | bundled example | the window, driven through a scan and a run |
|---|---|---|---|
| Windows Server 2025 | 10/10 in 239 s | verified; OWN01 PASS, OWN02 WARN | pass, 72 s |
| macOS 26, Apple silicon | 10/10 in 268 s | verified; OWN01 PASS, OWN02 WARN | pass, 85 s |
| Ubuntu 26.04 | 10/10 in 118 s | verified; OWN01 WARN, OWN02 WARN | pass, 55 s |

On Linux the one-view session carries `flash_disagrees (+0.150s)` too: the clip decodes to 363 frames with the
subject in them against 374 on the other two systems, and the search for a sync light finds a step it does not
find there. The sync itself is the same (0.816 s against 0.819 s).

One real fault for a PC turned up before any of that ran. Text files were opened in the system's default
encoding, which is UTF-8 on macOS and Linux and is not on most Windows installations. With the default set to
plain ASCII and a data folder named with accents, spaces and Japanese characters, the first log line that held
the folder name stopped the run. Every text file the package, the tools, the examples and the tests read or write
now names UTF-8 (99 calls), console output can no longer fail on a name it cannot show, and the processes the
package starts run in UTF-8 mode. The same folder then runs to PASS and verifies. `tests/test_text_encoding.py`
fails if a new call leaves the encoding out.

Not covered: the lift on Windows (GVHMR is Linux software in our hands; a PC runs the CPU stages and the window,
and the lift runs on a Linux machine or, untested by us, under WSL2), paths longer than Windows' 260 characters,
and a clinic's own PC.

**The bar.** It took one step per session and switched to a bouncing block for the dry run, so a one-session run
showed an empty bar for a minute and then a full one. Finding people frame by frame is 86 to 97 % of a session's
CPU time on the bundled examples, so that is what the bar now follows: the tracker reports frames as it reads
them, `wgvp/progress.py` turns the pipeline's events into one share-of-run number and a line of text ("session 3
of 3: OWN03, finding people in OWN03_SP.mp4: frame 392 of 930 (72%)"), a session's share of the bar follows its
video size, and with the lift configured the lifter's pieces fill their own share. Against the clock the bar stays
within 0.10 of the elapsed share on the bundled examples. Each session's row in the list now takes its result
when that session finishes, and the running one says so.

**The window, seen.** A process may draw its own window into a picture without any screen permission (AppKit's
`cacheDisplayInRect:toBitmapImageRep:`, reached through `ctypes`), which is how the window was finally looked at
and how the clip in the README was recorded: the real window, driven through a scan and a CPU run of the three
bundled sessions, 36 frames of 236. `launch(on_ready=...)` is the hook the recording and `tests/test_gui_smoke.py`
drive it through.

**No GPU.** GVHMR's demo moves its models and batches to the GPU by name (`.cuda()`, `torch.cuda.synchronize`),
so it stops at once on a machine without one. `wgvp/gvhmr_cpu.py` runs the same script with those moves made
into no-ops and every checkpoint loaded onto the CPU; `--device cpu` on the stage (`--lift-device cpu` on `run`
and `gvhmr`, or `gvhmr.device` in the config) routes through it with the GPU hidden. Measured on the T4 box with
its GPU hidden, one 393-frame chunk of OWN01:

| | GPU (T4) | CPU (8 Xeon cores) | difference of the poses |
|---|---|---|---|
| ViTPose + HMR2 features | 126 s | 442 s | keypoints identical to 2e-4 px |
| GVHMR network | 1.1 s | 0.6 s | joint angles p99 0.08 deg, worst frame 2.8 deg |
| whole chunk | 159 s | 464 s | world position p99 2.9 cm, in-camera position identical |

So about 35 s of lifting per second of video per view, and the poses are the same to well under the
frame-to-frame motion. The first attempt died at `torch.cuda.synchronize` in GVHMR's own timer, which the wrapper
now covers; a second attempt ran through. Not covered: a CPU build of the GVHMR environment set up from scratch
(the test used the box's CUDA build with the GPU hidden, which exercises the same code path), and Windows.

**What stays open.** The lift on Windows (WSL2 is the likely route and is untested); the four GUI items of 2.20
that are still there (no stop button, no single-session run, Check GPU blocks the window, no demo button); the
QC table's width. The bar is calibrated on the bundled examples; on a long clinic session with many recordings
per video the matching share may be larger than the 8 % allowed for, and the bar would then sit near full for
longer than it should.

## 3. What has not been tested on real clinic recordings

* The real seam / letterbox geometry was tested only on a video synthesised from one screenshot (static scene with
  injected per-panel motion) and on the rendered sample; real encoder noise, camera jitter or a third panel are
  untested. A view that fills only a strip of the frame beside a full-height view was read as a single view until
  round ten (2.17); that layout has met one tiled recording of our own and a synthetic case. The GUI preview and
  `views.override` are the fallback.
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
* A real camera's field of view: `tools/estimate_focal.py` has read one real camera, a hand-held phone's, where the
  only checks of its number are the lifted floor coming out level and two speeds agreeing (section 2.14). No clinic
  camera has been read, a camera square to the walkway is refused by design, and a lens that bends straight lines
  would bias it.
* The field of view at the lift (2.13): the gain is measured on rendered pinhole cameras whose field of view is
  known exactly. A real lens has distortion, a combined video's panels may be cropped or scaled copies of the
  cameras' pictures, and nobody has yet measured a clinic camera's field of view for this; a wrong number has not
  been tested against the default either.
* The rest hold (2.13) on real people: a rendered walker at rest does not sway, shift weight or get approached by a
  clinician. The keypoint check is what stands between a held box and somebody else's body, and it has only seen
  rendered bodies.
* The pelvis path fused from steps, the stride-swing rule and the velocity on the mat (2.13), all tuned and scored on
  the same six rendered recordings, two cameras near right angles.
* The scale of the lifted body on a real person. On the one real recording with a truth (2.16, 2.17) the subject
  stands 1.651 m and the bodies 1.74 and 1.78 m, 5 to 8 % taller, and the 13.5 ft measured between the chalk lines
  disagrees with that height by a further 4.8 % that is not resolved (2.19). With the side view cut below the head, as a
  clinic's side camera frames a walker, the same recording's paths came out 7 to 8 % longer. Every distance and speed
  in metres rests on the body's height and on how much of the body the side view shows.
* The neck and head where no view shows the head (2.17). The shoulder keypoints at the top edge pass the usable-joint
  rule, so the lifter's guess is exported: 19 to 23 deg off for the neck on the rendered sessions, with no truth on
  real video and no flag in the export. A clinic's side view that cuts the head leaves the neck and head to the along
  view.
* The joint two-view fit is the training target wherever its checks pass (2.18), and it has met no clinic recording.
  Its settings were chosen on the six rendered sessions. The one real recording with a truth is of one person
  outdoors and scores distance alone, which the fit reads 9 % long if the subject's height is trusted and 18 % if
  the tape is. It assumes that both cameras stand still for the whole recording, and nothing detects one that does
  not: set `jointfit.enabled` to false for such a session and the merge is the target.

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
   Each camera's model and lens or zoom setting, or its field of view, and whether a panel shows the camera's whole
   picture or a crop of it: with those the lift is given the real field of view (`gvhmr.focal_35mm`, 2.13).
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
10. **Still cameras, stature, and the body model file.** Do both cameras stand on tripods for a whole recording, or
    is either one moved or re-aimed while it records? Is each participant's height on record? The joint fit (2.18)
    assumes still cameras, and a recorded height is what would correct its scale. The fit also needs the SMPL-X
    model file on the machine that runs the merge; the GVHMR checkout has it.
