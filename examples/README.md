# Examples: real footage, fabricated sensor

The dry-run sample is rendered, so it cannot show how the tracker and the selector behave on real people, real
lighting and real clutter. Three things here do, without needing a real inertial sensor: a clip of our own that
ships in the repository (`bundled/`, runs offline), a two-camera recording of our own that ships beside it
(`two_camera/`, runs offline) and a corpus of public footage that is fetched to your machine (`fetch_examples.py`,
nothing redistributed).

## Bundled offline example (`examples/bundled/`)

```bash
python -m wgvp run examples/bundled --cpu-only          # about a minute on a laptop CPU
python examples/verify_examples.py --data examples/bundled
```

| session | what it is | what it exercises |
|---|---|---|
| OWN01 | a 13-s portrait phone clip: one of us walks away down an office corridor, turns at the far end and walks back (720 x 1280, 30 fps, 3.6 MB) | a real person, real lighting, a four-fold change of apparent size, and the single-view path |
| OWN02 | a synthetic two-view file made from OWN01: the clip cropped to its left 85 % as the left panel, the clip mirrored and cropped as the right panel, letterboxed differently, black separator (1230 x 1432, 5.2 MB) | panel detection on the tiled layout, the joint two-view selection, the two-view reconciliation. It is **not** a second camera; it is the same frames twice, so the reconciliation is bound to agree |

The recording CSVs are fabricated exactly as for the fetched corpus (below): the walker's own pelvis signal plus
noise at a displacement SNR of 2, differentiated twice, IMU t = 0 planted 0.8 s into the clip; OWN02 reuses OWN01's
recording since the frames are the same. The person in the clip consented to its publication; it is released under
the repository licence.

Measured on 2026-11-02:

| session | planted offset | found | subject frames on target | `r_fine` | `margin` | panels | status |
|---|---|---|---|---|---|---|---|
| OWN01 | 0.8 s | 0.819 s | 100 % of 374 | 0.85 | 0.64 | single (correct) | PASS |
| OWN02 | 0.8 s | 0.821 s | 100 % of 374 | 0.86 / 0.82 (left / right) | 0.45 / 0.45 | left [0,0,613,1280] vs true [0,0,612,1280]; right [617,75,1230,1357] vs [618,76,1230,1356] | WARN: `flash_disagrees (+0.112 s)`; reconcile PASS, residual lag 4 ms |

The data root's `wgvp_config.json` sets `params.detector_imgsz` to 960. At the default 640 the walker at the far
end of the corridor (under a tenth of the frame height) drops out of detection for about three seconds and the
track splits into two 5-s fragments, which the selector declines by design (see *What was tried and left out*).
This is the one setting a lab whose along-walkway camera sees the participant far away may also need. The WARN on
OWN02 is the optional sync-light cross-check: the clip has no light, the automatic detector proposes a brightness
change in the footage, and it misses the cross-correlation by 0.112 s, just over the 3-frame tolerance (on OWN01
a different candidate happened to fall inside it and reads "validated by flash"; neither means anything here).

`make_bundled.py` rebuilds both sessions from the original recording.

## Two-camera example (`examples/two_camera/`)

```bash
python -m wgvp run examples/two_camera --cpu-only       # about a minute on a laptop CPU
python examples/verify_examples.py --data examples/two_camera
```

| session | what it is | what it exercises |
|---|---|---|
| OWN03 | 31 s of one of us walking back and forth between two chalk lines on pavement, filmed at the same time by a tablet on a box to the side (left panel) and a phone on the ground at one end (right panel), both still; the two views tiled into one file at half size (1516 x 960, 30 fps, 6.0 MB) | two real cameras with different lenses and unequal panels, a walker who turns five times, and the full two-view path through the lift, the merge and the joint fit |

The recording CSV is fabricated as for the other examples, from the walker's pelvis signal in the side view, with
IMU t = 0 planted 0.8 s into the clip. The two source videos were lined up by the clap at their start. The person
in the clip consented to its publication and it is released under the repository licence. Two signs in the
background are covered and a passer-by in the side view is painted out. No builder script ships for this session
because the two original recordings are not published.

Measured on 2026-11-13:

| session | planted offset | found | subject frames on target | `r_fine` | `margin` | panels | status |
|---|---|---|---|---|---|---|---|
| OWN03 | 0.8 s | 0.801 s | 100 % of 930 | 0.85 (left); right taken by cross-view match, r 0.83 | 0.73 | left [0,209,962,751] vs true [0,210,960,750]; right [976,0,1516,960] vs [976,0,1516,960] | WARN: `flash_disagrees (+1.532 s)`; reconcile PASS |

The WARN is the same one as on OWN02: the clip has no sync light, and the cross-check reports a brightness change
in the footage that does not agree with the cross-correlation. In the right view, the phone looking along the walk,
the tracker holds the walker as two fragments of 9.7 and 21.3 s, and neither matches the accelerometer by itself
(the recording is fabricated from the left view's signal). Its subject is therefore taken by matching the left
view's subject across the views, and the QC names the method per view (`cross_view`).

The data root's `wgvp_config.json` sets `params.detector_imgsz` to 960, as the bundled example does, and
`gvhmr.focal_35mm` to the two cameras' lenses (26.2 mm for the tablet, 30.4 mm for the phone, as 35 mm
equivalents), which the lift uses in place of its default. With the GPU step run, both views lift for all 930
frames, the merge and the joint fit export 100 % of them, and `target.json` names the joint fit.

**The chalk lines.** `expected.json` also records what is needed to score the distance walked: the lines are 13.5 ft
apart centre to centre (4.115 m), the walker stands 1.651 m, and `chalk_lines.marks_px_in_view` holds the two ends
of each line in the left panel's pixels. `tools/chalk_walk.py` takes them as `--marks` and scores a lifted run;
`docs/chalk_walk.json` has the result under "OWN03, published half-size clip" (the merge reads 1.07 of the true
distance in metres and the joint fit 1.13; counted in body heights, 0.96 and 1.04).

## Fetched corpus (`examples/data/`, built by `fetch_examples.py`)

This corpus runs the pipeline on public footage without redistributing anyone's video:

* three public YouTube clips are **downloaded onto your machine** by `fetch_examples.py` (nothing is stored in this
  repository except the video ids, the timecodes and the pixel position of the intended walker);
* each is cut to the listed segment and re-encoded to a constant 30 fps, at most 1280 px wide, in the WearGait
  folder layout (`EX01/SP/EX01_SP.mp4` beside `EX01/EX01_selfpace.csv`);
* the recording CSV is **fabricated**: the pipeline's own 2D tracker runs once on the cut, the walker under the
  manifest's hint becomes the subject, and that walker's pelvis vertical signal (hip row relative to the horizon, in
  torso units) gets band-limited noise added at a displacement SNR of 2, is differentiated twice and written as
  `LowerBack_FreeAcc_U` at 100 Hz, with IMU t = 0 planted `offset_s` seconds into the cut and `GeneralEvent = Walk`
  while the walker is visible;
* the planted offset and the walker's hip track go to `EXnn/expected.json`, and `verify_examples.py` checks a
  pipeline run against them.

Because the reference is derived from the same tracker the pipeline uses, these examples (bundled and fetched) show that the machinery
runs end to end on real footage and picks the intended person out of everyone else in the frame. They are **not**
an independent accuracy measurement; the unit tests, the rendered sample and the tests in
[docs/design.md](../docs/design.md) are that.

### Run it

```bash
pip install yt-dlp                                   # only for the download step
python examples/fetch_examples.py                    # ~2 min of CPU tracking after the downloads
python -m wgvp run examples/data --cpu-only          # ~3 min on a laptop CPU
python examples/verify_examples.py
```

If YouTube answers yt-dlp with "Sign in to confirm you're not a bot", re-run the fetch with
`--cookies-from-browser chrome` (or firefox / safari / edge), or download the three videos with any tool and pass
`--source-dir <folder>` with the files named `<video_id>.mp4`. The pipeline writes to `examples/data/wgvp_output/`;
open `QC_SUMMARY.html` there after the run. `examples/data/` and the download cache are git-ignored.

### The three clips

| example | source | cut | what it exercises |
|---|---|---|---|
| EX01 | `n-O8dHyYIF0`, "6 Minute Walk test" instructional video | 30-47 s | the easy case: one person stands for two seconds, then walks away down an empty corridor for 13 s, the geometry of an along-walkway view (the body shrinks four-fold as it recedes) |
| EX02 | `FTHc-TJOQ34`, "Shuttle walk test" demonstration | 200-250 s | the clinic-like case: one walker goes down a corridor and back while two staff stand in frame, one of them close to the camera and a third of the frame tall; the walker's track breaks at the far turn and must be re-joined at one lag |
| EX03 | `KnvYRnTA3XQ`, public plaza footage (60 fps source) | 70-100 s | the hard case: one passer-by crossing the frame for 14 s among a few hundred track fragments, with a reference that has no stand / walk structure |

All three are listed as healthy-walking sources in the GAVD dataset. They are used here only as footage of people
walking; nothing about the people in them is inferred or stored beyond the skeleton keypoints of the chosen walker.

### What to expect

Measured on 2026-11-02 (Apple M-series CPU; `python examples/verify_examples.py` after the run):

| example | planted offset | found | subject frames on target | `r_fine` | `margin` | fragments in frame | pipeline status |
|---|---|---|---|---|---|---|---|
| EX01 corridor | 1.0 s | 1.002 s | 100 % of 397 | 0.80 | 0.58 | 4 | WARN: `flash_disagrees` |
| EX02 shuttle walk | 2.0 s | 2.012 s | 100 % of 1260 | 0.88 | 0.66 | 48 | PASS |
| EX03 plaza crowd | 1.5 s | 1.512 s | 100 % of 413 | 0.88 | 0.50 | 322 | WARN: `flash_disagrees` |

All three recover the planted offset within a frame and put the chosen track on the intended walker in every shared
frame; the crowd clip's walker wins with a margin of 0.50 over the best of 321 other fragments. The two WARNs come
from the optional sync-light cross-check: the automatic detector found a persistent brightness step in the footage
(a camera exposure change, a person passing a light) and reports that it does not agree with the pelvis
cross-correlation. These clips have no sync light, so the disagreement is the cross-check being honest about a
false candidate, not a sync problem; in a recording with a real light the two agree within a few milliseconds
(the bundled sample: 9 ms).

Reading the QC rows (`examples/data/wgvp_output/qc_summary.csv`): the `imu_xcorr` selection with its `r_fine` and
`margin`, the offset against the planted one, `present_frac` (how much of the fabricated recording the walker was
in frame; EX02's walker spends part of the time at the far end of the corridor) and the flags. The crowd clip's
skeleton thumbnails hold dozens of people; they still pass the privacy check because the check accepts
palette-limited renders, not only sparse ones.

### What was tried and left out

A fourth clip, `CX0_rJZJ5_U` (one person walking toward a gym camera and back, 13 s), was dropped: the walker fills
the frame when closest, the tracker splits the track there into two fragments of 6.6 and 5.8 s, and the selector
declines both as designed. Short fragments reach high correlations by chance (a 6-s fragment reaches about 0.7 over
a full-range lag search), so the acceptance rule asks for an excess over that level that 6 s of a self-derived,
noisy reference rarely gives. The pipeline's working regime is 8 s or more of continuous visibility per view; a
clinic recording gives 20-45 s. In such a case the QC says FAIL with `no subject could be chosen` and names the two
operator tools (`pick-subject`, `set-roi`).

### Adding a clip

Add an entry to `sources.json`: the video id and URL, the cut, and one or more `subject_hints` as
`[frame_in_cut, x, y]` of the walker's hips (pick them from a frame of the cut; any point inside the person's box
works). Keep `offset_s` smaller than the time at which the walker first appears if you want the sync to be a real
test, and run `fetch_examples.py --only EXnn`.
