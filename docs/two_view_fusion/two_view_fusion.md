# Two compromised views, one pose: HC901

Measured on the first render of `sample_v2` (side camera aimed at hip height, head cut on 77 % of its frames). The shipped `sample_v2` has since taken the room-wide side camera of the other samples; the README's reconstruction section closes with its numbers.

1395 frames with the truth and both views lifted. Errors are geodesic angles (deg) between GVHMR's joint rotation and the rendered truth; medians over frames.

| estimate | output coverage | legs | legs p90 | legs > 20 deg | arms | spine | all 21 | knee-flexion RMSE | frame step p99 | steps > 20 deg |
|---|---|---|---|---|---|---|---|---|---|---|
| left | 100% | 14.9 | 56.3 | 34% | 39.1 | 20.0 | 21.7 | 14.8 | 6.0 | 0.05% |
| right | 100% | 10.3 | 20.3 | 11% | 16.1 | 10.1 | 11.5 | 8.2 | 7.2 | 0.02% |
| mean | 100% | 12.7 | 25.7 | 27% | 22.7 | 14.3 | 16.2 | 8.6 | 5.0 | 0.02% |
| confidence-weighted | 100% | 12.7 | 25.0 | 26% | 19.4 | 12.7 | 14.3 | 8.8 | 7.1 | 0.12% |
| visibility-weighted | 100% | 9.6 | 19.6 | 9% | 16.5 | 11.1 | 11.9 | 10.3 | 6.7 | 0.19% |
| segment rule (truth visibility) | 91% | 10.1 | 19.4 | 9% | 16.5 | 8.4 | 11.1 | 10.2 | 5.6 | 0.13% |
| segment rule (run-time visibility) | 98% | 10.1 | 19.3 | 9% | 21.0 | 11.2 | 13.1 | 8.6 | 16.8 | 0.79% |
| segment rule, cross-faded (run-time visibility) | 98% | 10.1 | 19.4 | 9% | 20.4 | 10.7 | 12.7 | 9.9 | 5.5 | 0.03% |
| segment rule, cross-faded + smoothed 5 fr | 98% | 10.1 | 19.4 | 9% | 20.3 | 10.7 | 12.7 | 9.9 | 4.1 | 0.00% |
| segment rule, cross-faded + smoothed 9 fr | 98% | 10.3 | 19.7 | 9% | 20.3 | 10.7 | 12.7 | 10.1 | 3.0 | 0.00% |
| segment rule, cross-faded x root share + smoothed 5 fr | 98% | 10.0 | 19.3 | 8% | 15.7 | 10.3 | 11.4 | 7.4 | 4.8 | 0.00% |
| oracle pick | 100% | 9.4 | 17.5 | 5% | 15.2 | 9.2 | 10.7 | 5.1 | 13.0 | 0.45% |

Errors of the segment rules are over the joints they emit; coverage is the share of (frame, joint) cells with an output. Both views and the other fusions emit every cell. Continuity: the largest per-joint rotation step between consecutive emitted frames, its 99th percentile and the share of steps over 20 deg.

Root orientation after a constant heading alignment of +89 deg (left) and +5 deg (right): median error 49.6 / 7.3 deg (frame step p99 47.4 / 7.3); the fused root (weighted by each view's ramped presence and the square of its usable fraction, sharpened towards the heavier view when the two disagree) 6.6 deg (step p99 5.4); mean share of the left view 0.14.


Root error decomposed (signed medians about the truth's axes, yaw / pitch / roll) and the hip joints' error, by what the along camera saw:

| along camera saw | n | root left | root right | root fused | hips left | hips right | hips merge (no share) | hips merge (root share) |
|---|---|---|---|---|---|---|---|---|
| feet cut | 7 | -19 / -6 / -7 | -10 / -11 / +5 | -13 / -8 / +0 | 49.3 | 16.6 | nan | nan |
| head cut | 143 | -2 / -5 / +1 | +9 / -6 / +3 | +4 / -6 / +2 | 14.2 | 13.5 | 13.4 | 13.3 |
| hips cut (legs only) | 798 | +1 / +50 / -8 | -2 / -3 / +2 | -2 / -1 / +2 | 56.2 | 8.4 | 8.3 | 8.3 |
| shoulders cut | 377 | -0 / +17 / -3 | +6 / -5 / +3 | +5 / -3 / +2 | 22.4 | 15.3 | 14.8 | 14.5 |
| whole body | 70 | -2 / -7 / +1 | +12 / -6 / -0 | +1 / -7 / +1 | 15.4 | 21.5 | 15.5 | 15.5 |

## left view by what the camera saw (truth)

| frames | n | legs | arms | spine | all |
|---|---|---|---|---|---|
| feet cut | 7 | 43.2 | 32.1 | 18.9 | 31.9 |
| head cut | 143 | 14.3 | 18.3 | 14.0 | 14.8 |
| hips cut (legs only) | 798 | 14.4 | 47.7 | 22.4 | 33.9 |
| shoulders cut | 377 | 17.1 | 31.6 | 17.4 | 20.0 |
| whole body | 70 | 14.2 | 19.2 | 14.8 | 15.7 |

## right view by what the camera saw (truth)

| frames | n | legs | arms | spine | all |
|---|---|---|---|---|---|
| feet cut | 84 | 17.6 | 19.0 | 9.4 | 14.9 |
| head cut | 956 | 8.5 | 15.7 | 10.3 | 10.7 |
| out of view | 3 | 14.4 | 20.0 | 7.9 | 14.2 |
| shoulders cut | 24 | 11.9 | 15.0 | 14.6 | 13.2 |
| whole body | 328 | 14.3 | 15.9 | 9.5 | 12.6 |

## Fused estimates by what the left view saw (legs / all 21; segment rules: over emitted joints, with coverage)

| frames | n | mean | confidence-weighted | visibility-weighted | segment rule (truth visibility) | segment rule (run-time visibility) | segment rule, cross-faded (run-time visibility) | segment rule, cross-faded + smoothed 5 fr | segment rule, cross-faded + smoothed 9 fr | segment rule, cross-faded x root share + smoothed 5 fr | oracle pick |
|---|---|---|---|---|---|---|---|---|---|---|---|
| feet cut | 7 | 18.2 / 18.5 | 18.2 / 18.5 | 18.2 / 18.5 | nan / nan (0%) | nan / nan (0%) | nan / nan (0%) | nan / nan (0%) | nan / nan (0%) | nan / nan (0%) | 15.1 / 14.4 |
| head cut | 143 | 13.0 / 12.6 | 13.0 / 12.6 | 13.0 / 12.5 | 13.0 / 12.5 (99%) | 13.0 / 12.3 (100%) | 13.0 / 12.3 (100%) | 13.0 / 12.3 (100%) | 13.2 / 12.2 (100%) | 13.0 / 12.3 (100%) | 11.5 / 11.3 |
| hips cut (legs only) | 798 | 11.6 / 19.0 | 11.5 / 14.5 | 7.5 / 11.1 | 8.2 / 9.7 (88%) | 8.2 / 13.0 (98%) | 8.2 / 12.5 (98%) | 8.2 / 12.5 (98%) | 8.2 / 12.5 (98%) | 7.9 / 10.7 (98%) | 7.9 / 10.3 |
| shoulders cut | 377 | 14.4 / 15.4 | 14.4 / 14.5 | 14.4 / 13.1 | 14.4 / 12.8 (95%) | 14.4 / 13.2 (100%) | 14.3 / 12.6 (100%) | 14.2 / 12.6 (100%) | 14.3 / 12.6 (100%) | 14.1 / 11.7 (100%) | 12.6 / 10.7 |
| whole body | 70 | 13.9 / 13.3 | 13.7 / 14.4 | 13.7 / 15.0 | 13.7 / 15.1 (100%) | 13.8 / 14.9 (100%) | 13.8 / 15.3 (100%) | 13.7 / 15.3 (100%) | 13.6 / 15.3 (100%) | 13.8 / 15.5 (100%) | 12.5 / 12.3 |
