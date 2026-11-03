# Two compromised views, one pose: HC901

671 frames with the truth and both views lifted. Errors are geodesic angles (deg) between GVHMR's joint rotation and the rendered truth; medians over frames.

| estimate | output coverage | legs | legs p90 | legs > 20 deg | arms | spine | all 21 | knee-flexion RMSE | frame step p99 | steps > 20 deg |
|---|---|---|---|---|---|---|---|---|---|---|
| left | 100% | 17.1 | 37.2 | 39% | 26.4 | 14.9 | 18.4 | 19.2 | 8.0 | 0.05% |
| right | 100% | 13.9 | 22.2 | 17% | 16.4 | 8.1 | 12.4 | 12.6 | 9.5 | 0.10% |
| mean | 100% | 14.1 | 23.6 | 21% | 19.9 | 11.3 | 14.4 | 11.3 | 6.5 | 0.02% |
| confidence-weighted | 100% | 13.9 | 23.1 | 19% | 18.6 | 9.8 | 13.6 | 11.1 | 20.6 | 1.03% |
| visibility-weighted | 100% | 13.9 | 22.5 | 18% | 17.2 | 9.4 | 12.9 | 11.6 | 17.6 | 0.79% |
| segment rule (truth visibility) | 92% | 13.7 | 21.9 | 16% | 16.9 | 8.1 | 12.5 | 10.4 | 8.7 | 0.34% |
| segment rule (run-time visibility) | 96% | 13.7 | 21.9 | 16% | 17.4 | 8.3 | 12.5 | 10.2 | 16.6 | 0.73% |
| segment rule, cross-faded (run-time visibility) | 96% | 13.7 | 21.8 | 15% | 16.8 | 8.2 | 12.5 | 10.3 | 7.0 | 0.07% |
| oracle pick | 100% | 12.2 | 19.6 | 9% | 14.9 | 7.2 | 11.3 | 5.7 | 18.6 | 0.87% |

Errors of the segment rules are over the joints they emit; coverage is the share of (frame, joint) cells with an output. Both views and the other fusions emit every cell. Continuity: the largest per-joint rotation step between consecutive emitted frames, its 99th percentile and the share of steps over 20 deg.

Root orientation after a constant heading alignment of +91 deg (left) and -2 deg (right): median error 15.0 / 9.5 deg.


## left view by what the camera saw (truth)

| frames | n | legs | arms | spine | all |
|---|---|---|---|---|---|
| feet cut | 9 | 36.0 | 39.7 | 21.5 | 31.8 |
| head cut | 143 | 14.4 | 17.6 | 12.6 | 14.4 |
| hips cut (legs only) | 72 | 31.1 | 44.4 | 21.4 | 30.7 |
| shoulders cut | 377 | 18.6 | 31.9 | 15.8 | 20.5 |
| whole body | 70 | 13.3 | 18.3 | 11.9 | 14.9 |

## right view by what the camera saw (truth)

| frames | n | legs | arms | spine | all |
|---|---|---|---|---|---|
| feet cut | 81 | 16.7 | 20.1 | 9.5 | 15.2 |
| head cut | 242 | 13.6 | 15.2 | 6.9 | 11.7 |
| out of view | 15 | 12.5 | 16.8 | 7.3 | 13.0 |
| shoulders cut | 5 | 13.2 | 22.1 | 6.3 | 10.5 |
| whole body | 328 | 13.7 | 16.5 | 8.7 | 12.5 |

## Fused estimates by what the left view saw (legs / all 21; segment rules: over emitted joints, with coverage)

| frames | n | mean | confidence-weighted | visibility-weighted | segment rule (truth visibility) | segment rule (run-time visibility) | segment rule, cross-faded (run-time visibility) | oracle pick |
|---|---|---|---|---|---|---|---|---|
| feet cut | 9 | 18.3 / 21.0 | 27.3 / 28.6 | 27.3 / 28.6 | 13.6 / 9.4 (10%) | 23.4 / 11.3 (16%) | 23.4 / 11.3 (16%) | 12.3 / 13.7 |
| head cut | 143 | 12.7 / 12.4 | 12.7 / 12.5 | 12.7 / 12.3 | 12.7 / 12.3 (99%) | 12.7 / 12.3 (100%) | 12.7 / 12.3 (100%) | 11.4 / 11.2 |
| hips cut (legs only) | 72 | 18.5 / 19.3 | 18.7 / 16.2 | 16.8 / 14.0 | 15.6 / 11.7 (64%) | 15.9 / 13.0 (75%) | 15.9 / 12.6 (75%) | 13.9 / 12.2 |
| shoulders cut | 377 | 14.1 / 15.2 | 14.0 / 13.7 | 14.0 / 12.6 | 14.0 / 12.4 (95%) | 14.0 / 12.3 (100%) | 13.8 / 12.1 (100%) | 12.3 / 11.0 |
| whole body | 70 | 14.1 / 13.1 | 13.2 / 13.5 | 13.2 / 14.2 | 13.2 / 14.3 (100%) | 13.2 / 13.8 (100%) | 13.2 / 14.3 (100%) | 12.2 / 12.2 |
