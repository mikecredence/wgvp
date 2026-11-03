# Two compromised views, one pose: HC901

671 frames with the truth and both views lifted. Errors are geodesic angles (deg) between GVHMR's joint rotation and the rendered truth; medians over frames.

| estimate | legs | legs p90 | legs > 20 deg | arms | spine | all 21 | knee-flexion RMSE |
|---|---|---|---|---|---|---|---|
| left | 17.1 | 37.2 | 39% | 26.4 | 14.9 | 18.4 | 19.2 |
| right | 13.9 | 22.2 | 17% | 16.4 | 8.1 | 12.4 | 12.6 |
| mean | 14.1 | 23.6 | 21% | 19.9 | 11.3 | 14.4 | 11.3 |
| confidence-weighted | 13.9 | 23.1 | 19% | 18.6 | 9.8 | 13.6 | 11.1 |
| visibility-weighted | 13.9 | 22.5 | 18% | 17.2 | 9.4 | 12.9 | 11.6 |
| oracle pick | 12.2 | 19.6 | 9% | 14.9 | 7.2 | 11.3 | 5.7 |

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

## Fused estimates by what the left view saw

| frames | n | mean: legs / all | confidence-weighted | visibility-weighted | oracle pick |
|---|---|---|---|---|---|
| feet cut | 9 | 18.3 / 21.0 | 27.3 / 28.6 | 27.3 / 28.6 | 12.3 / 13.7 |
| head cut | 143 | 12.7 / 12.4 | 12.7 / 12.5 | 12.7 / 12.3 | 11.4 / 11.2 |
| hips cut (legs only) | 72 | 18.5 / 19.3 | 18.7 / 16.2 | 16.8 / 14.0 | 13.9 / 12.2 |
| shoulders cut | 377 | 14.1 / 15.2 | 14.0 / 13.7 | 14.0 / 12.6 | 12.3 / 11.0 |
| whole body | 70 | 14.1 / 13.1 | 13.2 / 13.5 | 13.2 / 14.2 | 12.2 / 12.2 |
