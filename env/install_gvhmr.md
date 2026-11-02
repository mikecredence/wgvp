# GPU stage: GVHMR environment (the one hard dependency)

The GPU stage runs GVHMR (Shen et al., SIGGRAPH Asia 2024, https://github.com/zju3dv/GVHMR) on the subject's cropped
video chunks. Everything else in this package runs on CPU in a plain Python environment. Set up GVHMR once, then
tell the package where it is (GUI fields "GVHMR folder" and "Python of the GVHMR environment", or `gvhmr.root` /
`gvhmr.python` in `wgvp_config.json`) and press "Check GPU": the check lists every missing piece with a fix.

Tested configuration (our Tesla T4 box, Ubuntu 24.04, driver 580, CUDA 12.1 wheels):

```
conda create -n gvhmr python=3.10 -y && conda activate gvhmr
git clone https://github.com/zju3dv/GVHMR && cd GVHMR && git checkout 088caff492aa38c2d82cea363b78a3c65a83118f
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt && pip install -e .
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"      # needs nvcc (CUDA toolkit 12.x)
pip install ultralytics==8.3.80 smplx==0.1.28 timm==0.9.0 pytorch-lightning==2.4.0 hydra-core==1.3.2 einops imageio[pyav] ffmpeg-python==0.2.0
```

Weights go under `GVHMR/inputs/checkpoints/` exactly as GVHMR's `docs/INSTALL.md` describes:

| file | source |
|---|---|
| gvhmr/gvhmr_siga24_release.ckpt | GVHMR release (Google Drive / HF, see GVHMR docs) |
| vitpose/vitpose-h-multi-coco.pth | ViTPose release |
| hmr2/epoch=10-step=25000.ckpt | 4D-Humans (HMR2.0) release |
| yolo/yolov8x.pt | ultralytics release |
| body_models/smplx/SMPLX_NEUTRAL.npz, body_models/smpl/SMPL_NEUTRAL.pkl | smpl-x.is.tue.mpg.de / smpl.is.tue.mpg.de (registration; licence forbids redistribution, so they are NOT in this package or the Docker image) |

Known pitfalls (each cost us a run):

* The `ffmpeg` PyPI stub shadows `ffmpeg-python` and GVHMR dies with `module 'ffmpeg' has no attribute 'input'`.
  Fix inside the env: `rm -rf $(python -c 'import site;print(site.getsitepackages()[0])')/ffmpeg && pip install --force-reinstall --no-deps ffmpeg-python==0.2.0`.
  `wgvp check-gpu` tests for this.
* GVHMR's demo renders two QC videos after inference (a third of the wall time). Our stage sets `GVHMR_RENDER=0`; the
  guard is a two-line patch in `tools/demo/demo.py` (`if os.environ.get("GVHMR_RENDER", "1") == "1": render_incam(cfg); render_global(cfg); ...`).
  Without the patch the stage still works (slower) and deletes the rendered videos afterwards.
* ViTPose/feature extraction load the whole chunk into RAM at half resolution: with `max_chunk_frames` 3000 at 1280 px that is < 6 GB. Lower it on a small machine.
* Throughput on a T4: about 0.5 s per frame of inference, i.e. a 50-s view at 30 fps takes 12-15 min; two views per recording.

A container recipe is in `docker/` (untested build - our GPU box had no disk left to build it; the pinned versions are the ones running there).
