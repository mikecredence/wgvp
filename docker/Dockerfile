# wgvp GPU image (UNTESTED BUILD - see README "Container"). CUDA 12.1 + GVHMR pinned to the commit on our box.
# Build context: the scripts/wg_video_pipeline folder, with a weights/gvhmr_checkpoints/ folder you populate yourself
# (SMPL/SMPL-X body models are licensed and must be downloaded by the registered user; see env/install_gvhmr.md).
#   docker build -t wgvp-gpu -f docker/Dockerfile .
#   docker run --rm --gpus all -v /path/to/data:/data wgvp-gpu wgvp run /data --gvhmr-root /opt/GVHMR --gvhmr-python /opt/conda/envs/gvhmr/bin/python
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" FORCE_CUDA=1
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 wget ca-certificates build-essential python3-tk && rm -rf /var/lib/apt/lists/*
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-x86_64.sh -O /tmp/mc.sh && bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh
ENV PATH=/opt/conda/bin:$PATH
RUN conda create -n gvhmr python=3.10 -y
SHELL ["conda", "run", "-n", "gvhmr", "/bin/bash", "-c"]
RUN git clone https://github.com/zju3dv/GVHMR /opt/GVHMR && cd /opt/GVHMR && git checkout 088caff492aa38c2d82cea363b78a3c65a83118f
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 \
 && cd /opt/GVHMR && pip install -r requirements.txt && pip install -e . \
 && pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8" \
 && pip install ultralytics==8.3.80 smplx==0.1.28 timm==0.9.0 pytorch-lightning==2.4.0 hydra-core==1.3.2 einops==0.8.0 "imageio[pyav]==2.37.3" ffmpeg-python==0.2.0 scipy==1.15.3 opencv-python-headless==4.10.0.84 \
 && rm -rf $(python -c 'import site;print(site.getsitepackages()[0])')/ffmpeg && pip install --force-reinstall --no-deps ffmpeg-python==0.2.0
# render guard (skip the QC videos; our stage sets GVHMR_RENDER=0)
RUN cd /opt/GVHMR && python - <<'PY'
import re, pathlib
p = pathlib.Path("tools/demo/demo.py"); s = p.read_text()
s = s.replace("import cv2\n", "import cv2\nimport os\n", 1)
s = re.sub(r"\n    render_incam\(cfg\)\n    render_global\(cfg\)\n    if not Path\(paths.incam_global_horiz_video\).exists\(\):\n        Log.info\(\"\[Merge Videos\]\"\)\n        merge_videos_horizontal\(\[paths.incam_video, paths.global_video\], paths.incam_global_horiz_video\)",
           "\n    if os.environ.get('GVHMR_RENDER', '1') == '1':\n        render_incam(cfg)\n        render_global(cfg)\n        if not Path(paths.incam_global_horiz_video).exists():\n            Log.info('[Merge Videos]')\n            merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)", s)
p.write_text(s)
PY
# weights: copy your populated folder (gvhmr/, vitpose/, hmr2/, yolo/, body_models/) into the image
COPY weights/gvhmr_checkpoints/ /opt/GVHMR/inputs/checkpoints/
# the pipeline itself
COPY . /opt/wgvp
ENV PYTHONPATH=/opt/wgvp
RUN pip install imageio-ffmpeg==0.6.0 && printf '#!/bin/bash\nexec conda run --no-capture-output -n gvhmr python -m wgvp "$@"\n' > /usr/local/bin/wgvp && chmod +x /usr/local/bin/wgvp
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/wgvp"]
