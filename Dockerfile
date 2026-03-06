# ============================================================================
# MambaSegNet — Seismic Facies Segmentation
# Base: CUDA 12.4 + cuDNN — supported by drivers >= 550 (much more common).
# If nvidia-smi shows a lower max CUDA version, change 12.4.1 to match:
#   CUDA 12.1 → nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
#   CUDA 11.8 → nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# ============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# ── System packages ──────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        curl \
        git \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ── Install uv ───────────────────────────────────────────────────────────────
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Python dependencies via uv ───────────────────────────────────────────────
# Copy only the dependency spec first (better layer caching)
COPY pyproject.toml ./

RUN uv venv .venv --python python3.11 \
 && uv pip install --python .venv/bin/python \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
 && uv pip install --python .venv/bin/python \
        tifffile \
        Pillow \
        numpy \
        scipy \
        matplotlib \
        huggingface_hub \
        jupyterlab \
        ipywidgets \
        tqdm

# ── Optional: mamba-ssm fast CUDA scan ───────────────────────────────────────
# Uncomment the block below if you want the ~10-20x faster selective scan.
# Requires the full CUDA toolkit (already present in this image via devel tag).
#
# RUN uv pip install --python .venv/bin/python \
#         causal-conv1d \
#         mamba-ssm

# ── Activate venv for all subsequent commands ────────────────────────────────
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:$PATH"

# ── Copy project source files ─────────────────────────────────────────────────
COPY mamba_seg_net.py    ./
COPY pretrained_utils.py ./
COPY seismic_mamba_training.ipynb ./

# ── Jupyter config: disable token auth for local dev ─────────────────────────
RUN jupyter lab --generate-config \
 && echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py \
 && echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py \
 && echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py \
 && echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py \
 && echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py \
 && echo "c.ServerApp.disable_check_xsrf = True" >> /root/.jupyter/jupyter_lab_config.py

# ── Expose JupyterLab port ────────────────────────────────────────────────────
EXPOSE 8888

# ── Default command: JupyterLab ───────────────────────────────────────────────
CMD ["jupyter", "lab", "--notebook-dir=/workspace"]
