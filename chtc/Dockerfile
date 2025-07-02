###############################################################################
# Dockerfile — CUDA 11.7, Python 3.10, single conda env (“base”)
###############################################################################
# Always build an x86-64 image (CUDA only exists for that arch):
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# ---- OS packages ------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl bzip2 ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# ---- micromamba bootstrap ---------------------------------------------------
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

RUN curl -L \
      https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 \
      -o /usr/local/bin/micromamba && \
    chmod +x /usr/local/bin/micromamba

# ---- Create the *only* environment (“base”) ---------------------------------
RUN micromamba install -y -n base -c conda-forge \
        python=3.10 \
        numpy=1.26.4 \
        matplotlib \
        scikit-image \
        submitit \
        pybind11 \
        omegaconf \
        torchmetrics=0.10.3 \
        fvcore \
        iopath \
        pip && \
    micromamba clean -a -y

# ---- GPU / pip-only packages (same env) -------------------------------------
RUN micromamba run -n base pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu117 \
        --extra-index-url https://pypi.nvidia.com \
        torch==2.0.0 \
        torchvision==0.15.0 \
        xformers==0.0.18 \
        timm \
        wandb \
        opencv-python \
        cuml-cu11 \
        safetensors

# ---- Final touches ----------------------------------------------------------
WORKDIR /workspace
CMD ["/bin/bash"]
