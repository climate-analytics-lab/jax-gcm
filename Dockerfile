# GPU build of the JCM image. Uses an NVIDIA CUDA runtime base and JAX's
# bundled CUDA 12 wheels. JAX falls back to CPU if no GPU is visible at
# runtime, so the same image works on hosts without `--gpus all`, just
# without acceleration.
#
# Build:
#     docker build -t jcm .
#
# Run on a GPU host:
#     docker run --rm --gpus all jcm physics=icon run.total_time=2
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Ubuntu 22.04 ships Python 3.10; pull 3.11 from deadsnakes to satisfy
# JCM's >=3.11 requirement. DEBIAN_FRONTEND=noninteractive prevents tzdata
# (a transitive dep of build-essential) from prompting at build time.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        build-essential \
        libopenblas-dev \
        liblapack-dev \
        libffi-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN pip install --no-cache-dir -e .

# Replace the CPU JAX pulled in by requirements.txt with the CUDA 12
# build. The cuda12 extra bundles the NVIDIA wheels JAX needs at runtime.
RUN pip install --no-cache-dir --upgrade "jax[cuda12]"

# Run the JCM Hydra CLI by default. Anything passed after the image name on
# `docker run` is forwarded as arguments to `python -m jcm.main`, so Hydra
# overrides (e.g. `physics=icon run.total_time=30`) work out of the box.
# Override the entrypoint (`--entrypoint bash`) to drop into a shell.
ENTRYPOINT ["python", "-m", "jcm.main"]
CMD []
