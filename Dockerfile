FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y  \
    git \
    curl \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN pip install -e .

RUN pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Run the JCM Hydra CLI by default. Anything passed after the image name on
# `docker run` is forwarded as arguments to `python -m jcm.main`, so Hydra
# overrides (e.g. `physics=icon run.total_time=30`) work out of the box.
# Override the entrypoint (`--entrypoint bash`) to drop into a shell.
ENTRYPOINT ["python", "-m", "jcm.main"]
CMD []