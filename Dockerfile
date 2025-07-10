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

CMD ["bash"]