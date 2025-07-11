Getting Started
===============

.. _installation:

Installation
------------

To use Jax-GCM, first install it using pip:

.. code-block:: console

   (.venv) $ pip install -e .

.. _build_docker:

Building the Docker Image
-------------------------

1. Make sure you're using an x86 device (not Apple M chip or Windows ARM).
2. Setup Google Cloud:

   a. Enable the Artifact Registry API:

   .. code-block:: console

      $ gcloud services enable artifactregistry.googleapis.com

   b. Create the Docker repository (only needed once):

   .. code-block:: console

      $ gcloud artifacts repositories create <repo-name> \
          --repository-format=docker \
          --location=<location> \
          --description="JAX GCM Docker images"

3. Log in to Google Cloud:

   .. code-block:: console

      $ gcloud auth login

4. Configure Docker to use gcloud authentication:

   .. code-block:: console

      $ gcloud auth configure-docker <location>-docker.pkg.dev

5. Ask Duncan to grant you the `roles/artifactregistry.writer` permission.

6. Build the Docker image:

   .. code-block:: console

      $ docker build -t <location>-docker.pkg.dev/<project-id>/<repo-name>/<image-name>:latest .

7. Push the Docker image:

   .. code-block:: console

      $ docker push <location>-docker.pkg.dev/<project-id>/<repo-name>/<image-name>:latest

.. _setup_tpu:

Setting up a TPU
----------------

1. CHECK AVAILABLE TPU RESOURCES! 

2. Create the TPU VM:

   .. code-block:: console

      $ gcloud compute tpus tpu-vm create <tpu-name> \
          --project=<project-id> \
          --zone=<region> \
          --version=tpu-ubuntu2204-base \
          --accelerator-type=<type> \
          --metadata-from-file startup-script=startup.sh \
          --preemptible

3. SSH into the TPU with port forwarding:

   .. code-block:: console

      $ gcloud compute tpus tpu-vm ssh <docker-name> \
          --project=<project-id> \
          --zone=europe-west4-a \
          -- -L 8888:localhost:8888

4. Open your browser and go to: http://localhost:8888





