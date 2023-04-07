# Remote Setup

Instructions for provisioning a VM used for remote development. This is particularly useful for testing and debugging GPU specific code or running in a non-resource-constrained environment.

```sh
# Define variables
PROJECT_ID="<PROJECT_ID>" # ex: internal-test-environment
ZONE="<ZONE>" # example: us-central1-a
VM_NAME="<VM_NAME>" # example: gpu-support
# Set the GCP project
gcloud config set project $PROJECT_ID
```

## Provisioning a VM

This step outlines the creation of a VM for development purposes. Note that this VM is configured with an Nvidia T4 GPU and an Nvidia base image.

```sh
gcloud compute instances create $VM_NAME \
    --project $PROJECT_ID \
    --zone $ZONE \
    --machine-type n1-highmem-2 \
    --boot-disk-size 128 \
    --boot-disk-type pd-standard \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --image nvidia-gpu-cloud-image-pytorch-20200629 \
    --image-project nvidia-ngc-public \
    --maintenance-policy TERMINATE \
    --restart-on-failure
```

### VM Management

It may be desirable to temporarily [suspend](https://cloud.google.com/compute/docs/instances/suspend-resume-instance) your VM while not actively in-use (ex: off-hours). Additionally, you may want to [delete](https://cloud.google.com/compute/docs/instances/deleting-instance) the VM when you no longer need it. The following describes a few useful commands for managing your VM.

```sh
# List existing VM's
gcloud compute instances list

# Suspend existing VM
gcloud compute instances suspend $VM_NAME

# Resume existing VM
gcloud compute instances resume $VM_NAME

# Delete existing VM
gcloud compute instances delete $VM_NAME
```

### Connecting to the VM

Once provisioned, you can SSH into the VM using the following command

```sh
gcloud compute ssh $VM_NAME
```

## Repository Setup

Once a new VM is provisioned you will need to configure access to the Text-to-Speech repository. This involves generating an ssh key in order to read and write to our private repository.

**_The following steps assume you have ssh'd into the VM_**

1. Generate an SSH key.

   ```sh
   # Generate ssh key
   ssh-keygen -t rsa -b 4096 -C "<EMAIL>"
   # Start ssh agent
   eval "$(ssh-agent -s)"
   # Add ssh key to running agent
   ssh-add ~/.ssh/id_rsa
   # Add github to known_hosts
   ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
   ```

1. Upload the SSH public key to your GitHub account. See [Adding a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) for detailed instructions.

   ```sh
   # Output contents of SSH public key. Copy this output into GitHub SSH key
   cat ~/.ssh/id_rsa.pub
   ```

1. Close the [Text-to-Speech](https://github.com/wellsaid-labs/Text-to-Speech) repository.

   ```sh
   git clone https://github.com/wellsaid-labs/Text-to-Speech.git
   cd ./Text-to-Speech
   ```

1. Install dependencies. Note that the base VM image does not contain python.

   ```sh
   # Install python
   sudo apt-get install python3.8 python3.8-dev python3.8-venv python3-venv
   # Setup venv
   python3.8 -m venv venv
   . venv/bin/activate
   python -m pip install pip --upgrade
   # Install dependencies
   python -m pip install -r requirements.txt --upgrade
   ```

At this point your development environment should be setup within the VM. You may find some of the [LOCAL_SETUP](./LOCAL_SETUP.md) and/or [BUILD](./BUILD.md) steps relevant once the VM & repository are setup.

## Running & Debugging the TTS Worker

**_The following steps assume you have ssh'd into the VM and activated the python virtual env_**

Start by ensuring the model checkpoints are loaded on the VM.

```sh
# Checkpoints are pulled from GCS and requires google auth credentials
gcloud auth login
# Load checkpoints
CHECKPOINTS="<CHECKPOINT_NAME>" # ex: v10_2022_06_15_staging
python -m run.deploy.package_tts $CHECKPOINTS
```

Run the TTS worker app. You may want to reference the default CMD in the worker [Dockerfile](../run/deploy/Dockerfile) or the container args in the [deployment configurations](../ops/run/svc.libsonnet).

```sh
GUNICORN=1 ./venv/bin/gunicorn run.deploy.worker:app \
  --bind=0.0.0.0:8000 \
  --workers=1 \
  --access-logfile=- \
  --config=./run/deploy/gunicorn.conf.py
```

Example stream request with active running worker:

```sh
curl -d '{"speaker_id":"7", "text":"hello"}' -H "Content-Type: application/json" -X POST http://localhost:8000/api/text_to_speech/stream --output sample.mp3
```

### Profiling the TTS worker

Leveraging Python's `cProfile` we can quickly profile the lifecycle of a request going through the TTS worker. This can be accomplished by running the app with a separate configuration file:

```sh
GUNICORN=1 ./venv/bin/gunicorn run.deploy.worker:app \
  --bind=0.0.0.0:8000 \
  --workers=1 \
  --access-logfile=- \
  --config=./run/deploy/gunicorn.profile.conf.py
```

### Building & Running the Docker Image

Follow the [Building the Docker image](./BUILD.md) steps for building the docker image within the VM. You may then run the built image in the following ways:

```sh
# Run the image
docker run --rm -p 8000:8000 --gpus all gcr.io/$PROJECT_ID/speech-api-worker:$IMAGE_TAG
# Run detached (useful for avoiding multiple terminals)
docker run -d -p 8000:8000 --gpus all gcr.io/$PROJECT_ID/speech-api-worker:$IMAGE_TAG
# Run with overridden CMD
docker run --rm -p 8000:8000 --gpus all gcr.io/$PROJECT_ID/speech-api-worker:$IMAGE_TAG /bin/sh -c "venv/bin/gunicorn run.deploy.worker:app --bind=0.0.0.0:8000 --timeout=3600 --graceful-timeout=600 --workers=2 --access-logfile=-"
```

### Running the Load Test Script

A common use case for performance testing is running the load testing script with an active running worker. See the load_test [README](../load_test/README.md) for more information.

**_The following assumes you have an actively running TTS worker in the background_**

1. Navigate to the load testing directory

   ```sh
   cd ~/Text-to-Speech/load_test/
   ```

1. Create an env file, for example `./load_test/local.env`

   ```txt
   ORIGIN=http://localhost:8000
   API_PATH_PREFIX=api/text_to_speech
   SKIP_VALIDATION_ENDPOINT=true
   FIXED_TEXT_LENGTH=80
   SCENARIO=single_constant_vu
   ```

1. Run the load test

   ```sh
   # network=host required to reach tts service running on host machine
   sudo docker run --rm -it --env-file local.env -v $(pwd):/app/ --network=host $(docker build -q .)
   ```

## GPU (Nvidia cli)

Debugging or inspecting GPU resources can be done via the `nvidia-smi` cli. This should be available in the VM's base image, other can be installed separately.

```sh
# Snapshot of GPU resources including CUDA & driver versions.
nvidia-smi
# List of available GPU devices
nvidia-smi --list-gpus
```
