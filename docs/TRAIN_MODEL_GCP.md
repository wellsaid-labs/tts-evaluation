# Train a Model with Google Cloud Platform (GCP) Compute

This markdown will walk you through the steps required to train a model on a GCP virtual
machine.

Related Documentation:

- During any point in this process, you may want to image the disk so that you don't have to start
  from scratch every time. In order to do so, please follow the instructions
  [here](https://cloud.google.com/compute/docs/images/create-delete-deprecate-private-images).

- Would you like to train a end-to-end TTS model? Please follow
  [this documentation](TRAIN_TTS_MODEL_GCP.md) instead.

- You may want to learn more about the available VM configurations, you can learn more
  [here](https://console.cloud.google.com/compute/instancesAdd?project=voice-research-255602&organizationId=530338208816)
  and [here](https://cloud.google.com/sdk/gcloud/reference/compute/instances/create).

## Prerequisites

1. Setup your local development environment by following [these instructions](LOCAL_SETUP.md).

2. Install `gcloud compute` by following the instructions
   [here](https://cloud.google.com/compute/docs/gcloud-compute/)

3. Ask a team member to grant you access to our GCP project called "voice-research".

4. Install these dependencies:

   ```bash
   brew install rsync
   brew install lsyncd
   ```

## From your local repository

1. Setup your environment variables

   ... for training a spectrogram model

   ```bash
   VM_MACHINE_TYPE=n1-highmem-8
   VM_ACCELERATOR_TYPE=nvidia-tesla-p100,count=2
   ```

   ... for training a signal model

   ```bash
   VM_MACHINE_TYPE=n1-highmem-32
   VM_ACCELERATOR_TYPE=nvidia-tesla-v100,count=8
   ```

   Also set these ...

   ```bash
   VM_INSTANCE_NAME=your-vm-instance-name

   # Pick a zone that supports your choosen `VM_ACCELERATOR_TYPE` using this chart:
   # https://cloud.google.com/compute/docs/gpus/
   # Note you'll want to spread your experiments out accross multiple zones to mitigate the risk of
   # your experiments getting throttled.
   VM_INSTANCE_ZONE=your-vm-instance-zone
   ```

2. Create your virtual machine, like so:

   ```bash
   gcloud compute --project=voice-research-255602 instances create $VM_INSTANCE_NAME \
     # A zone with the required resources using this chart
     # https://cloud.google.com/compute/docs/gpus/
     --zone=$VM_INSTANCE_ZONE \

     # Required resources
     --min-cpu-platform="Intel Broadwell" \
     --machine-type=$VM_MACHINE_TYPE \
     --accelerator=type=$VM_ACCELERATOR_TYPE \
     --boot-disk-size=512GB \
     --boot-disk-type=pd-standard \

     # Restarts are handled by `src/bin/gcp/keep_alive.py`
     --preemptible \  # Preemtiple machines cost up to 50% less
     --no-restart-on-failure \
     --maintenance-policy=TERMINATE \

     # Ensure machine can communicate with other VMs on GCP
     --scopes=https://www.googleapis.com/auth/cloud-platform \

     --image=ubuntu-1804-lts \
     --image-project=ubuntu-os-cloud
   ```

3. From your local repository, ssh into your new VM instance, like so:

   ```bash
   . src/bin/cloud/ssh_gcp.sh $VM_INSTANCE_NAME
   ```

### On the VM instance

1. Install these packages, like so:

   ```bash
   sudo apt-get update
   sudo apt-get install python3-venv -y
   sudo apt-get install python3-dev -y
   sudo apt-get install gcc -y
   sudo apt-get install sox -y
   sudo apt-get install ffmpeg -y
   sudo apt-get install ninja-build -y
   ```

2. Install GPU drivers on your VM by installing
   [CUDA-10-0](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork)
   , like so:

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   sudo apt-get update
   sudo apt-get install cuda
   ```

3. Verify CUDA installed correctly by running and ensuring no error messages print.

   ```bash
   nvidia-smi
   ```

4. Create a directory for our software.

   ```bash
   sudo chmod -R a+rwx /opt
   mkdir /opt/wellsaid-labs
   cd /opt/wellsaid-labs
   ```

### From your local repository

1. Use `src.bin.cloud.lsyncd` to live sync your repository to your VM instance:

   ```bash
   VM_USER=$(gcloud compute ssh $VM_INSTANCE_NAME --command="echo $USER")
   VM_IP_ADDRESS=$(gcloud compute instances describe $VM_INSTANCE_NAME --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
   VM_IDENTITY_FILE=~/.ssh/google_compute_engine
   python3 -m src.bin.cloud.lsyncd --public_dns $VM_IP_ADDRESS \
                                 --identity_file $VM_IDENTITY_FILE
                                 --source $(pwd) \
                                 --destination /opt/wellsaid-labs/Text-to-Speech \
                                 --user $VM_USER
   ```

2. When prompted, give your local sudo password for your laptop.

   Keep this process running on your local machine until you've started training, it'll
   allow you to make any hot-fixes to your code in case you run into an error.

### On the VM instance

1. Start a screen session:

   ```bash
   screen
   ```

2. Navigate to the repository, activate a virtual environment, and install package requirements:

   ```bash
   cd /opt/wellsaid-labs/Text-to-Speech

   # Note: You will always want to be in an active venv whenever you want to work with python.
   python3 -m venv venv
   . venv/bin/activate

   python -m pip install wheel
   python -m pip install -r requirements.txt --upgrade

   sudo bash src/bin/install_mkl.sh
   ```

3. Pick or create a comet project [here](https://www.comet.ml/wellsaid-labs). Afterwards set
   this variable:

   ```bash
   COMET_PROJECT="your-comet-project"
   ```

4. Train your ...

   ... spectrogram model

   ```bash
   pkill -9 python; \
   nvidia-smi; \
   PYTHONPATH=. python src/bin/train/spectrogram_model/__main__.py \
       --project_name $COMET_PROJECT
   ```

   ... signal model

   ```bash
   pkill -9 python; \
   nvidia-smi; \
   PYTHONPATH=. python src/bin/train/signal_model/__main__.py --project_name $COMET_PROJECT \
   --spectrogram_model_checkpoint $SPECTROGRAM_CHECKPOINT
   ```

   Note: the `--spectrogram_model_checkpoint` argument is optional
   (for example, see [here](TRAIN_TTS_MODEL_GCP.md#on-the-vm-instance)).

   We run `pkill -9 python` to kill any leftover processes from previous runs and `nvidia-smi`
   to ensure the GPU has no running processes.

5. Detach from your screen session by typing `Ctrl-A` then `D`. You can exit your VM with the
   `exit` command.

### From your local repository

1. Kill your `lsyncd` process by typing `Ctrl-C`.

2. Because we use preemptible cloud machines, they may occasionally be shut down, even mid-training.
   The following script will check the status of your machine, restart it if necessary, and restart
   the training process from the last recorded checkpoint. Keep this script running in order to keep
   your training running smoothly!

   For a spectrogram model ...

   ```bash
   python -m src.bin.cloud.keep_alive_gcp \
       --project_name $COMET_PROJECT \
       --instance $VM_INSTANCE_NAME \
       --command="screen -dmL bash -c \
                   'sudo chmod -R a+rwx /opt/;
                   cd /opt/wellsaid-labs/Text-to-Speech;
                   . venv/bin/activate;
                   PYTHONPATH=. python src/bin/train/spectrogram_model/__main__.py --checkpoint;'"
   ```

   For a signal model ...

   ```bash
   python -m src.bin.cloud.keep_alive_gcp \
       --project_name $COMET_PROJECT \
       --instance $VM_INSTANCE_NAME \
       --command="screen -dmL bash -c \
                   'sudo chmod -R a+rwx /opt/;
                   cd /opt/wellsaid-labs/Text-to-Speech;
                   . venv/bin/activate;
                   PYTHONPATH=. python src/bin/train/signal_model/__main__.py --checkpoint;'"
   ```

   If you're running this script from your laptop, then we recommend you install
   [Amphetamine](https://apps.apple.com/us/app/amphetamine/id937984704?mt=12) to keep your laptop
   from sleeping and stopping the script.

3. Once training has finished ...

   ... stop your VM

   ```bash
   gcloud compute instances stop $VM_INSTANCE_NAME --zone=$VM_INSTANCE_ZONE
   ```

   ... or delete your VM

   ```bash
   gcloud compute instances delete $VM_INSTANCE_NAME --zone=$VM_INSTANCE_ZONE
   ```
