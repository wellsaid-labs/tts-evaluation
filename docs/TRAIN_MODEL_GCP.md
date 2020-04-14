# Train a Model with Google Cloud Platform (GCP) Compute

This markdown will walk you through the steps required to train a model on an GCP virtual
machine.

Related Documentation:

- Would you like to train a end-to-end TTS model? Please follow
  [this documentation](TRAIN_TTS_MODEL_GCP.md) instead.

- You may want to learn more about the available VM configurations, you can learn more
  [here](https://console.cloud.google.com/compute/instancesAdd?project=voice-research-255602&organizationId=530338208816)
  and [here](https://cloud.google.com/sdk/gcloud/reference/compute/instances/create).

## Prerequisites

1. Setup your local development environment by following [these instructions](LOCAL_SETUP.md).

1. Install `gcloud compute` by following the instructions
   [here](https://cloud.google.com/compute/docs/gcloud-compute/)

1. Ask a team member to grant you access to our GCP project called "voice-research".

1. Install these dependencies:

   ```bash
   brew install rsync lsyncd
   ```

## Train a Model with Google Cloud Platform (GCP) Compute

### From your local repository

1. Setup your environment variables...

   Set these variables for training the spectrogram model...

   ```bash
   VM_MACHINE_TYPE=n1-highmem-8
   VM_ACCELERATOR_TYPE=nvidia-tesla-p100,count=2
   ```

   Set these variables for training the signal model...

   ```bash
   VM_MACHINE_TYPE=n1-highmem-32
   VM_ACCELERATOR_TYPE=nvidia-tesla-v100,count=8
   ```

   Also set these environment variables...

   ```bash
   VM_NAME=$USER"_your-instance-name" # EXAMPLE: michaelp_baseline

   # Pick a zone that supports your choosen `VM_ACCELERATOR_TYPE` using this chart:
   # https://cloud.google.com/compute/docs/gpus/
   # Note you'll want to spread your experiments out accross multiple zones to mitigate the risk of
   # your experiments getting throttled.
   VM_ZONE=your-vm-instance-zone
   ```

1. Create your virtual machine, like so:

   ```bash
   gcloud compute --project=voice-research-255602 instances create $VM_NAME \
     # A zone with the required resources using this chart
     # https://cloud.google.com/compute/docs/gpus/
     --zone=$VM_ZONE \

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

1. From your local repository, ssh into your new VM instance, like so:

   ```bash
   VM_ZONE=$(gcloud compute instances list | grep "^$VM_NAME\s" | awk '{ print $2 }')
   gcloud compute ssh --zone=$VM_ZONE $VM_NAME
   ```

   Continue to run this command until it succeeds.

### On the VM instance

1. Install these packages, like so...

   ```bash
   sudo apt-get update
   sudo apt-get install python3-venv python3-dev gcc g++ sox ffmpeg espeak -y
   ```

   💡 TIP: After setting up your VM, you may want to
   [create an image](https://cloud.google.com/compute/docs/images/create-delete-deprecate-private-images
   so you don't need to setup your VM from scratch again.

1. Install GPU drivers on your VM by installing
   [CUDA-10-0](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork)
   , like so:

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   sudo apt-get update
   sudo apt-get install cuda
   ```

1. Verify CUDA installed correctly by running and ensuring no error messages print.

   ```bash
   nvidia-smi
   ```

1. Create a directory for our software...

   ```bash
   sudo chmod -R a+rwx /opt
   mkdir /opt/wellsaid-labs
   ```

### From your local repository

1. In a new terminal window, setup your environment variables again...

   ```bash
   VM_NAME=$USER"_your-instance-name" # EXAMPLE: michaelp_baseline
   ```

1. Use `src.bin.cloud.lsyncd` to live sync your repository to your VM instance...

   ```bash
   VM_ZONE=$(gcloud compute instances list | grep "^$VM_NAME\s" | awk '{ print $2 }')
   VM_USER=$(gcloud compute ssh $VM_NAME --zone $VM_ZONE --command="echo $USER")
   VM_IP_ADDRESS=$(gcloud compute instances describe --zone $VM_ZONE $VM_NAME \
      --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
   IDENTITY_FILE=~/.ssh/google_compute_engine
   python3 -m src.bin.cloud.lsyncd --public_dns $VM_IP_ADDRESS \
                                 --identity_file $IDENTITY_FILE \
                                 --source $(pwd) \
                                 --destination /opt/wellsaid-labs/Text-to-Speech \
                                 --user $VM_USER
   ```

   When prompted, enter your sudo password.

1. Leave this processing running until you've started training. This will allow you to make any
   hot-fixes to your code in case you run into an error.

### On the VM instance

1. Start a `screen` session...

   ```bash
   screen
   ```

1. Navigate to the repository, activate a virtual environment, and install package requirements...

   ```bash
   cd /opt/wellsaid-labs/Text-to-Speech

   # NOTE: You will always want to be in an active venv whenever you want to work with python.
   python3 -m venv venv
   . venv/bin/activate

   python -m pip install wheel
   python -m pip install -r requirements.txt --upgrade

   # NOTE: PyTorch 1.4 relies on CUDA 10.1, we enable it.
   sudo rm /usr/local/cuda
   sudo ln -s /usr/local/cuda-10.1 /usr/local/cuda
   ```

1. For [comet](https://www.comet.ml/wellsaid-labs), name your experiment and pick a project...

   ```bash
   COMET_PROJECT='your-comet-project'
   EXPERIMENT_NAME='Your experiment name'
   ```

   ... and this variable for the spectrogram model ...

   ```bash
   TRAIN_SCRIPT_PATH='src/bin/train/spectrogram_model/__main__.py'
   ```

   ... or this variable for the signal model ...

   ```bash
   TRAIN_SCRIPT_PATH='src/bin/train/signal_model/__main__.py'
   ```

1. Start training...

   ```bash
   # Kill any leftover processes from other runs...
   pkill -9 python; sleep 5s; nvidia-smi; \
   PYTHONPATH=. python $TRAIN_SCRIPT_PATH --project_name $COMET_PROJECT --name "$EXPERIMENT_NAME";
   ```

   💡 TIP: You may want to include the optional `--spectrogram_model_checkpoint=your-checkpoint.pt`
   argument.

1. Detach from your screen session by typing `Ctrl-A` then `D`.

1. You can now exit your VM with the `exit` command.

### From your local repository

1. Kill your `lsyncd` process by typing `Ctrl-C`.

1. Start the below process and leave it running for the duration of the training to deal with any
   unexpected instance interruptions...

   ```bash
   TRAIN_SCRIPT_PATH='your-choosen-training-script-from-earlier'
   COMET_PROJECT='your-choosen-comet-project-from-earlier'
   python -m src.bin.cloud.keep_alive_gcp \
       --project_name $COMET_PROJECT \
       --instance $VM_NAME \
       --command="screen -dmL bash -c \
                   'sudo chmod -R a+rwx /opt/;
                   cd /opt/wellsaid-labs/Text-to-Speech;
                   . venv/bin/activate;
                   PYTHONPATH=. python $TRAIN_SCRIPT_PATH --checkpoint;'"
   ```

   💡 TIP: If you're running this script from your laptop, then we recommend you install
   [Amphetamine](https://apps.apple.com/us/app/amphetamine/id937984704?mt=12) to keep your laptop
   from sleeping and stopping the script.

## Post Training Clean Up

### From your local repository

1. Setup your environment variables again...

   ```bash
   VM_NAME=$USER"_your-instance-name" # EXAMPLE: michaelp_baseline
   VM_ZONE=$(gcloud compute instances list | grep "^$VM_NAME\s" | awk '{ print $2 }')
   ```

1. Stop your instance...

   ```bash
   gcloud compute instances stop $VM_NAME --zone=$VM_ZONE
   ```

   or delete your instance...

   ```bash
   gcloud compute instances delete $VM_NAME --zone=$VM_ZONE
   ```
