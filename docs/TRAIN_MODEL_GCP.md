# Train a Model with Google Cloud Platform (GCP) Preemptible Instances

This markdown will walk you through the steps required to train a model on an GCP virtual
machine.

Related Documentation:

- Would you like to train a end-to-end TTS model? TODO

## Prerequisites

Setup your local development environment by following [these instructions](LOCAL_SETUP.md).

## Train a Model with Google Cloud Platform (GCP)

### From your local repository

1. Setup your environment variables...

   Also set these environment variables...

   ```zsh
   TRAIN_SCRIPT_PATH='path/to/train' # EXAMPLE: run/train/spectrogram_model
   ZONE='your-vm-zone' # EXAMPLE: us-east1-c
   NAME=$USER"-your-instance-name" # EXAMPLE: michaelp-baseline
   GCP_USER='your-gcp-user-name' # Example: michaelp
   ```

   💡 TIP: Don't place all your preemptible instances in the same zone, just in case one zone
   runs out of capacity.

1. Create an instance for training...

   ```zsh
   python -m run.utils.gcp make-instance \
      --name=$NAME \
      --zone=$ZONE \
      --machine-type='n1-standard-32' \
      --gpu-type='nvidia-tesla-t4' \
      --gpu-count=4 \
      --disk-size=512 \
      --disk-type='pd-balanced' \
      --image-project='ubuntu-os-cloud' \
      --image-family='ubuntu-1804-lts' \
      --metadata="startup-script-user=$GCP_USER" \
      --metadata="train-script-path=$TRAIN_SCRIPT_PATH" \
      --metadata-from-file="startup-script=run/utils/gcp/resume_training_on_start_up.sh"
   python -m run.utils.gcp watch-instance --name=$NAME --zone=$ZONE
   ```

   ❓ LEARN MORE: See our machine type benchmarks [here](./TRAIN_MODEL_GCP_BENCHMARKS.md).

   💡 TIP: The output of the startup script will be saved on the VM here:
   `/var/log/syslog`. The relevant logs will start after
    "Starting Google Compute Engine Startup Scripts..." is logged.

1. SSH into the instance...

   ```zsh
   VM_NAME=$(python -m run.utils.gcp most-recent --name $NAME)
   echo "VM_NAME=$VM_NAME"
   gcloud compute ssh --zone=$ZONE $VM_NAME
   ```

   Continue to run this command until it succeeds.

### On the instance

1. Create a directory for our software...

   ```bash
   sudo chmod -R 777 /opt
   mkdir /opt/wellsaid-labs
   ```

### From your local repository

1. Use `run.utils.lsyncd` to live sync your repository to your VM instance...

   ```bash
   VM_NAME=$(python -m run.utils.gcp most-recent --filter $USER)
   echo "VM_NAME=$VM_NAME"
   VM_ZONE=$(python -m run.utils.gcp zone --name $VM_NAME)
   VM_IP=$(python -m run.utils.gcp ip --name $VM_NAME --zone=$VM_ZONE)
   VM_USER=$(python -m run.utils.gcp user --name $VM_NAME --zone=$VM_ZONE)
   ```

   ```bash
   sudo python3 -m run.utils.lsyncd $(pwd) /opt/wellsaid-labs/Text-to-Speech \
                                    --public-dns $VM_IP \
                                    --user $VM_USER \
                                    --identity-file ~/.ssh/google_compute_engine
   ```

   When prompted, enter your sudo password.

   💡 TIP: The `VM_NAME` filter can be `$NAME` or any relevant substring in the `$VM_NAME`.

1. Leave this process running until you've started training. This will allow you to make any
   hot-fixes to your code in case you run into an error.

### On the instance

1. Navigate to the repository, activate a virtual environment, and install package requirements...

   ```bash
   cd /opt/wellsaid-labs/Text-to-Speech

   . run/utils/gcp/install_drivers.sh
   . run/utils/apt_install.sh
   ```
   **NOTE:** You will always want to be in an active `venv` whenever you want to work with python.
   ```
   python3.8 -m venv venv
   . venv/bin/activate

   python -m pip install wheel pip --upgrade
   python -m pip install -r requirements.txt --upgrade

   # NOTE: Set a flag to restart training if the instance is rebooted
   # NOTE: Learn more about this command:
   # https://askubuntu.com/questions/21556/how-to-create-an-empty-file-from-command-line
   :>> /opt/wellsaid-labs/AUTO_START_FROM_CHECKPOINT
   ```

   💡 TIP: After setting up your VM, you may want to
   [create an Google Machine Image](https://cloud.google.com/compute/docs/machine-images/create-machine-images)
   so you don't need to setup your VM from scratch again.

1. Start a `screen` session with a new virtual environment...

   ```bash
   screen
   . venv/bin/activate
   ```

1. For [comet](https://www.comet.ml/wellsaid-labs), name your experiment and pick a project...

   ```bash
   COMET_PROJECT='your-comet-project'
   EXPERIMENT_NAME='Your experiment name'
   ```

1. Start training... For example, run this command to train a spectrogram model:

   ```bash
   pkill -9 python; sleep 5s; nvidia-smi; \
   PYTHONPATH=. python $TRAIN_SCRIPT_PATH start $COMET_PROJECT "$EXPERIMENT_NAME";
   ```

   ❓ LEARN MORE: PyTorch leaves zombie processes that must be killed, check out:
   https://leimao.github.io/blog/Kill-PyTorch-Distributed-Training-Processes/

1. Detach from your screen session by typing `Ctrl-A` then `D`.

1. You can now exit your VM with the `exit` command.

### From your local repository

1. Kill your `lsyncd` process by typing `Ctrl-C`.

## Post Training Clean Up

### On the instance

1. (Optional) Upload the checkpoints to Google Cloud Storage...

   ```bash
   NAME='' # EXAMPLE: super_hi_fi__custom_voice
   gsutil -m cp -r -n disk/experiments/ gs://wsl_experiments/$NAME
   ```

### From your local repository

1. Setup your environment variables again...

   ```zsh
   ZONE='your-vm-zone' # EXAMPLE: us-central1-a
   NAME=$USER"-your-instance-name" # EXAMPLE: michaelp-baseline
   ```

1. (Optional) Download checkpoints to your local drive...

   ```bash
   VM_NAME=$(python -m run.utils.gcp most-recent --name $NAME)
   VM_ZONE=$(python -m run.utils.gcp zone --name $VM_NAME)

   DIR_NAME='' # EXAMPLE: spectrogram_model
   CHECKPOINT='' # EXAMPLE: '**/**/checkpoints/step_630927.pt'

   DEST="disk/experiments/$DIR_NAME/$VM_NAME/"
   mkdir -p $DEST
   gcloud compute scp \
      $VM_NAME:/opt/wellsaid-labs/Text-to-Speech/disk/experiments/$DIR_NAME/$CHECKPOINT \
      $DEST --zone=$VM_ZONE
   ```

1. Delete your instance...

   ```zsh
   python -m run.utils.gcp delete-instance --name=$NAME --zone=$ZONE
   ```

   You may need to run the above a couple of times.

   💡 TIP: The instance can be imaged and deleted. For example:

   ```zsh
   IMAGE_FAMILY=$NAME # EXAMPLE: michaelp-baseline
   IMAGE_NAME="$IMAGE_FAMILY-v1" # EXAMPLE: michaelp-baseline-v1
   VM_NAME=$(python -m run.utils.gcp most-recent --name $NAME)
   VM_ZONE=$(python -m run.utils.gcp zone --name $VM_NAME)
   gcloud compute ssh --zone=$ZONE $VM_NAME \
      --command="rm /opt/wellsaid-labs/AUTO_START_FROM_CHECKPOINT"
   python -m run.utils.gcp image-and-delete \
      --image-family=$IMAGE_FAMILY \
      --image-name=$IMAGE_NAME \
      --name=$NAME \
      --vm-name=$VM_NAME \
      --zone=$VM_ZONE
   gcloud compute disks delete $VM_NAME --zone=$VM_ZONE
   ```
