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
   NAME=$USER"-your-instance-name" # EXAMPLE: michaelp-baseline
   GCP_USER='your-gcp-user-name' # Example: michaelp
   TYPE='preemptible' # Either 'preemptible' or 'persistent'
   ```

   💡 TIP: Find zones with that support T4 GPUs here:
   https://cloud.google.com/compute/docs/gpus/gpu-regions-zones

   💡 TIP: Don't place all your preemptible instances in the same zone, just in case one zone
   runs out of capacity.

   💡 TIP: For persistent instances, the `ZONE` parameter is optional. If it's not provided, then
    this will try all the zones until one is found.

   If starting from scratch, use a standard ubuntu image:

   ```zsh
   IMAGE_PROJECT='ubuntu-os-cloud'
   IMAGE_FAMILY='ubuntu-1804-lts'
   ```

   If starting from your own image, set the image project and family appropriately:

   ```zsh
   IMAGE_PROJECT='voice-research-255602'
   IMAGE_FAMILY='your-image-family-name'    # Example: $IMAGE_FAMILY used to image your machine
   ```

1. Create an instance for training...

   ```zsh
python -m run.utils.gcp $TYPE make-instance \
  --name=$NAME \
  --machine-type='n1-highmem-32' \
  --gpu-type='nvidia-tesla-t4' \
  --gpu-count=4 \
  --disk-size=1024 \
  --disk-type='pd-balanced' \
  --image-project=$IMAGE_PROJECT \
  --image-family=$IMAGE_FAMILY \
  --metadata="startup-script-user=$GCP_USER" \
  --metadata="train-script-path=$TRAIN_SCRIPT_PATH" \
  --metadata-from-file="startup-script=run/utils/gcp/resume_training_on_start_up.sh"
   ```

   ❓ LEARN MORE: See our machine type benchmarks [here](./TRAIN_MODEL_GCP_BENCHMARKS.md).

   💡 TIP: The output of the startup script will be saved on the VM here:
   `/var/log/syslog`. The relevant logs will start after
    "Starting Google Compute Engine Startup Scripts..." is logged.

1. SSH into the instance...

   ```zsh
   VM_NAME=$(python -m run.utils.gcp $TYPE most-recent --name $NAME)
   echo "VM_NAME=$VM_NAME"
   VM_ZONE=$(python -m run.utils.gcp zone --name $VM_NAME)
   gcloud compute ssh --zone=$ZONE $NAME
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
   VM_NAME=$(python -m run.utils.gcp $TYPE most-recent --name $NAME)
   echo "VM_NAME=$VM_NAME"
   ```

   ```bash
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

   ```bash
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
   [create a Google Machine Image](https://cloud.google.com/compute/docs/machine-images/create-machine-images)
   so you don't need to setup your VM from scratch again.

1. Download any spaCy models that you may need, potentially including...

   ```bash
   python -m spacy download en_core_web_md
   python -m spacy download de_core_news_md
   python -m spacy download es_core_news_md
   python -m spacy download pt_core_news_md
   ```

1. Start a `screen` session with a new virtual environment...

   ```bash
   screen
   ```

   ```bash
   . venv/bin/activate
   ```

1. For [comet](https://www.comet.ml/wellsaid-labs), name your experiment and pick a project...

   ```bash
   COMET_PROJECT='your-comet-project'
   EXPERIMENT_NAME='Your experiment name'
   ```

1. Start training...

   For example, run this command to train a spectrogram model:

   ```bash
   pkill -9 python; sleep 5s; nvidia-smi; \
   PYTHONPATH=. python $TRAIN_SCRIPT_PATH start $COMET_PROJECT "$EXPERIMENT_NAME";
   ```

   ---
   Or select a `SPECTROGRAM_CHECKPOINT`...
   ```
   find /opt/wellsaid-labs/Text-to-Speech/disk/experiments/spectrogram_model/ -name <step_######.pt>
   ```
   And store it...
   ```
   SPECTROGRAM_CHECKPOINT="<paste>"
   ```

   Then run the following command to train a signal model...

   ```bash
   SPECTROGRAM_CHECKPOINT="/opt/wellsaid-labs/Text-to-Speech/path/to/spectrogram/checkpoint"
   ```

   ```bash
   pkill -9 python; sleep 5s; nvidia-smi; \
   PYTHONPATH=. python $TRAIN_SCRIPT_PATH start $SPECTROGRAM_CHECKPOINT $COMET_PROJECT "$EXPERIMENT_NAME";
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
   NAME=$USER"-your-instance-name" # EXAMPLE: michaelp-baseline
   VM_NAME=$(python -m run.utils.gcp $TYPE most-recent --name $NAME)
   echo "VM_NAME=$VM_NAME"
   VM_ZONE=$(python -m run.utils.gcp zone --name $VM_NAME)
   ```

1. (Optional) Download checkpoints to your local drive...

   ```bash
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
   python -m run.utils.gcp $TYPE delete-instance --name=$VM_NAME --zone=$VM_ZONE
   ```

   You may need to run the above a couple of times.

   💡 TIP: The instance can be imaged and deleted. For example:

   ```zsh
   IMAGE_FAMILY=$NAME # EXAMPLE: michaelp-baseline
   IMAGE_NAME="$IMAGE_FAMILY-v1" # EXAMPLE: michaelp-baseline-v1
   gcloud compute ssh --zone=$VM_ZONE $VM_NAME \
      --command="rm /opt/wellsaid-labs/AUTO_START_FROM_CHECKPOINT"
   python -m run.utils.gcp $TYPE image-and-delete \
      --image-family=$IMAGE_FAMILY \
      --image-name=$IMAGE_NAME \
      --name=$NAME \
      --vm-name=$VM_NAME \
      --zone=$VM_ZONE
   ```

   When you're ready to begin signal model training, start from the top of these instructions and
   use your `$IMAGE_FAMILY` envrionment variable to build your new instance!
