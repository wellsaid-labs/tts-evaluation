# Train a Model with Google Cloud Platform (GCP) Preemptible Instances

This markdown will walk you through the steps required to train a model on a GCP virtual
machine.

## Prerequisites

Set up your local development environment by following [these instructions](LOCAL_SETUP.md).

## Train a Model with Google Cloud Platform (GCP)

### From your local repository

1. Setup your variable environment...

   ```zsh
   NAME=$USER"-your-instance-name" # EXAMPLE: michaelp-baseline
   vars make $NAME
   vars activate $NAME
   export NAME=$NAME
   export TRAIN_SCRIPT_PATH='path/to/train' # EXAMPLE: run/train/spectrogram_model
   export TYPE='preemptible' # Either 'preemptible' or 'persistent'
   ```

   ❓ LEARN MORE: Preemptible instances are much lower cost than standard/persistent VMs, with the
   tradeoff that "Compute Engine might stop (preempt) these instances if it needs to reclaim the
   compute capacity for allocation to other VMs." Read more:
   [Preemptible VM instances](https://cloud.google.com/compute/docs/instances/preemptible)

   💡 TIP: Find zones with that support T4 GPUs here:
   [GPU regions and zones availability](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones)

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
   IMAGE_FAMILY='<your-image-family-name>'    # Example: $IMAGE_FAMILY used to image your machine
   ```

   Alternatively, you can specify an image instead of image family:

  ```zsh
  IMAGE_PROJECT='voice-research-255602'
  IMAGE='<your-image-name>'
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
      --image-family=$IMAGE_FAMILY \  # Or swap to --image=$IMAGE
      --metadata="startup-script-user=$USER" \
      --metadata="train-script-path=$TRAIN_SCRIPT_PATH" \
      --metadata-from-file="startup-script=run/utils/gcp/resume_training_on_start_up.sh" \
   ```

    You can just use these zones for persistent training...

    ```zsh
      --preferred-zone="us-central1-a" \
      --preferred-zone="us-central1-c"
    ```

   You may exlcude these zones from pre-emptible training...

   ```zsh
      --exclude-zone="asia-east1-a" \
      --exclude-zone="asia-east1-b" \
      --exclude-zone="asia-northeast2-a" \
      --exclude-zone="asia-northeast2-c" \
      --exclude-zone="asia-northeast3-a" \
      --exclude-zone="asia-northeast3-a" \
      --exclude-zone="asia-northeast3-b" \
      --exclude-zone="asia-northeast3-b" \
      --exclude-zone="asia-south1-c" \
      --exclude-zone="asia-southeast1-a" \
      --exclude-zone="australia-southeast1-a" \
      --exclude-zone="australia-southeast2-c" \
      --exclude-zone="europe-central2-a" \
      --exclude-zone="europe-central2-b" \
      --exclude-zone="europe-north1-a" \
      --exclude-zone="europe-north1-b" \
      --exclude-zone="europe-west12-a" \
      --exclude-zone="europe-west12-b" \
      --exclude-zone="europe-west2-c" \
      --exclude-zone="europe-west3-c" \
      --exclude-zone="europe-west4-c" \
      --exclude-zone="europe-west6-a" \
      --exclude-zone="europe-west8-b" \
      --exclude-zone="europe-west8-c" \
      --exclude-zone="europe-west9-b" \
      --exclude-zone="europe-west9-c" \
      --exclude-zone="me-west1-b" \
      --exclude-zone="me-west1-c" \
      --exclude-zone="northamerica-northeast1-b" \
      --exclude-zone="northamerica-northeast2-b" \
      --exclude-zone="northamerica-northeast2-c" \
      --exclude-zone="southamerica-east1-a" \
      --exclude-zone="southamerica-east1-b" \
      --exclude-zone="southamerica-east1-c" \
      --exclude-zone="southamerica-west1-a" \
      --exclude-zone="southamerica-west1-c" \
      --exclude-zone="us-central1-a" \
      --exclude-zone="us-central1-b" \
      --exclude-zone="us-central1-c" \
      --exclude-zone="us-central1-f" \
      --exclude-zone="us-east1-b" \
      --exclude-zone="us-south1-a" \
      --exclude-zone="us-south1-b" \
      --exclude-zone="us-south1-c" \
      --exclude-zone="us-west1-a" \
      --exclude-zone="us-west2-b" \
      --exclude-zone="us-west2-c" \
      --exclude-zone="us-west3-a" \
      --exclude-zone="us-west3-b" \
      --exclude-zone="us-west3-c"
    ```

   📙 NOTE: Please add a zone to our `--exclude-zone` list if your instance dies frequently
   or quickly in that zone.

   📙 NOTE: This will use the environment variable `$USER` as the username whilst the browser
   console will use your email address username.

   ❓ LEARN MORE: See our machine type benchmarks [here](./TRAIN_MODEL_GCP_BENCHMARKS.md).

   📙 NOTE: You can look for your image and see its status via the Google Cloud console:
   [VM instances](https://console.cloud.google.com/compute/instances?project=voice-research-255602).

   ❓ LEARN MORE: See our machine type benchmarks:
   [Train a Model with Google Cloud Platform (GCP) Benchmarks](./TRAIN_MODEL_GCP_BENCHMARKS.md).

   💡 TIP: The output of the startup script will be saved on the VM at:
   `/var/log/syslog`. The relevant logs will start after
    "Starting Google Compute Engine Startup Scripts..." is logged.

1. SSH into the instance...

   ```zsh
   export VM_NAME=$(python -m run.utils.gcp $TYPE most-recent --name $NAME)
   echo "VM_NAME=$VM_NAME"
   export VM_ZONE=$(python -m run.utils.gcp zone --name $VM_NAME)
   export VM_IP=$(python -m run.utils.gcp ip --name $VM_NAME --zone=$VM_ZONE)

   gcloud compute ssh --zone=$VM_ZONE $VM_NAME
   ```

   Continue to run this command until it succeeds. It may take up to a few minutes for the instance to be available.

   💡 TIP: Preemptible machines will be periodically recreated, so you will need fetch a new
   `VM_NAME` and `VM_IP`, every so often.

   💡 TIP: Keep in mind, if this continues to time out (e.g. "port 22: Operation timed out"), your
   router may be blocking SSH connections
   <https://serverfault.com/questions/25545/why-block-port-22-outbound>.

   💡 TIP: This command may create a user and transfer SSH keys. You may delete those, here:
   <https://console.cloud.google.com/compute/metadata?tab=sshkeys&project=voice-research-255602&orgonly=true&supportedpurview=organizationId>

### On the instance

1. Create a directory for our software...

   ```bash
   sudo chmod -R 777 /opt
   mkdir /opt/wellsaid-labs
   ```

### From your local repository

1. (Optional) You may need to open a console, connected to your environment...

   ```bash
   NAME=$USER"-your-instance-name" # EXAMPLE: michaelp-baseline
   vars activate $NAME
   ```

1. Live sync your repository to your VM instance...

   ```bash
   sudo python3 -m run.utils.lsyncd $(pwd) /opt/wellsaid-labs/Text-to-Speech \
                                    --public-dns $VM_IP \
                                    --user $USER \
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
   COMET_PRJ='<your-comet-project>'
   EXP_NAME='<Your experiment name>'
   ```

1. You may want to double check any code changes you made, before running the experiment...

   ```bash
   git --no-pager diff run/ lib/
   ```

1. You may want to reset various services and clear out old processes, just in case...

   ```bash
   sudo rmmod nvidia_uvm
   sudo modprobe nvidia_uvm
   pkill -9 python
   sleep 5s
   nvidia-smi
   ```

1. Start training...

   For example, run this command to train a spectrogram model:

   ```bash
   PYTHONPATH=. python $TRAIN_SCRIPT_PATH start $COMET_PRJ "$EXP_NAME";
   ```

   For example, run this command to find the latest spectrogram model and train a signal model:

   ```bash
   DIR_NAME="disk/experiments/spectrogram_model/"
   SPEC_CHKPNT=$( \
    find $DIR_NAME -name '*.pt' -printf '%T+ %p\n' | \
    sort -r | head -n 1 | cut -f2- -d" ")
   SPEC_CHKPNT="/opt/wellsaid-labs/Text-to-Speech/$SPEC_CHKPNT"
   echo $SPEC_CHKPNT
   ```

   ```bash
   PYTHONPATH=. python $TRAIN_SCRIPT_PATH start $SPEC_CHKPNT $COMET_PRJ "$EXP_NAME";
   ```

   For example, run this command to fine tune an older signal model, with a new spectrogram
   checkpoint:

   ```bash
   FINE_TUNE="<your_deployed_checkpoint>"  # EXAMPLE: `v11_2023_04_25_staging`
   PYTHONPATH=. python $TRAIN_SCRIPT_PATH fine-tune $FINE_TUNE $SPEC_CHKPNT $COMET_PRJ "$EXP_NAME";
   ```

   ❓ LEARN MORE: PyTorch leaves zombie processes that must be killed, check out:
   <https://leimao.github.io/blog/Kill-PyTorch-Distributed-Training-Processes/>

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

1. (Optional) You may need to open a console, connected to your environment...

   ```zsh
   NAME=$USER"-your-instance-name" # EXAMPLE: michaelp-baseline
   vars activate $NAME
   ```

1. (Optional) Download the latest checkpoint to your local drive...

   ```bash
   DIR_NAME='spectrogram_model' # EXAMPLE: spectrogram_model
   ```

   ```bash
   python -m run.utils.checkpoints download-latest $VM_ZONE $VM_NAME $DIR_NAME
   ```

   You can also batch download all the latest checkpoints from every machine online...

   ```bash
   python -m run.utils.checkpoints download-all-latest $USER $DIR_NAME
   ```

1. Delete your instance...

   ```zsh
   python -m run.utils.gcp $TYPE delete-instance --name=$NAME --zone=$VM_ZONE
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
   use your `$IMAGE_FAMILY` environment variable to build your new instance!
