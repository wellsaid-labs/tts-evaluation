# Train a Model with Amazon Web Services (AWS) Spot Instances

This markdown will walk you through the steps required to train a model on an AWS virtual
machine.

Related Documentation:

- Would you like to train a end-to-end TTS model? Please follow
  [this documentation](TRAIN_TTS_MODEL_AWS.md) instead.

## Prerequisites

Setup your local development environment by following [these instructions](LOCAL_SETUP.md).

## Train a Model with Amazon Web Services (AWS)

### From your local repository

1. Setup your environment variables...

   ... for training the spectrogram model...

   ```bash
   TRAIN_SCRIPT_PATH='run/train/spectrogram_model.py'
   ```

   ... for training the signal model...

   ```bash
   TRAIN_SCRIPT_PATH='run/train/signal_model.py'
   ```

   ❓ LEARN MORE: See our machine type benchmarks [here](./TRAIN_MODEL_AWS_BENCHMARKS.md).

   Also set these environment variables...

   ```bash
   AWS_DEFAULT_REGION='your-vm-region' # EXAMPLE: us-west-2
   VM_NAME=$USER"_your-instance-name" # EXAMPLE: michaelp_baseline

   echo "[default]
   region=$AWS_DEFAULT_REGION" > ~/.aws/config
   VM_MACHINE_TYPE='g4dn.12xlarge'
   VM_IMAGE_NAME='Deep Learning Base AMI (Ubuntu 18.04) Version 31.0'
   VM_IMAGE_ID=$(python -m run.utils.aws image-id $VM_IMAGE_NAME)
   VM_IMAGE_USER='ubuntu'
   AWS_SSH_KEY=~/.ssh/$USER"_amazon_web_services"
   ```

   ❓ LEARN MORE: About the default image
   [here](https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-Base-AMI-Ubu/B07Y3VDBNS)

   💡 TIP: Run this script to find regions with the least spot instance interruptions...
   `python -m run.utils.aws interruptions $VM_MACHINE_TYPE`

   💡 TIP: Don't place all your spot instances in the same region, just in case one region
   runs out of capacity.

1. Setup a startup script for the instance, like so...

   ```bash
   STARTUP_SCRIPT=$(cat run/utils/checkpoint_start_up.sh)
   STARTUP_SCRIPT=${STARTUP_SCRIPT//'$VM_USER'/\'$VM_IMAGE_USER\'}
   STARTUP_SCRIPT=${STARTUP_SCRIPT//'$TRAIN_SCRIPT_PATH'/\'$TRAIN_SCRIPT_PATH\'}
   ```

   💡 TIP: The output of the startup script will be saved on the VM here:
   `/var/log/cloud-init-output.log`

1. Create an EC2 instance for training...

   ```bash
   python -m run.utils.aws spot-instance --name=$VM_NAME --image-id=$VM_IMAGE_ID \
      --machine-type=$VM_MACHINE_TYPE --ssh-key-path=$AWS_SSH_KEY --startup-script="$STARTUP_SCRIPT"
   ```

   This instance will stay online for seven days or until you cancel the spot request.

   💡 TIP: "Spot request cannot be fulfilled due to invalid availability zone" can
   be resolved by setting the `--availability-zone`.

1. SSH into the instance...

   ```bash
   VM_PUBLIC_DNS=$(python -m run.utils.aws public-dns --name=$VM_NAME)
   ssh -i $AWS_SSH_KEY -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no \
      $VM_IMAGE_USER@$VM_PUBLIC_DNS
   ```

   Continue to run this command until it succeeds.

### On the instance

1. Create a directory for our software...

   ```bash
   sudo chmod -R a+rwx /opt
   mkdir /opt/wellsaid-labs
   ```

### From your local repository

1. Use `run.utils.lsyncd` to live sync your repository to your VM instance...

   ```bash
   VM_IMAGE_USER='ubuntu'
   AWS_SSH_KEY=~/.ssh/$USER"_amazon_web_services"
   VM_NAME=$(python -m run.utils.aws most-recent)
   VM_PUBLIC_DNS=$(python -m run.utils.aws public-dns --name=$VM_NAME)
   sudo python3 -m run.utils.lsyncd $(pwd) /opt/wellsaid-labs/Text-to-Speech \
                                    --public-dns $VM_PUBLIC_DNS \
                                    --user $VM_IMAGE_USER \
                                    --identity-file $AWS_SSH_KEY
   ```

   When prompted, enter your sudo password.

1. Leave this process running until you've started training. This will allow you to make any
   hot-fixes to your code in case you run into an error.

### On the instance

1. Install these packages, like so...

   ```bash
   sudo apt-get update
   ```

   If you get a `dkpg` error, wait a minute or so and try again.

   💡 TIP: After setting up your VM, you may want to
   [create an Amazon Machine Image (AMI)](https://docs.aws.amazon.com/cli/latest/reference/ec2/create-image.html)
   so you don't need to setup your VM from scratch again. Your first AMI for a particular setup may
   take a long time to create (1 hour or more) but it'll take less time for subsequent AMIs. You
   can see the AMI creation progress in the AWS console by viewing the AMIs corresponding snapshot.

1. Start a `screen` session...

   ```bash
   screen
   ```

1. Navigate to the repository, activate a virtual environment, and install package requirements...

   ```bash
   cd /opt/wellsaid-labs/Text-to-Speech

   . run/utils/apt_install.sh

   # NOTE: You will always want to be in an active venv whenever you want to work with python.
   python3 -m venv venv
   . venv/bin/activate

   python -m pip install wheel pip --upgrade
   python -m pip install -r requirements.txt --upgrade

   # NOTE: PyTorch 1.7 relies on CUDA 10.2, we enable it.
   sudo rm /usr/local/cuda
   sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda

   # NOTE: Set a flag to restart training if the instance is rebooted
   touch /opt/wellsaid-labs/AUTO_START_FROM_CHECKPOINT
   ```

1. Authorize GCP...

   ```bash
   . run/utils/gcp_install.sh
   gcloud init --console-only
   gcloud auth application-default login --no-launch-browser
   ```

1. For [comet](https://www.comet.ml/wellsaid-labs), name your experiment and pick a project...

   ```bash
   COMET_PROJECT='your-comet-project'
   EXPERIMENT_NAME='Your experiment name'
   ```

1. Start training...

   ```bash
   # NOTE: Kill any leftover processes from other runs...
   pkill -9 python; sleep 5s; nvidia-smi; \
   PYTHONPATH=. python $TRAIN_SCRIPT_PATH start --project $COMET_PROJECT --name "$EXPERIMENT_NAME";
   ```

   💡 TIP: You may want to include the optional
   `--spectrogram_model_checkpoint=$SPECTROGRAM_CHECKPOINT` argument.

1. Detach from your screen session by typing `Ctrl-A` then `D`.

1. You can now exit your VM with the `exit` command.

### From your local repository

1. Kill your `lsyncd` process by typing `Ctrl-C`.

## Post Training Clean Up

### From your local repository

1. Setup your environment variables again...

   ```bash
   VM_NAME=$USER"_your-instance-name" # EXAMPLE: michaelp_baseline
   ```

1. Cancel your spot instance request...

   ```bash
   SPOT_REQUEST_ID=$(python -m run.utils.aws spot-request-id --name=$VM_NAME)
   aws ec2 cancel-spot-instance-requests --spot-instance-request-ids $SPOT_REQUEST_ID
   ```

1. Delete your instance...

   ```bash
   VM_ID=$(python -m run.utils.aws instance-id --name=$VM_NAME)
   aws ec2 terminate-instances --instance-ids $VM_ID
   ```
