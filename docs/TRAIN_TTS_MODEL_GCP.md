# Train a TTS Model with Google Cloud Platform (GCP) Compute

This markdown will walk you through the steps to train an end-to-end text-to-speech model
on a GCP virtual machine.

## Train a Spectrogram Model

1. Follow this [document](TRAIN_MODEL_GCP.md) to train a spectrogram model.

1. Wait until the `dev_epoch/post_spectrogram_loss` in your Comet experiment stops decreasing. This
   typically will happen within 72 hours of training, or \~150,000 steps.

## Train a Signal Model

### From your local repository

1. Setup your environment variables...

   ```bash
   VM_NAME=$USER"_your-experiment-name" # EXAMPLE: michaelp_baseline
   VM_ZONE=$(gcloud compute instances list | grep "^$VM_NAME\s" | awk '{ print $2 }')
   ```

1. Now that you've finished training your spectrogram model, stop your VM instance, like so:

   ```bash
   gcloud compute instances stop $VM_NAME --zone=$VM_ZONE
   ```

1. Follow [this guide](https://cloud.google.com/compute/docs/instances/changing-machine-type-of-stopped-instance)
   to adjust the machine type from `n1-highmem-8` to `n1-highmem-16`.

1. Follow "Adding or removing GPUs on existing instances" on
   [this webpage](https://cloud.google.com/compute/docs/gpus/add-gpus) to adjust the number of
   GPUs from `nvidia-tesla-p100,count=2` to `nvidia-tesla-v100,count=8`.

1. Start your VM instance, like so:

   ```bash
   gcloud compute instances start $VM_NAME --zone=$VM_ZONE
   ```

### On the VM instance

1. During the training of the spectrogram model, it saved a number of model checkpoints. Find
   a checkpoint with a path similar to this one:

   ```bash
   SPECTROGRAM_CHECKPOINT="/opt/wellsaid-labs/Text-to-Speech/disk/experiments/" \
                          "spectrogram_model/{your_experiment}/runs/{a_run}/checkpoints/step-*.pt"
   ```

   Pick the checkpoint with a step count that correlates with the point of convergence
   `dev_epoch/post_spectrogram_loss` value.

1. From [this point](TRAIN_MODEL_GCP.md#on-the-vm-instance-1), continue following the instructions
   to train your signal model.

   ❗IMPORTANT: Include the optional `--spectrogram_model_checkpoint=$SPECTROGRAM_CHECKPOINT`
   argument to start training.

1. Wait until `dev_epoch/coarse_loss` and `dev_epoch/fine_loss` stop rapidly
   decreasing. This typically will happen within 18-24 hours of training, or \~150,000 steps.
