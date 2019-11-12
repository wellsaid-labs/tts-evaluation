# Train a TTS Model with Google Cloud Platform (GCP) Compute

This markdown will walk you through the steps to train an end-to-end text-to-speech model
on a GCP virtual machine.

## Train a Spectrogram Model

1. Follow this [document](TRAIN_MODEL.md) to train a spectrogram model.

   You'll want to train the spectrogram model until `dev_epoch/post_spectrogram_loss` stops
   rapidly decreasing. This typically will happen within 3 or so days of training.

## Train a Signal Model

### From your local repository

1. Now that you've finished training your spectrogram model, stop your VM instance, like so:

   ```bash
   gcloud compute instances stop $VM_INSTANCE_NAME --zone=$VM_INSTANCE_ZONE
   ```

2. Follow [this guide](https://cloud.google.com/compute/docs/instances/changing-machine-type-of-stopped-instance)
   to adjust the machine type from `n1-highmem-8` to `n1-highmem-16`.

3. Follow "Adding or removing GPUs on existing instances" on
   [this webpage](https://cloud.google.com/compute/docs/gpus/add-gpus) to adjust the number of
   GPUs from `nvidia-tesla-p100,count=2` to `nvidia-tesla-v100,count=8`.

4. Start your VM instance, like so:

   ```bash
   gcloud compute instances start $VM_INSTANCE_NAME --zone=$VM_INSTANCE_ZONE
   ```

### On the VM instance

1. During the training of the spectrogram model, it saved a number of model checkpoints. Find
   a checkpoint with a path similar to this one:

   ```bash
   SPECTROGRAM_CHECKPOINT="/opt/wellsaid-labs/Text-to-Speech/disk/experiments/" \
                          "spectrogram_model/{your_experiment}/runs/{a_run}/checkpoints/step-*.pt"
   ```

   Pick the checkpoint with a step count that correlates with the point of covergence
   `dev_epoch/post_spectrogram_loss` value.

2. From [this point](TRAIN_MODEL.md#on-the-vm-instance-1), continue following the instructions
   to train your signal model. That said, during the step where you run
   `python src/bin/train/signal_model/__main__.py` pass in the checkpoint you found earlier
   to the command line parameter `--spectrogram_model_checkpoint`.

   Finally, you'll want to train the signal model until `dev_epoch/coarse_loss` and
   `dev_epoch/fine_loss` stop rapidly decreasing. This typically will happen within 1 or so days of
   training.
