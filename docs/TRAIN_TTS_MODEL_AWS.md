# Train a TTS Model with Amazon Web Services (AWS)

This markdown will walk you through the steps to train an end-to-end text-to-speech model
on a AWS virtual machine.

## Train a Spectrogram Model

1. Follow this [document](TRAIN_MODEL_AWS.md) to train a spectrogram model.

   You'll want to train the spectrogram model until `dev_epoch/post_spectrogram_loss` stops
   rapidly decreasing. This typically will happen within 72 hours of training, or \~150,000 steps.

## Train a Signal Model

### From your local repository

1. Setup your environment variables...

   ```bash
   VM_REGION='your-vm-region' # EXAMPLE: us-east-1
   VM_NAME=$USER"_your_experiment_name" # EXAMPLE: michaelp_baseline
   VM_MACHINE_TYPE=p3.16xlarge
   ```

   ```bash
   VM_ID=$(aws --region=$VM_REGION ec2 describe-instances \
      --filters Name=tag:Name,Values=$VM_NAME \
      --query 'Reservations[0].Instances[0].InstanceId' \
      --output text)
   ```

1. Now that you've finished training your spectrogram model, stop your VM instance, like so:

   ```bash
   aws --region=$VM_REGION ec2 stop-instances --instance-ids $VM_INSTANCE_ID
   ```

1. Adjust the instance-type to the required type for the signal model...

   ```bash
   aws ec2 modify-instance-attribute \
      --instance-id $VM_INSTANCE_ID \
      --instance-type "Value=$VM_MACHINE_TYPE"
   ```

1. Start your VM instance, like so:

   ```bash
   aws --region=$VM_REGION ec2 start-instances --instance-ids $VM_INSTANCE_ID
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

2. From [this point](TRAIN_MODEL_AWS.md#on-the-vm-instance-1), continue following the instructions
   to train your signal model. That said, during the step where you run
   `python src/bin/train/signal_model/__main__.py` pass in the checkpoint you found earlier
   to the command line parameter `--spectrogram_model_checkpoint`.

   Finally, you'll want to train the signal model until `dev_epoch/coarse_loss` and
   `dev_epoch/fine_loss` stop rapidly decreasing. This typically will happen within 18-24 hours of
   training.
