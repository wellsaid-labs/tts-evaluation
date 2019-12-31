# Train a TTS Model with Amazon Web Services (AWS) Spot Instances

This markdown will walk you through the steps to train an end-to-end text-to-speech model
on a AWS virtual machine.

## Train a Spectrogram Model

1. Follow this [document](TRAIN_MODEL_AWS_SPOT.md) to train a spectrogram model.

1. Wait until the `dev_epoch/post_spectrogram_loss` in your Comet experiment stops decreasing. This
   typically will happen within 72 hours of training, or \~150,000 steps.

## Train a Signal Model

### From your local repository

1. Setup your environment variables...

   ```bash
   export AWS_DEFAULT_REGION='your-vm-region' # EXAMPLE: us-west-2
   VM_NAME=$USER"_your-experiment-name" # EXAMPLE: michaelp_baseline
   COMET_EXPERIMENT_KEY='Your experiment id' # EXAMPLE: 28c3b1e63ff04f5aaefb4c2033ac4dae
   COMET_EXPERIMENT_STEPS='Number of steps completed' # EXAMPLE: 150k

   VM_ID=$(aws ec2 describe-instances --filters Name=tag:Name,Values=$VM_NAME \
                --query 'Reservations[0].Instances[0].InstanceId' --output text)
   COMET_EXPERIMENT_LINK="https://www.comet.ml/api/experiment/redirect?experimentKey=$COMET_EXPERIMENT_KEY"
   ```

1. Now that you've finished training your spectrogram model, image your VM instance, like so:

   ```bash
   VM_IMAGE_ID=$(aws ec2 create-image --instance-id $VM_ID \
      --name "Experiment $COMET_EXPERIMENT_KEY" \
      --description "An image of a Comet experiment at $COMET_EXPERIMENT_STEPS steps." \
      "Learn more: $COMET_EXPERIMENT_LINK" | \
      jq '.ImageId' | xargs)
   ```

   💡 TIP: You can image an experiment at any time, it'll continue training after the image
   has been created.

   ❗IMPORTANT: This process can take from a couple minutes to an hour and a half.

1. You can monitor the creation of your image, here:

   ```bash
   open https://$AWS_DEFAULT_REGION.console.aws.amazon.com/ec2/v2/home?region=$AWS_DEFAULT_REGION#Snapshots:sort=startTime
   ```

### On the VM instance

1. Follow this [document](TRAIN_MODEL_AWS_SPOT.md) to train a signal model.

   ❗IMPORTANT: Include the optional `--spectrogram_model_checkpoint=$SPECTROGRAM_CHECKPOINT`
   argument to start training.

   ❗IMPORTANT: Set the `VM_IMAGE_ID` parameter to the new image you created earlier.

   💡 TIP: In order to set the `--spectrogram_model_checkpoint` argument, find a checkpoint with a
   path similar to this one:

   ```bash
   SPECTROGRAM_CHECKPOINT="/opt/wellsaid-labs/Text-to-Speech/disk/experiments/" \
                          "spectrogram_model/{your_experiment}/runs/{a_run}/checkpoints/step-*.pt"
   ```

   Pick the checkpoint with a step count that correlates with the point of convergence
   `dev_epoch/post_spectrogram_loss` value.

1. Wait until `dev_epoch/coarse_loss` and `dev_epoch/fine_loss` stop rapidly
   decreasing. This typically will happen within 18-24 hours of training, or \~150,000 steps.
