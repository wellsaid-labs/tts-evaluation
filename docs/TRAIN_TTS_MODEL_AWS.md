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
   EXPERIMENT_KEY='' # Example: 16721307e95742cab096accb43cf3177
   ```

1. Now that you've finished training your spectrogram model, image your VM instance, like so:

   ```bash
   COMET_API_KEY=$(grep 'rest_api_key' .comet.config | cut -f2- -d'=')
   COMET_REST_API=https://www.comet.ml/api/rest/v2/experiment
   VM_NAME=$(curl --silent $COMET_REST_API/system-details?experimentKey=$EXPERIMENT_KEY \
     -H"Authorization: $COMET_API_KEY" | jq '.hostname' | xargs)
   VM_ID=$(aws ec2 describe-instances --filters Name=tag:Name,Values=$VM_NAME \
     --query 'Reservations[0].Instances[0].InstanceId' --output text)

   EXPERIMENT_LINK="https://www.comet.ml/api/experiment/redirect?experimentKey=$EXPERIMENT_KEY"
   EXPERIMENT_STEPS=$(curl --silent $COMET_REST_API/parameters?experimentKey=$EXPERIMENT_KEY \
     -H"Authorization: $COMET_API_KEY" |
     jq '.values[] | select(.name == "train_curr_step") | .valueCurrent' | xargs)
   IMAGE_DESCRIPTION="An image of a Comet experiment at $EXPERIMENT_STEPS steps. Link to experiment: $EXPERIMENT_LINK"

   VM_IMAGE_ID=$(aws ec2 create-image --instance-id $VM_ID \
     --name "Experiment $EXPERIMENT_KEY at $EXPERIMENT_STEPS steps" \
     --description "$IMAGE_DESCRIPTION" |
     jq '.ImageId' | xargs)
   ```

   💡 TIP: You can image an experiment at any time, it'll continue training after the image
   has been created.

   ❗IMPORTANT: This process can take from a couple minutes to an hour and a half.

1. You can monitor the creation of your image, here:

   ```bash
   open "https://$AWS_DEFAULT_REGION.console.aws.amazon.com/ec2/v2/home?region=$AWS_DEFAULT_REGION#Snapshots:sort=startTime"
   ```

   The UI will take a couple minutes to update before your experiment shows up.

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
