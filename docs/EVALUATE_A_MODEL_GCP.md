# Model Evaluation

WellSaid Labs utilizes an industry standard of Mean Opinion Scores to gauge synthetic audio quality.
We use [MTurk](https://www.mturk.com/) to crowd source opinion scores of various audio samples and
derive a MOS per synthetic voice.

This document outlines the process for creating the MTurk surveys to obtain these scores.

## Background

MTurk utilizes a CSV upload in order to generate survey questions in batches. There are several
steps involved in generating the CSV for upload to MTurk and for storing the associated audio files
in our Google Cloud Storage bucket.

## Prerequisites

1. Setup your local development environment by following [these instructions](LOCAL_SETUP.md).

2. Install `gcloud compute` by following the instructions
   [here](https://cloud.google.com/compute/docs/gcloud-compute/)

3. Use a VM that was setup to train a spectrogram or signal model with
   [these instructions](TRAIN_MODEL.md).

4. Ask a team member to grant you access to our GCP project called "voice-research".

## [OPTIONAL] Modify the script

Should you be interested in curating the random samples of audio being sent to MTurk for testing,
you can modify `evaluate.py` to filter the samples before generating the CSV. For example, perhaps
you want to test the quality of one specific `Speaker` from a model. You can filter your models for
that Speaker in main, before running `evaluate.py`.

```python
def main(
  ...
  speakers=[Speaker(name="Hilary Noriega", gender=Gender.FEMALE)],
  ...
  ):
```

## On the VM instance

1. Setup your Python environment.

   ```bash
   cd /opt/wellsaid-labs/Text-to-Speech

   # Activate the virtual environment
   . venv/bin/activate
   ```

2. Setup your environment variables, once per model, like so:

   ```bash
   ROOT_PATH='/opt/wellsaid-labs/Text-to-Speech/disk/experiments'
   SIGNAL_MODEL="$ROOT_PATH/signal_model/[DATE]-[TIME]-[PID]/runs/[DATE]-[TIME]-[PID]/checkpoints/step_XXXXXX.pt"
   SPECTROGRAM_MODEL="$ROOT_PATH/spectrogram_model/[DATE]-[TIME]-[PID]/runs/[DATE]-[TIME]-[PID]/checkpoints/step_XXXXXX.pt"
   MODEL_NAME='your-model-name'
   NUM_SAMPLES=512
   ```

3. Run `evaluate.py` once per model, like so:

   ```bash
   python -m src.bin.evaluate \
      --signal_model=$SIGNAL_MODEL \
      --spectrogram_model=$SPECTROGRAM_MODEL \
      --name=$MODEL_NAME \
      --num_samples=$NUM_SAMPLES \
      --no_target_audio \
      --no_griffin_lim \
      --obscure_filename
   ```

   If you'd like to learn more about these parameters, please run:

   ```bash
   python -m src.bin.evaluate --help
   ```

   Go back to the previous step if you have other models you'd like to include in this evaluation.

   This may take a couple hours by the way.

4. Run `evaluate.py` an extra time to get ground truth samples to compare your TTS model against,
   like so:

   ```bash
   python -m src.bin.evaluate --num_samples=$NUM_SAMPLES --obscure_filename --name='ground-truth'
   ```

## From your local repository

1. Setup your Python environment.

   ```bash
   . venv/bin/activate
   ```

2. Setup your environment variables, like so:

   ```bash
   VM_INSTANCE_NAME=''
   ```

3. Download your samples to `~/Downloads`.

   ```bash
   VM_ZONE=$(gcloud compute instances list | grep "^$VM_INSTANCE_NAME\s" | awk '{ print $2 }')
   VM_USER=$(gcloud compute ssh $VM_INSTANCE_NAME --zone=$VM_ZONE --command="echo $USER")
   mkdir ~/Downloads/samples/
   gcloud compute scp \
      --recurse \
      --zone=$VM_ZONE \
      $VM_USER@$VM_INSTANCE_NAME:/opt/wellsaid-labs/Text-to-Speech/disk/samples/ \
      ~/Downloads/
   ```

   If your working with multiple VMs, you'll want to rerun this step for each VM.

4. Combine the the samples you generated into one batch, like so:

   ```bash
   BATCH_NAME_SUFFIX=''
   ```

   ```bash
   BATCH_NAME_PREFIX=$(date '+%Y-%m-%d')
   BATCH_NAME="$BATCH_NAME_PREFIX"'_'"$USER"'_'"$BATCH_NAME_SUFFIX"
   BATCH_DIRECTORY=~/Downloads/samples/_mturk/$BATCH_NAME
   mkdir -p $BATCH_DIRECTORY
   python -m src.bin.combine_csv \
      --csvs ~/Downloads/samples/*/metadata.csv \
      --shuffle \
      --name "$BATCH_DIRECTORY/metadata.csv"
   cp ~/Downloads/samples/*/*.wav $BATCH_DIRECTORY
   cp ~/Downloads/samples/*/*.log $BATCH_DIRECTORY
   ```

5. Upload your batch to a publicly accessible host, like so:

   ```bash
   gsutil -m cp -r ~/Downloads/samples/_mturk/$BATCH_NAME gs://mturk_samples
   ```

6. Update your local metadata file with the new URL.

   ```bash
   python -m src.bin.csv_add_prefix \
      --csv "$BATCH_DIRECTORY/metadata.csv" \
      --column_name audio_path \
      --prefix "https://storage.googleapis.com/mturk_samples/$BATCH_NAME/" \
      --destination "$BATCH_DIRECTORY/metadata_with_prefix.csv"
   ```

7. Send the `metadata_with_prefix.csv` CSV to Michael until he figures out how to share access to
   our MTurk account. He'll get the batch started. It'll take around 20 minutes per 1000 samples
   for the batch to be completed.

8. Michael will send you a CSV containing the results of the MTurk HITs.
   Use `jupyter notebook` to open
   [`notebooks/QA Datasets/Analyze MTurk MOS Scores.ipynb`](https://github.com/wellsaid-labs/Text-to-Speech/blob/master/notebooks/QA%20Datasets/Analyze%20MTurk%20MOS%20Scores.ipynb)
   and analyze your results!
