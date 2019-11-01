# Model Evaluation
WellSaid Labs utilizes an industry standard of Mean Opinion Scores to gauge synthetic audio quality. We use MTurk to crowd source opinion scores of various audio samples and derive a MOS per synthetic voice.

This document outlines the process for creating the MTurk surveys to obtain these scores.

## Background
MTurk utilizes a CSV upload in order to generate survey questions in batches. There are several steps involved in generating the CSV for upload to MTurk and for storing the associated audio files in our Google Clouse Storage bucket.

## 1. [OPTIONAL] Modify the script
Should you be interested in curating that random samples of audio being sent to MTurk for testing, you can modify `evaluate.py` to filter the samples before generating the CSV. For example, perhaps you want to test the quality of one specific Speaker from a model. You can filter your models for that Speaker in main, before running `evaluate.py`.
```python
def main(dataset, ...
  ...
  speakers=[Speaker(name="Hilary Noriega", gender=Gender.FEMALE)],
  ...
  ):
```

## 2. Run script
On your GCP VM instance, navigate to your Text-to-Speach repository & update the machine's Python dependencies, if necessary:
```
cd /opt/wellsaid-labs/Text-to-Speech

# Activate the virtual environment
. venv/bin/activate

# Install Python dependencies
python -m pip install -r requirements.txt --upgrade
```

Run `evaluate.py` with the appropriate arguments.

*ARGUMENT*                 | *DEFINITION/VALUE*
-------------------------- | ------------------------------------------------------------------------------------------------------------------------
**`--signal_model`**       | Provide the relative path to the signal model checkpoint you trained (`experiments/signal_model/[DATE]/...`)
**`--spectrogram_model`**  | Provide the relative path to the spectrogram model checkpoint you trained (`experiments/spectrogram_model/[DATE]/...`)
**`--samples`**            | Number of samples to gather; standard is 220 samples
**`--no-griffin-lim`**     | Use when gathering synthetic audio; Do not use when gathering Ground Truth
**`--no-target-audio`**    | Use when gathering synthetic audio; Do not use when gathering Ground Truth
**`--obscure`**            | Use to obscure the filenames of the wav files so as not to bias the MTurk worker
..                         | ..

Other command line arguments are defined in `evaluate.py`

Example of running `evaluate.py` for gathering **synthetic audio**:
```
rm -r samples/; \
python3 -m src.bin.evaluate \
  --signal_model="experiments/signal_model/[DATE]/checkpoints/[DATE]/step_XXXXXX.pt" \
  --spectrogram_model="experiments/spectrogram_model/[DATE]/checkpoints/[DATE]/step_XXXXXX.pt" \
  --no_target_audio --no_griffin_lim --num_samples=220 --obscure_filename
```

In order to compare MOS results from your experiment against actual audio, you will also want to collect Ground Truth samples. Becuase this doesn't require spectrogram or signal models, it can be run on your local machine.
Example of running `evaluate.py` for gathering **Ground Truth audio**:
```
rm -r samples/; \
python3 -m src.bin.evaluate \
  --num_samples=220 --obscure_filename
```

## 3. Copy results to your local machine
Use `scp` to copy the contents of the `samples/` directory to a well-named location on your local machine:
```
gcloud compute scp --recurse --zone=YOUR_GCP_VM_ZONE YOUR_GCP_USERNAME@YOUR_GCP_VM_NAME:/opt/wellsaid-labs/Text-to-Speech/samples/* /Users/YOU/path/to/experiment/results/samples/
```

If you are running several experiments in parallel, be sure to copy your results to unique locations. You will combine these results in the next step.

## 4. Combine .wav files and upload to GCP Storage Bucket
First, create a directory on your local machine in the Text-to-Speech repo. Name it something useful for team management, beginning with the year and month. For example:
`/Users/me/dev/Text-to-Speech/2019_10_YourFirstName_YourExperimentName_MOS`.  You can see other examples in the **Voice Research** project, **mturk_samples** bucket on [**GCP**](https://console.cloud.google.com/storage/browser/mturk_samples). *Note*: MTurk workers will have the ability to view the url 

Copy all .wav files from your various experiment results locations into this directory.

Navigate to the **Voice Research** project, **mturk_samples** bucket on [**GCP**](https://console.cloud.google.com/storage/browser/mturk_samples). 

Select **Upload folder**. Choose your directory and select **Upload**. This may take a while so leave your machine up and running until all your .wav files have been uploaded.

## 5. Combine .csv files, prepare for MTurk, and send to Michael
Run `combine_csv.py` with arg `--shuffle` to combine all metadata.csv results into one. Again, name your output something useful.
Example:
```
python src/bin/combine_csv.py --csvs \
'/Users/path/to/small-dataset-MOS/Master/samples/metadata.csv' \
'/Users/path/to/small-dataset-MOS/Base/samples/metadata.csv' \
'/Users/path/to/small-dataset-MOS/10H/samples/metadata.csv' \
'/Users/path/to/small-dataset-MOS/8H/samples/metadata.csv' \
'/Users/path/to/small-dataset-MOS/4H/samples/metadata.csv' \
'samples_ground_truth/metadata.csv' \
--name 'samples/DatasetSizeExperiments_Combined_MOS.csv' --shuffle
```

The column order does not matter in this CSV, so long as it contains **audio_url** and **text** columns. You will need to create the **audio_url** column yourself. This column will contain the GCP URL to the uploaded .wav file named in the **audio_path** column. In your project directory in the **mturk_samples** bucket, click on a .wav file to open the **Object Details**. Select the URL at the top of the screen, up to the .wav file name, and copy it. This part of the URL is common to all the uploaded .wav files, so you can use a formula to create the **audio_url** quickly across all rows in the CSV:
Open the combined CSV and create a new column with header **audio_url**. In the first cell, type the formula `="URL_TO_BUCKET_YOU_COPIED"&(SELECT_CELL_IN_AUDIO_PATH_COLUMN). Copy this formula down the entire column so every audio file name is prepended with the GCP URL.
Rename the first column '**index**'.

Save and send to Michael for upload!

Once launched, it takes about 4 hours for MTurk workers to finish a batch.

Michael will send you a CSV containing the results of the MTurk HITs. Use `jupyter notebook` to open [`notebooks/QA Datasets/Analyze MTurk MOS Scores.ipynb`](https://github.com/wellsaid-labs/Text-to-Speech/blob/master/notebooks/QA%20Datasets/Analyze%20MTurk%20MOS%20Scores.ipynb) and analyze your results!
