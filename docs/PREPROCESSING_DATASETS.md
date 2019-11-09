# Pre-Processing Datasets Example Workflow

This documentation will walk you through preprocessing the data from one of our actors on MacOS.

For more software details, please visit `src/bin/chunk_wav_and_text.py` and read the inline
documentation.

## Prerequisites

1. Setup your local development environment by following [these instructions](LOCAL_SETUP.md).

2. Install `gcloud compute` by following the instructions
   [here](https://cloud.google.com/compute/docs/gcloud-compute/)

3. Ask a team member to grant you access to our GCP project called "voice-research".

## Download

The data for each voice actor who has contributed to WellSaid Labs is organized in the WSL-Team
Google Drive under
**[01 Voice/Datasets](https://drive.google.com/drive/u/1/folders/1_QGAJqcklVHrY4Pjyg5405_n-QauhM_z)**.
Find the actor whose data needs to be processed and note the directory structure should look like
this:

```bash
├── Datasets
│ ├── ...
│ │ ├── Actor Name
│ │ │ ├── 00 Audio Samples - contains Actor audition, tone preworking clips
│ │ │ ├── 01 Agreements - contains Actor contracts + agreements with WSL
│ │ │ ├── 02 Scripts - contains the scripts in .docx for Actor use
│ │ │ ├── 03 Recordings - contains the audio in .wav uploaded by the Actor
│ │ │ ├── 04 Scripts (CSV) - contains the scripts in .csv for WSL processing
```

You're going to want to download the contents of **03 Recordings** and **04 Scripts (CSV)**, using a
structure outlined in the following example.

Note: Check the file size of each wav stored in `03 Recordings` and verify they are appropriate
size (a good check is that all files in the directory are _similar_ in size and appropriate for the
length of the script text.) Due the size of these files, the `03 Recordings` directory will most
likely download as _multiple zip files_. Don't forget to unzip _each_ and move the contents into the
structure outlined!

Suppose that the actor we're working with is called "Jack Rutkowski". Please download the actor's
corresponding CSV and WAV files into a directory like so:

```bash
.
├── /jack_rutkowski/
│   ├── /wavs/
│   │   ├── WSL_JackRutkowski_Script-DIPHONE-1.wav
│   │   ├── WSL_JackRutkowski_Script-DIPHONE-2.wav
│   │   ...
│   │   ├── WSL_JackRutkowski_ENTHUSIASTIC_Script 1.wav
│   │   ├── WSL_JackRutkowski_ENTHUSIASTIC_Script 2.wav
│   │   ...
│   ├── /csvs/
│   │   ├── DIPHONE_Script-1.csv
│   │   ├── DIPHONE_Script-2.csv
│   │   ...
│   │   ├── ENTHUSIASTIC_Script-1.csv
│   │   ├── ENTHUSIASTIC_Script-2.csv
└── └── ...
│   ├── /processed/
```

Pay close attention to the following:

- All files in `/csvs/` must be `.csv` files
- All files in `/wavs/` must be `.wav` files
- There must be the same number of CSV files as there are WAV files
- The CSV and WAV files must be named similarly. More specifically, the script uses
  https://en.wikipedia.org/wiki/Natural_sort_order to pair the CSV and WAV files together.
  Pay attention to the natural sort order of the files in each directory; you may need to do some
  renaming.
- The directory can live anywhere on your system, so long as it follows this structure.

Create a **processed** directory to store the output from running `chunk_wav_and_text.py`. It's
easiest if it lives here, but can also live anywhere on your system.

## Pre-Processing

Follow the inline documentation for `src/bin/chunk_wav_and_text.py` to pre-process the audio data!

The output of the script will be stored in the `/processed/` directory like this:

```bash
.
├── /processed/
│   ├── /.sst/
│   │   ├── bits(rate(WSL_JackRutkowski_ENTHUSIASTIC_Script 22,24000),16).json
│   │   ├── bits(rate(WSL_JackRutkowski_ENTHUSIASTIC_Script 23,24000),16).json
│   │   ...
│   │   ├── rate(WSL_JackRutkowski_ENTHUSIASTIC_Script 59,24000).json
│   │   ├── rate(WSL_JackRutkowski_ENTHUSIASTIC_Script-60,24000).json
│   │   ...
│   ├── /wavs/
│   │   ├── /bits(rate(WSL_JackRutkowski_ENTHUSIASTIC_Script-22,24000),16)/
│   │   │   ├── script_0_chunk_0.wav
│   │   │   ├── script_0_chunk_1.wav
│   │   │   ├── ...
│   │   │   ├── script_19_chunk_4.wav
│   │   ├── /bits(rate(WSL_JackRutkowski_ENTHUSIASTIC_Script-23,24000),16)/
│   │   ...
│   │   ├── /rate(WSL_JackRutkowski_ENTHUSIASTIC_Script-59,24000)/
│   │   ├── /rate(WSL_JackRutkowski_ENTHUSIASTIC_Script-60,24000)/
└── └── ...
│   ├── metadata.csv
│   ├── stderr*.log
│   ├── stdout*.log
```

- The `.sst` directory is HIDDEN, but contains the cached results from Google Speech-to-Text.
- The `/wavs/` directory contains subdirectories for each original wav file; each subdirectory
  contains the original wav sliced into smaller wav files.
- `metadata.csv` is a large CSV file that pairs the original text to the chunked wav file and
  its file path.
- `stderr*.log` is a log of errors that were thrown while running the script.
- `stdout*.log` is a log of the written standard output while running the script.

Should the pre-processing run into any errors OR if the output is bad and you need to re-run
`src/bin/chunk_wav_and_text.py`, you will need to delete everything but the `.sst` directory.
Running Google Speech-to-Text is the most time-consuming portion of the `chunk_wav_and_text.py`
script, so caching the results is important for future runs.

## Quality Assurance

The script tends to print a lot of logs to help you validate that the process ran smoothly. The
errors that may come up are:

- The transcript doesn't match the spoken words in the audio.
- There are not enough hours of audio.
- The audio files were not created consistently.
- The audio skips some parts of the transcript.
- The transcript has odd characters.
- The script crashed and was restarted in a compromised state.
- The script made incorrect assumptions about the alignment of the audio and transcript.

In order to mitigate these errors, please check:

- [ ] That the original audio files have reasonable and consistent file sizes.
- [ ] The script reports less than 0.5% "unaligned characters"; otherwise, the transcript likely
      doesn't match the audio.
- [ ] The dataset created is of a reasonable length. For example, the script should log that
      the final Jack dataset is ~14 hours in length.
- [ ] The longest "unaligned substring" is small; otherwise, the transcript likely doesn't match
      the audio.
- [ ] The `/processed/wavs/` files are named consistently; otherwise, the audio formats of the
      original files are not the same. This could mean that the audio was recorded inconsistently.
      For example, some of Jacks audio files required a `bits` normalization and others required
      a `rate` normalization.
- [ ] The metadata has the same number of entries as the number of audio files; otherwise, the
      script may have been run in a compromised state.
- [ ] There are only two logs files; otherwise, the script may have been run in a compromised state.
- [ ] "Unable to align" logs for any red text, they represents text that wasn't included in the
      final dataset. It is expected for a word or two to be not included; however, its not
      expected for a long phrase not to be included unless the transcript or audio has mistakes.
- [ ] "unaligned text spans between SST and transcript" logs for any red text, they represent
      text that was included in the final dataset but maybe wasn't spoken in the audio. It is
      expected for a word or two to be included despite not being detected in the audio; however,
      its not expected for a long phrase to be included but not detected at all.

Finally, please run this notebook as a final check: `notebooks/QA Datasets/Sample Dataset.ipynb`.
The notebook will enable you to check for any weird characters in the finished dataset. Also, it
will allow you to listen to a random sample of audio to ensure that the final dataset was
processed correctly.

If you find any errors have occurred, please update the transcript or audio files and rerun the
scripts. In the end, we want to have an accurate transcript, audio files, and processed dataset.

## Afterwards

Zip up the `metadata.csv`, log files, and wavs directory into a TAR, like so:

```bash
ACTOR_NAME='JackRutkowski'
tar -czvf "$ACTOR_NAME.tar.gz" metadata.csv *.log wavs
```

Navigate to the actor's folder on the WSL Google Drive and create a new folder called
**(05) Processed**. Upload the TAR there.

## Note

- You can bulk rename on MacOS.
