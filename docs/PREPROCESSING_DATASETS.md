# Pre-Processing Datasets Example Workflow

This documentation will walk you through preprocessing the data from one of our actors on MacOS.

For more software details, please visit `src/bin/chunk_wav_and_text.py` and read the inline
documentation.

## Prerequisites

The data for each voice actor who has contributed to WellSaid Labs is organized in the WSL-Team Google Drive under **[01 Voice/Datasets](https://drive.google.com/drive/u/1/folders/1_QGAJqcklVHrY4Pjyg5405_n-QauhM_z)**. Find the actor whose data needs to be processed and note the directory structure should look like this:

├── Datasets
│   ├── ...
│   │   ├── Actor Name
│   │   │   ├── 00 Audio Samples  - contains Actor audition, tone preworking clips
│   │   │   ├── 01 Agreements     - contains Actor contracts + agreements with WSL
│   │   │   ├── 02 Scripts        - contains the scripts in .docx for Actor use
│   │   │   ├── 03 Recordings     - contains the audio in .wav uploaded by the Actor
│   │   │   ├── 04 Scripts (CSV)  - contains the scripts in .csv for WSL processing

You're going to want to download the contents of **03 Recordings** and **04 Scripts (CSV)**, using a structure outlined in the following example. Note: Check the file size of each wav stored in 03 Recordings and verify they are appropriate size (a good check is that all files in the directory are *similar* in size and appropriate for the length of the script text.) Due the size of these files, the 03 Recordings directory will most likely download as *multiple zip files*. Don't forget to unzip *each* and move the contents into the structure outlined!

Suppose that the actor we're working with is called "Jack Rutkowski". Please download the actor's corresponding CSV and WAV files into a directory like so:

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
│   │   
│   ├── /processed/
```

Pay close attention to the following:
- All files in /csvs/ must be `.csv` files
- All files in /wavs/ must be `.wav` files
- There must be the same number of CSV files as there are WAV files
- The CSV and WAV files must be named similarly. More specifically, the script uses
  https://en.wikipedia.org/wiki/Natural_sort_order to pair the CSV and WAV files together.
  Pay attention to the natural sort order of the files in each directory; you may need to do some renaming.
- The directory can live anywhere on your system, so long as it follows this structure.

Create a **processed** directory to store the output from running `chunk_wav_and_text.py`. It's easiest if it lives here, but can also live anywhere on your system.


## Pre-Processing
Follow the inline documentation for `src/bin/chunk_wav_and_text.py` to pre-process the audio data!

The output of the script will be stored in the /processed/ directory like this:

```bash
.
├── /processed/
│   ├── /*.sst*/ 
│   │   ├── bits(rate(WSL_JackRutkowski_ENTHUSIASTIC_Script 22,24000),16).json
│   │   ├── bits(rate(WSL_JackRutkowski_ENTHUSIASTIC_Script 23,24000),16).json
│   │   ...
│   │   ├── rate(WSL_JackRutkowski_ENTHUSIASTIC_Script 59,24000).json
│   │   ├── rate(WSL_JackRutkowski_ENTHUSIASTIC_Script-60,24000).json
│   │   ...
│   ├── /wavs/
│   │   ├── bits(rate(WSL_JackRutkowski_ENTHUSIASTIC_Script-22,24000),16)
│   │   │   ├── script_0_chunk_0.wav
│   │   │   ├── script_0_chunk_1.wav
│   │   │   ├── ...
│   │   │   ├── script_19_chunk_4.wav
│   │   ├── bits(rate(WSL_JackRutkowski_ENTHUSIASTIC_Script-23,24000),16)
│   │   ...
│   │   ├── rate(WSL_JackRutkowski_ENTHUSIASTIC_Script-59,24000)
│   │   ├── rate(WSL_JackRutkowski_ENTHUSIASTIC_Script-60,24000)
└── └── ...
│   ├── metadata.csv
│   ├── stderr####.log
│   ├── stdout####.log
```
 * The *.sst* directory is HIDDEN, but contains the cached results from Google Speech-to-Text
 * The /wavs/ directory contains subdirectories for each original wav file; each subdirectory contains the original wav sliced into smaller wav files
 * *metadata.csv* is a large CSV file that pairs the original text to the chunked wav file and its file path
 * *stderr.log* is a log of errors that were thrown while running the script
 * *stdout.log* is a log of the output sent to std_out while running the script.
 
Should the pre-processing run into any errors OR if the output is bad and you need to re-run `src/bin/chunk_wav_and_text.py`, you will need to delete everything but the *`.sst`* directory. Running Google Speech-to-Text is the most time-consuming portion of the `chunk_wav_and_text.py` script, so caching the results is important for future runs.

Some things to be aware of when running `chunk_wav_and_text.py`.

- Checklist:
- [ ] Verify the audio files in megabytes
- [ ] < 1% unaligned characters | The transcript was accurate
- [ ] ~14 hours of audio | There was enough audio data
- [ ] Longest unaligned sequence < 500 characters | The transcript was accurate
- [ ] Audio files have same format | The audio recording was consistent
- [ ] Metadata has the same number of lines as wav files | Sanity check
- [ ] Listened to 25 random samples with `notebooks/QA Datasets/Sample Dataset.ipynb` | Sanity check
- [ ] There is only two logs files | Sanity check
- [ ] Logs do not reveal missing or skipped script
- [ ] There are no weird characters in the script | The script has been cleaned
- [ ] Add double check “unaligned text spans between SST and transcript” to the dataset list. This includes text that maybe wasn’t spoken but got included anyways. The reason that it may have been included is that we try to save it. I should probably adjust the script to handle this.

- Update the transcript if you notice any significant errors

## Afterwards
Zip up the metadata.csv, log files, and wavs directory into a TAR.
```bash
tar -czvf ActorName.tar.gz metadata.csv stderr.#####.log stdout.#####.log wavs
```

Navigate to the actor's folder on the WSL Google Drive and create a new folder called **(05) Processed**.
Upload the TAR there.


## Note

- You can bulk rename on MacOS
