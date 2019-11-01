# Preprocessing Datasets Example Workflow

This documentation will walk you through preprocessing the data from one of our actors called
Jack Rutkowski on MacOS.

For more software details, please visit `src/bin/chunk_wav_and_text.py` and read the inline
documentation.

## Prerequisites

Jack Rutkowski's data can be found on Google Drive.

Suppose that our the actor we'd like to preprocess is called "Jack Rutkowski". Please download
the actors corresponding CSV and WAV files into a directory like so:

```bash
.
├── /jack_rutkowski/
│   ├── /wavs/
│   │   ├──WSL_JackRutkowski_Script-DIPHONE-1.wav
│   │   ├──WSL_JackRutkowski_Script-DIPHONE-2.wav
│   │   ...
│   │   ├──WSL_JackRutkowski_ENTHUSIASTIC_Script 1.wav
│   │   ├──WSL_JackRutkowski_ENTHUSIASTIC_Script 2.wav
│   │   ...
│   ├── /csvs/
│   │   ├──DIPHONE_Script-1.csv
│   │   ├──DIPHONE_Script-2.csv
│   │   ...
│   │   ├──ENTHUSIASTIC_Script-1.csv
│   │   ├──ENTHUSIASTIC_Script-2.csv
└── └── ...
```

- The dataset must be a `.csv` file
- The dataset must be a `.wav` file
- There must be the same number of CSV files as there are WAV files.
- The CSV and WAV files must be named similarly. More specifically, the script uses
  https://en.wikipedia.org/wiki/Natural_sort_order to pair up the CSV and WAV files.
- The directory can be placed anywhere on your system.
- A potential directory structure is like so:

## Preprocessing

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
