# Pre-Processing Datasets Example Workflow

This documentation will walk you through preprocessing the data from one of our actors on MacOS.

For more software details, please visit `src/bin/sync_script_with_audio.py` and read the inline
documentation.

## Prerequisites

1. Setup your local development environment by following [these instructions](LOCAL_SETUP.md).

2. Install `gcloud compute` by following the instructions
   [here](https://cloud.google.com/compute/docs/gcloud-compute/)

3. Ask a team member to grant you access to our GCP project called "voice-research".

## TODO

1. How do you verify the metadata in the audio is consistent?
2. How does the actor upload there data to Google Cloud Storage?
3. What is required to create a good dataset? We're not going to cover the entire section here.
4. In this guide, do we need to define machine learning, and how to create a consistent and
    unambiguous task? This guide is likely for beginners to pre-process a dataset; however, it's not
    for a complete beginner. We don't need to revisit

## Data

This data will be used to train our text-to-speech model. As in any other modeling task, the task
must be clearly defined. For example, the data should not include random and unexplained variations
in the voice-over.

You'll find the data for each voice-actor working with WellSaid Labs in the
[wellsaid_labs_datasets](https://console.cloud.google.com/storage/browser/wellsaid_labs_datasets;tab=objects?project=voice-research-255602&prefix=)
Google Storage Bucket. Find the actor whose data needs to be processed and note the directory
structure should look like this:

```bash
└── wellsaid_labs_datasets
    ├── actor_name
    │   ├── scripts
    │   │   ├── DIPHONE_Script-1.csv
    │   │   ├── DIPHONE_Script-2.csv
    │   │   ...
    │   │   ├── ENTHUSIASTIC_Script-1.csv
    │   │   └── ENTHUSIASTIC_Script-2.csv
    │   └── recordings
    │       ├── WSL_ActorShmactor_Script-DIPHONE-1.wav
    │       ├── WSL_ActorShmactor_Script-DIPHONE-2.wav
    │       ...
    │       ├── WSL_ActorShmactor_ENTHUSIASTIC_Script 1.wav
    │       └── WSL_ActorShmactor_ENTHUSIASTIC_Script 2.wav
    └── ....
```

In order to assure data consistency, you'll want to pay attention to these items:

- All files in `/scripts/` must be `.csv` files.
- All files in `/recordings/` must be audio files.
- There must be the same number of script files as there are audio files.
- The script and audio files must be named similarly. More specifically, the script uses
  https://en.wikipedia.org/wiki/Natural_sort_order to pair the script and audio files together.
  Pay attention to the natural sort order of the files in each directory; you may need to do some
  renaming.
- The file size of each audio file stored in `recordings` are _similar_ in size and appropriate for
  the length of the script text.
- The script files are normalized such that any odd characters or phrases that the voice actor
  ignored have been removed or replaced.
- The audio files are self-consistent, for example:
  - They should have similar metadata like format, sample rate, and channels.
  - Aside from changes in the content, the voice-over sounds similar.
  - Unless there is additional context given, the voice-over does not change prosody or language.

## Synchronize Script with Audio

Follow the inline documentation for `src/bin/sync_script_with_audio.py` to synchronize the script
with the audio, and to save a file with the alignment. You'll need to audit the results of the
synchronization, and re-run the script if necessary. The issues that may arise are:

- The dataset length is shorter or longer than expected.
- The voice-over is cutoff, and it does not have enough hours of audio.
- The voice-over skips parts of the script.
- The script has odd characters that are not read by the voice-actor.
- There are a large number of unaligned words.

In order to mitigate these errors, please check: (TODO)

- [ ] That the original audio files have reasonable and consistent file sizes.
- [ ] The script reports less than 0.5% "unaligned characters"; otherwise, the transcript likely
      doesn't match the audio.
- [ ] The dataset created is of a reasonable length. For example, the script should log that
      the final Actor Shmactor dataset is ~14 hours in length.
- [ ] The longest "unaligned substring" is small; otherwise, the transcript likely doesn't match
      the audio.
- [ ] The `/processed/wavs/` files are named consistently; otherwise, the audio formats of the
      original files are not the same. This could mean that the audio was recorded inconsistently.
      For example, some of Actor Shmactor's audio files required a `bits` normalization and others
      required a `rate` normalization.
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
processed correctly. (TODO)

If you find any errors have occurred, please update the transcript or audio files and rerun the
scripts. In the end, we want to have an accurate transcript, audio files, and processed dataset.
(TODO)

## Afterwards

TODO: Add the dataset to the code base
TODO: The dataset will be automatically process during runtime.
