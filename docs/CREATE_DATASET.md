# Getting Started Creating a Text-to-Speech Dataset

This will be walking you through the process of creating a text-to-speech dataset.

## Prerequisites

Setup your local development environment by following [these instructions](LOCAL_SETUP.md).

## 1. Upload scripts and recordings

First, you'll need to upload scripts and recordings to
[wellsaid_labs_datasets](https://console.cloud.google.com/storage/browser/wellsaid_labs_datasets;tab=objects?project=voice-research-255602)
. The uploaded directory should look similar to this example...

```bash
└── wellsaid_labs_datasets
    ├── hilary_noriega
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

The script files must be in a CSV format that contains a column called "Content" filled in with
the script the voice actor read. The audio files can be in any audio format that's compatible
with `ffmpeg`. Lastly, there should be one script per audio file, similarly named.

Aside from that, for this dataset to be useful, it must be consistent. The TTS will be a
reflection of the dataset. For example, here are a couple of consistencies you should
watch out for:

- Aside from changes in the content, the voice-over pitch, timbre (tone color) and loudness
  should stay consistent.
- Unless there is additional context given, the voice-over should not change prosody or language.
- The voice-over noise level shouldn't change.
- The environment in which the voice-over is recorded in, should not change.
- The script should be an accurate transcription of the recordings, and it should not contain any
  funky characters.

## 2. Process data

### Make a virtual machine (VM)

In order to process the scripts and recordings, you'll need to make a virtual machine.

1. Set these variables...

   ```zsh
   VM_NAME=$USER"-your-instance-name" # EXAMPLE: michaelp-dataset-processing
   # NOTE: Pick a zone that's closest to the GCS bucket `wellsaid_labs_datasets`.
   VM_ZONE=your-vm-instance-zone # EXAMPLE: us-central1-a

   PROJECT=voice-research-255602
   VM_MACHINE_TYPE=n1-standard-2
   gcloud config set project $PROJECT
   gcloud auth application-default set-quota-project $PROJECT
   ```

1. Create a VM, like so...

   ```zsh
   gcloud compute instances create $VM_NAME \
      --zone=$VM_ZONE \
      --machine-type=$VM_MACHINE_TYPE \
      --boot-disk-size=512GB \
      --boot-disk-type=pd-ssd \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --image-family=ubuntu-2004-lts \
      --image-project=ubuntu-os-cloud
   ```

1. From your local machine, `ssh` into your new VM instance, like so...

   ```zsh
   gcloud compute ssh --zone=$VM_ZONE $VM_NAME --command="sudo chmod -R a+rwx /opt"
   gcloud compute ssh --zone=$VM_ZONE $VM_NAME --command="mkdir /opt/wellsaid-labs"
   gcloud compute ssh --zone=$VM_ZONE $VM_NAME
   ```

   These commands may exit with the return code 255, if so, try again.

1. In another terminal window, run `lsyncd` to sync your local files to your virtual machine...

   ```zsh
   VM_NAME=$(python -m run.utils.gcp_vm most-recent --filter $USER)
   VM_ZONE=$(python -m run.utils.gcp_vm zone --name $VM_NAME)
   VM_IP=$(python -m run.utils.gcp_vm ip --name $VM_NAME --zone=$VM_ZONE)
   VM_USER=$(python -m run.utils.gcp_vm user --name $VM_NAME --zone=$VM_ZONE)
   sudo python3 -m run.utils.lsyncd $(pwd) /opt/wellsaid-labs/Text-to-Speech \
                                    --public-dns $VM_IP \
                                    --user $VM_USER \
                                    --identity-file ~/.ssh/google_compute_engine
   ```

### Download your data onto the VM

1. Set these variables...

   ```bash
   NAME=actor_name # Example: hilary_noriega
   ROOT=/opt/wellsaid-labs/Text-to-Speech/disk/data/$NAME
   PROCESSED=$ROOT/processed
   ENCODING=.wav
   ```

1. Download the dataset, like so...

   ```bash
   GCS_URI=gs://wellsaid_labs_datasets/$NAME
   mkdir -p $ROOT
   gsutil -m cp -r -n $GCS_URI/scripts $ROOT/
   gsutil -m cp -r -n $GCS_URI/recordings $ROOT/
   ```

### Process data

1. Install these dependencies onto the VM, like so...

   ```bash
   cd /opt/wellsaid-labs/Text-to-Speech
   sudo apt-get update
   sudo apt-get install python3-venv python3-dev sox ffmpeg espeak gcc libsox-fmt-mp3 -y

   python3 -m venv venv
   . venv/bin/activate
   python -m pip install wheel pip --upgrade
   python -m pip install -r requirements.txt --upgrade
   ```

1. (Optional) Ensure the directories have similar numberings...

   ```bash
   python -m run.data numberings $ROOT/scripts $ROOT/recordings
   ```

1. Normalize file names...

   ```bash
   python -m run.data rename $ROOT/
   ```

1. (Optional) Ensure the recording and script pairings make sense...

   ```bash
   RECORDINGS=$(ls $ROOT/recordings/*$ENCODING | python -m run.utils.sort)
   SCRIPTS=$(ls $ROOT/scripts/*.csv | python -m run.utils.sort)
   python -m run.data pair $(python -m run.utils.prefix --recording $RECORDINGS) \
      $(python -m run.utils.prefix --script $SCRIPTS)
   ```

1. (Optional) Review dataset audio file metadata for inconsistencies...

   ```bash
   python -m run.data audio metadata $ROOT/recordings/*$ENCODING
   ```

1. Normalize audio file format...

   ```bash
   mkdir -p $PROCESSED/recordings
   python -m run.data audio normalize $ROOT/recordings/*$ENCODING $PROCESSED/recordings
   ```

1. Normalize audio file format for
   [Google speech-to-text](https://cloud.google.com/speech-to-text/docs/encoding)...

   ```bash
   mkdir -p $PROCESSED/speech_to_text
   python -m run.data audio normalize $ROOT/recordings/*$ENCODING $PROCESSED/speech_to_text \
                                      --encoding='pcm_s16le'
   ```

1. (Optional) Review audio file loudness for inconsistencies...

   ```bash
   python -m run.data audio loudness $PROCESSED/recordings/*.wav
   ```

1. Normalize CSV file text...

   ```bash
   mkdir -p $PROCESSED/scripts
   python -m run.data csv normalize $ROOT/scripts/*.csv $PROCESSED/scripts
   ```

1. Upload the processed files back to GCS, like so...

   ```bash
   gsutil -m cp -r -n $PROCESSED/ $GCS_URI/processed
   ```

1. (Optional) From your local machine, review CSV normalization, like so...

   ```zsh
   NAME=actor_name # Example: hilary_noriega
   GCS_URI=gs://wellsaid_labs_datasets/$NAME
   python -m run.data diff "$GCS_URI/scripts/Script 1 - Hilary.csv" \
                           "$GCS_URI/processed/scripts/script_1_-_hilary.csv"
   ```

   💡 TIP: This script can be run to compare log files generated by `sync_script_with_audio`.

1. Generate time alignments that synchronize the scripts and recordings...

   ```bash
   RECORDINGS=$(gsutil ls "$GCS_URI/processed/speech_to_text/*.wav" | python -m run.utils.sort)
   SCRIPTS=$(gsutil ls "$GCS_URI/processed/scripts/*.csv" | python -m run.utils.sort)
   python -m run.data.sync_script_with_audio \
      $(python -m run.utils.prefix --voice-over $RECORDINGS) \
      $(python -m run.utils.prefix --script $SCRIPTS) \
      --destination $GCS_URI/processed/
   ```

   Audit the results of the synchronization, and re-run the script if necessary. The issues that may
   arise are:

    - The sorting didn't work, and the script files didn't match up with the voice-over files,
      correctly.
    - The voice-over includes too much, or too little audio.
    - The voice-over skips phrases in the script. For example, the script has odd characters or
      duplicate phrases that are not read by the voice-actor.
    - Google’s speech recognition made a mistake.

    Most of these issues can be resolved by updating the script or recording, and rerunning the
    synchronization.

## 3. Review dataset

TODO

## 4. Clean up

1. Kill your `lsyncd` process by typing `Ctrl-C`.

1. Exit your VM with the `exit` command.

1. Delete your instance...

   ```zsh
   gcloud compute instances delete $VM_NAME --zone=$VM_ZONE
   ```
