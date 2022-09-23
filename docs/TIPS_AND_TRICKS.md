# Tips & Tricks

## Finding the latest checkpoint

You can quickly find the latest checkpoint by running the below command. It prints all the files
and directories in order of creation date.

```bash
find disk/experiments/spectrogram_model/ -printf '%T+ %p\n' | sort -r | head
```

Learn more about the above script
[here](https://askubuntu.com/questions/61179/find-the-latest-file-by-modified-date).

## Recovering deleted files

You can recover any deleted files tracked by git by running the below command:

```bash
git status
git ls-files -d | sed -e "s/\(.*\)/'\1'/" | xargs git checkout --
git status
```

Learn more about the above script
[here](https://stackoverflow.com/questions/11956710/git-recover-deleted-file-where-no-commit-was-made-after-the-delete).

## Deleting the TTS cache

During training this program will cache a number of items to disk. You can delete these various
caches.

- Delete the cache'd dataset preprocessing, like so:

  ```bash
  find disk/data -name '.tts_cache' -type d -exec rm -rf {} \;
  ```

## Find the largest directories

You may find it useful to sort directories by their size, in order to find the largest
directories. If so try this command:

```bash
du --human-readable  | sort --human-numeric-sort
```

## Scroll inside a screen session

It's useful to be able to scroll through your logs in a screen session, you can do so by following
the instructions [here](https://unix.stackexchange.com/a/40243).

## Port Forwarding

You may find it useful to run an application, like `streamlit` remotely while accessing it locally.
Here is how you would go about doing that:

```zsh
VM_NAME="name-of-remote-machine"
VM_ZONE="zone-of-remote-machine"
PROJECT_ID=voice-research-255602
LOCAL_PORT=2222
REMOTE_PORT=8501

gcloud compute ssh $VM_NAME \
    --project $PROJECT_ID \
    --zone $VM_ZONE \
    -- -NL $LOCAL_PORT":localhost:"$REMOTE_PORT
```
