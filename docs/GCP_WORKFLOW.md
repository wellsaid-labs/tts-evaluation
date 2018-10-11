
# GCP Workflow

This documentation describes the a workflow with GCP GPU machines on OSX.

## Prerequisite: install

Install ``gcloud compute`` by following the instructions
[here](https://cloud.google.com/compute/docs/gcloud-compute/).

Install the latest version of `lsyncd` and `rsync` by running:
```
brew install rsync
brew install lsyncd
```

## Prerequisite: install (optional)

To switch python versions, install ``pyenv`` on a GCP machine. First, install
the ``pyenv`` requirements listed [here](https://github.com/pyenv/pyenv/wiki/Common-build-problems).
Finally, install ``pyenv`` with the scripts [here](https://github.com/pyenv/pyenv-installer).

## Synchronize files

Frequently, you'll want to share a files between your local machine and GCP. We allow for that via:

```
python3 -m src.bin.lsyncd --source /path/to/WellSaid-Labs-Text-To-Speech \
                          --destination /path/to/WellSaid-Labs-Text-To-Speech \
                          --user someone --instance a_gcp_instance
```

Remember to kill your ``lsyncd`` process after your done via keyboard interrupt.

## Keep alive instance

Preemtible machines on GCP are contracted to die within 24 hours of booting, we provide a script
to keep alive GCP machines.

Here's an example of using the script:
```
python3 -m src.bin.keep_alive --command="screen -dm bash -c \
        'cd WellSaid-Labs-Text-To-Speech/; \
        ulimit -n 65536; \
        python3 -m src.bin.signal_model.train -c;'"
```

The ``--command`` flag runs a command on restart of the GCP server.

## Synchronize tensorboard

When running multiple experiments, we provide a tool to synchronize multiple tensorboard events
to one GCP instance. Use it like so:

```
# Light GCP instance for running tensorboard
gcloud compute ssh tensorboard --zone=us-us-west1-b

# Find any new servers
gcloud compute config-ssh

python3 -m src.bin.sync_instances --destination ~/WellSaid-Labs-Text-To-Speech/sync/ \
                                  --source ~/WellSaid-Labs-Text-To-Speech/experiments/signal_model
```

Now, GCP instance ``tensorboard`` will periodically pull events from other GCP instances.

## Download

Following running experiments, you may want to download files off of a GCP instance, consider
this method for doing so:

```
gcloud compute scp --recurse --zone=some_zone instance:~/path/to/results ~/Desktop/results
```
