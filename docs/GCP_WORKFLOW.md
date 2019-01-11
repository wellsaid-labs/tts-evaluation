
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

To switch python versions, install ``pyenv`` on a GCP machine. First, install the ``pyenv``
requirements listed [here](https://github.com/pyenv/pyenv/wiki/Common-build-problems). Finally,
install ``pyenv`` with the scripts [here](https://github.com/pyenv/pyenv-installer).

Note that distributed PyTorch is only verified to work on Python 3.6.6.

## Synchronize files

Frequently, you'll want to share a files between your local machine and GCP. We allow for that via:

```
python3 -m src.bin.gcp.lsyncd --source /path/to/WellSaidLabs \
                          --destination /path/to/WellSaidLabs \
                          --user someone --instance a_gcp_instance
```

Remember to kill your ``lsyncd`` process after your done via keyboard interrupt.

## Keep alive instance

Preemtible machines on GCP are contracted to die within 24 hours of booting, we provide a script
to keep alive GCP machines.

Here's an example of using the script:
```
python3 -m src.bin.gcp.keep_alive --command="screen -dm bash -c \
        'cd WellSaidLabs/; \
        ulimit -n 65536; \
        python3 -m src.bin.train.signal_model -c;'"
```

The ``--command`` flag runs a command on restart of the GCP server.

## Download

Following running experiments, you may want to download files off of a GCP instance, consider
this method for doing so:

```
gcloud compute scp --recurse --zone=some_zone instance:~/path/to/results ~/Desktop/results
```
