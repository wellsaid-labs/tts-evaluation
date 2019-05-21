
# GCP Workflow

This documentation describes a workflow with GCP GPU machines on OSX.

# Run an experiment

## 1. Install

Install ``gcloud compute`` by following the instructions
[here](https://cloud.google.com/compute/docs/gcloud-compute/).

Install the latest version of `lsyncd` and `rsync` by running:
```
brew install rsync
brew install lsyncd
```

## 2. Start GCP VM instance

Our models are trained on a GCP VM instance. A GCP VM instance can be created on this
[webpage](https://console.cloud.google.com/compute/instances?project=mythical-runner-203817&instancessize=50)
. There are available preconfigured "base" VM instances named similar to
"base-v1-spectrogram-model". Create a similar VM instance using the "base" VM instance by following
the instructions
[here](https://cloud.google.com/compute/docs/instances/create-vm-from-similar-instance).

## 3. Synchronize files

Before training, you will need to sync files between your local machine and GCP. That is supported
via:

```bash
python3 -m src.bin.gcp.lsyncd --source /path/to/WellSaidLabs \
                              --destination /path/to/WellSaidLabs \
                              --user someone \
                              --instance a_gcp_instance
```

Remember to kill your ``lsyncd`` process after your done via keyboard interrupt.

## 4. Run experiment

Follow any remaining steps listed [here](https://github.com/AI2Incubator/WellSaidLabs) to start
your experiment.

## (Optional) Keep alive instance

Preemtible machines on GCP are contracted to die within 24 hours of booting, we provide a script
to reboot GCP machines.

For example:

```bash
python3 -m src.bin.gcp.keep_alive --command="screen -dm bash -c \
        'cd ../WellSaidLabs/; python3 -m src.bin.train.signal_model -c;'"
```

The ``--command`` flag runs a command on restart of the GCP server.

## (Optional) Download experiment data

Following running experiments, you may want to download files off of a GCP instance, consider
this method for doing so:

```
gcloud compute scp --recurse --zone=some_zone instance:~/path/to/results ~/Desktop/results
```

## 5. Delete VM instance

Please delete your VM instance after the experiment and use a new VM instance for new experiment(s).

While running your experiment(s) your VM instance will accumulate various state. For example,
for your experiment you may require you to install package(s) onto your VM. Now, if you run a new
experiment on that VM, it may be affected by the installed package for the last experiment. Without
proper documentation of the VM state your new experiment may not be reproducible. In order to avoid
the above scenario, we recommend that you always run experiment(s) on new VM instances.

# Create "base" VM

A "base" VM has a preconfigured disk and settings allowing for running experiments without
reconfiguring for each experiment.

## 1. Settings

Create a new temporary VM with these recommended settings:

1. Set the machine to be preemtible. A preemtible machine is 50%+ cheaper than a none-preemtible
machine; however, a preemtible machine is contracted to die within 24 hours of booting. We use
checkpointing to deal with VM death.

2. Choose to have 2 - 4 times the CPU memory as GPU memory. Note that this guideline has not been
thoroughly vetted.

3. Choose the most recent CPU architecture for the best performance.

4. Enable "Allow full access to all Cloud APIs", allowing for communication across GCP VMs.

## 2. Setup VM

Following creating the VM and logging in:

1. Install CUDA via: https://developer.nvidia.com/cuda-toolkit

1. Synchronize your local files with the VM by following the above instructions.

1. Install any other requirements listed on https://github.com/AI2Incubator/WellSaidLabs

1. Preprocess and cache on disk any required data by starting a new training run.

## 3. Image

Now that the VM disk has been configured to run experiments, we recommend creating a GCP image of
the VM disk. The GCP image is useful for configuring the disk of new VM instances. An image can be
created on this
[webpage](https://console.cloud.google.com/compute/images?project=mythical-runner-203817&tab=images&imagessize=50)
.

To ensure the image is useable by teammates, before creating the image, clean up any personal
files from the "base" VM. For example, delete the ``.env`` and ``.git`` directories. Finally,
move all files outside of your user directory (i.e. ``/home/michaelp``) and change their permissions
to be accessible by everyone (i.e. ``sudo chmod -R a+rwx /your/directory``).

Note GCP has a similar feature called snapshots that also images your VM disk. Following a short
investigation, starting a VM with a snapshot and image both take 3 minutes; however, starting
a secondary VM with a image takes 1 minute while with a snapshot it takes 3 minutes. For that
reason, we use images instead of snapshots.

## 4. Create "base" VM

Finally, create a "base" VM within GCP with the above recommended settings and using the above
image. The new "base" VM is now preconfigured to be duplicated for new experiments without
reconfiguration.
