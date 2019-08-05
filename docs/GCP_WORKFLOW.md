
# Google Cloud Platform (GCP) Workflow

This documentation describes a workflow with GCP GPU machines and MacOS.

## Training A Model

### Prerequisites

Before running this workflow, please follow through on the "Get Started" guide
[here](/README.md).

### 1. Install Google Cloud Platform Tools

Install `gcloud compute` by following the instructions
[here](https://cloud.google.com/compute/docs/gcloud-compute/). Finally, you'll need to ask a team
member to grant you access to our GCP project "WellSaidLabs" with the ID "mythical-runner-203817".

### 2. Install Additional Dependencies

We use `lsyncd` and `rsync` for syncing our source code with a remote machine. Please install these
via:

```bash
brew install rsync
brew install lsyncd
```

### 3. Start Your GCP VM Instance

Our models are trained on a GCP VM instance. You'll want to create a VM instance from a
preconfigured VM instance (i.e. instances with the prefix `base-*`). To do so, you'll need to follow
the instructions
[here](https://cloud.google.com/compute/docs/instances/create-vm-from-similar-instance)
on this
[webpage](https://console.cloud.google.com/compute/instances?project=mythical-runner-203817&instancessize=50).

Then you'll need to SSH into your new VM instance like so:

```bash
gcloud compute ssh your_gcp_instance_name --zone=your_gcp_instance_zone
```

On the VM instance, please take note of your username. You can find out your user name like so:

```bash
echo $USER
```

Note, by default our preconfigured VM(s) are set to be preemtible. This means that sometimes GCP
will not have enough resources for us. In those cases, you may get a none-preemtible machine.

### 4. Synchronize Your Files

To train your model, you'll want to synchronize your local source code to your VM instance. On
your local machine run:

```bash
python -m src.bin.gcp.lsyncd --source $(pwd) \
                              --destination /opt/wellsaid-labs/Text-to-Speech \
                              --user your_gcp_user_name_from_earlier \
                              --instance your_gcp_instance_name
```

You'll want to keep this process running as long as you are changing your local source code.

### 5. Update Your VM Instance Dependancies

Your VM instance will most likely have outdated Python dependancies. Please update them before
running an experiment, like so:

```bash
cd /opt/wellsaid-labs/Text-to-Speech

# Activate the virtual environment
. venv/bin/activate

# Install Python dependencies
python -m pip install -r requirements.txt
```

You may encounter a permissions errors when running the above commands, it can be resolve like
so `sudo chmod -R a+rwx /opt/`.

### 6. Start Training Spectrogram Model

Finally, your ready to start training your model. You can do so, like so:

```bash
# We use screen to keep training going after ending our SSH session, learn more:
# https://askubuntu.com/questions/8653/how-to-keep-processes-running-after-ending-ssh-session
screen

# Activate the virtual environment
. venv/bin/activate

# Kill any existing processes (sometimes `src/bin/train/spectrogram_model/__main__.py` does not
# exit cleanly).
pkill -9 python; nvidia-smi;

PYTHONPATH=. python src/bin/train/spectrogram_model/__main__.py \
    --project_name your_comet_ml_project_name
```

You'll want to train the `signal_model` after you've trained the `spectrogram_model` most likely.
The procedure is similar.

### (Optional) Keeping Your VM Instance Alive

Your machine is most likely a preemtible instance and will die within 24 hours of booting. For
training runs that last longer than 24 hours, you'll want to use the provided script below
to automatically restart your VM instances.

```bash
python -m src.bin.gcp.keep_alive --project_name 'your_comet_ml_project_name' \
          --instance your_gcp_instance_name \
          --command="screen -dm bash -c \
              'cd /opt/wellsaid-labs/Text-to-Speech;
              . venv/bin/activate
              PYTHONPATH=. python src/bin/train/spectrogram_model/__main__.py -c;'"
```

The `--command` flag sets the command that'll be run on your VM instance after it restarts. You can
learn more about the above `screen` command
[here](https://superuser.com/questions/454907/how-to-execute-a-command-in-screen-and-detach).

### (Optional) Download Your Results

Following running experiments, you may want to download files off of a GCP instance, like so:

```bash
gcloud compute scp --recurse --zone=your_gcp_instance_zone \
    your_gcp_instance_name:/path/to/download ~/Downloads/
```

### 7. Delete Your VM Instance

To ensure that experiments are reproducible, please delete your VM instance after you've completed
your experiments and use a new VM instance for any new experiment(s).

## Create Base VM (Advanced)

You may want to create a preconfigured VM instance to cache your VM instance disk contents and
settings.

### 1. GCP VM Instance Settings

First, create a temporary VM instance with these settings:

1. You'll want to set the machine to be preemtible. This will save our company 50% or more on the VM
   instance cost.
1. You'll want approximately 1 - 4x CPU memory as the total GPU memory for optimal performance.
1. You'll want to enable "Allow full access to all Cloud APIs". This will allow your VM instance
   to communicate with other VM instances.

### 2. Setup Your VM Instance

Second, create your VM instance and login. On the VM instance, you'll want to:

1. Install CUDA [here](https://developer.nvidia.com/cuda-toolkit).
1. Create an empty source code directory at `/opt/wellsaid-labs/Text-to-Speech`. Before doing so,
   you'll want to give your self permission to create directories in `/opt/` via
   `sudo chmod -R a+rwx /opt/`.
1. Synchronize your local source code to to your VM following the above instructions.
1. Install any dependencies required to run the source code listed [here](/README.md).
1. Start a new training run to preprocess and cache any required data.

You've now setup an instance for training that can be imaged and replicated.

### 3. Polish And Clean Up

You'll want to polish and clean up the disk before imaging it so that it's useful to your team
members. First, you'll want to remove any personal and temporary files. These files include but are
not limited to your `.git` directory, `.env` settings and the `experiments` directory. Second,
you'll want to run `sudo chmod -R a+rwx /opt/` to ensure all your team members have access to use
your source code. Finally, you'll want to upgrade the system packages like so:

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade
```

### 3. Create An Image

Next, you'll want to image the preconfigured disk. An image can be created on this
[webpage](https://console.cloud.google.com/compute/images?project=mythical-runner-203817&tab=images&imagessize=50)
.

### 4. One Last Step

You'll want to now create a `base` VM within GCP. Create a new VM instance with the above recommend
settings that boots with your image. The new VM instance will start after creation. You can
shut it down because it's only meant to be used as a template.

Finally, you can delete the temporary VM instance we created earlier used to create the image.
