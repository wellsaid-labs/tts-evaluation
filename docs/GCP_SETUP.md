# Setting Up a Virtual Machine on Google Cloud Platform

[TOC]

## ...for Training a Spectrogram Model

1. Go to WellSaid Lab's [Google Cloud Platform](https://console.cloud.google.com/projectselector/compute/instances)
2. Under 'Select or create a project', choose **Voice Research** from the drop-down and click 'Continue'
3. Click **create instance** at the top
4. You'll want to set up the machine with the following configuration:
    * **Name**:  name your machine something descriptive, useful, and pertinent to your experiments
    * **Region**:  us-west1(Oregon)  |  **Zone**: us-west1-b  | . 
      * *Note: You may need to adjust the region and zone to find available resources. Choose a combination that allows for the rest of the outlined configuration below.*
    * **Machine Configuration** | **Machine Type**: select *n1-standard-16 (16 vCPU, 60 GB memory)* from the drop-down
    * Expand '*CPU platform and GPU*'
      * **CPU platform**: Intel Skylake or later
      * **GPUs** | **+Add GPU**:  **GPU type**: NVIDIA Tesla P100  |  **Number of GPUs**: 2
    * **Boot disk**:  click 'Change'
      * Select the *Ubuntu 18.04 LTS* OS image
      * **Boot disk type**: Standard persistent disk  |  **Size (GB)**: 512
      * Click 'Select'
    * **Identity and API access** | **Access scopes**: Allow full access to all Cloud APIs
    * Expand '*Management, security, disks, networking, sole tenancy*'
      * **Availability policy** | **Preemptibility**: On
          
    Click 'Create'. Wait until machine is created. A green checkmark will appear next to the instance name when the creation process is finished and the machine is running.
  
5. From your local repository, ssh into your new VM instance:
```bash
. src/bin/gcp/ssh.sh YOUR-VM-INSTANCE-NAME
```

### On the VM instance...
6. Set up permissions, install packages, & create a wellsaid-labs directory:
```bash
sudo apt-get update
sudo apt-get install python3-venv -y
sudo apt-get install python3-dev -y
sudo apt-get install gcc -y
sudo apt-get install sox -y
sudo apt-get install ffmpeg -y
sudo apt-get install ninja-build -y
```

7. Install [CUDA-10-0](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork) and Nvidia Cuda Toolkit
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

Verify CUDA installed correctly by running `nvidia-smi` with no error messages occuring.

8. Make a wellsaid-labs directory
```bash
cd /opt
mkdir wellsaid-labs
cd wellsaid-labs
```

### From your local repository
9. Follow the instructions in the [ReadMe](https://github.com/wellsaid-labs/Text-to-Speech/blob/master/README.md#4-configure-visualization-dependencies) to setup your Comet configuration.

10. Use lsync to copy your repo to your VM instance:
```bash
python3 -m src.bin.gcp.lsyncd --instance YOUR-VM-INSTANCE-NAME \
                              --source /Users/YOU/path/to/Text-to-Speech \
                              --destination /opt/wellsaid-labs/Text-to-Speech \
                              --user YOUR-VM-USER
```
When prompted, give your local sudo password for your laptop.

### On the VM instance
11. Navigate to the repository, activate a virtual environment, and install package requirements:
```
cd Text-to-Speech

python3 -m venv venv
. venv/bin/activate

python -m pip install wheel
python -m pip install -r requirements.txt --upgrade

sudo bash src/bin/install_mkl.sh -y
```

12. Train a Spectrogram Model
```bash
screen
. venv/bin/activate

pkill -9 python; nvidia-smi; PYTHONPATH=. python src/bin/train/spectrogram_model/__main__.py     --project_name='YOUR-COMET-PROJECT'
```
