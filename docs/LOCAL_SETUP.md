# Local Setup

This document summarizes how to setup this repository locally for development.

## 1. Install System Dependencies

Please install the below requirements:

- [Docker](https://www.docker.com/products/docker-desktop)
- [brew](https://brew.sh/)
- [Visual Studio Code](https://code.visualstudio.com/docs/setup/mac). In addition to the
  initial steps, please run through the section called "Launching from the command line".

Afterwards, please install other system dependencies, like so:

```bash
brew install git
brew install python@3.8
brew install sox # Audio processing
brew install ffmpeg # Audio processing
brew install espeak # Speech synthesizer
brew install rsync lsyncd # File transfer
```

## 2. Clone the Repository

Using `git` clone the repository onto your system:

```bash
git clone --depth=10 --no-single-branch --recurse-submodules https://github.com/wellsaid-labs/Text-to-Speech.git
```

The reason for using `--depth=10 --no-single-branch` is to reduce the size of the `.git` directory.
On May 20th, these flags would reduce the repository size by 80%
(i.e. 58 megabytes to 12 megabytes).

## 3. Install Python Dependencies

Install Python dependencies, like so:

```bash
# Start a virtual environment, learn more:
# https://realpython.com/python-virtual-environments-a-primer/
python -m venv venv
. venv/bin/activate

# Install Python dependencies
python -m pip install -r requirements.txt --upgrade
```

## 4. Configure Visualization Dependencies

This repository requires [Comet](https://www.comet.ml) for visualization, and you'll need to ask
a team member to create you an account.

With your new account, you'll need to create a `.comet.config` file in this repositories root
level directory. The file should contain the `api_key`, `rest_api_key` and `workspace`
configurations. Learn more on
[this web page](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration).

## 5. Install Google Cloud SDK

This repository relies on GCP, and you'll need to ask team member to get access to our GCP projects,
"Voice Research" and "Voice Service".

Afterwards, install [Google Cloud SDK](https://cloud.google.com/sdk/) with these
[installation scripts](https://cloud.google.com/sdk/docs/downloads-interactive) and authorize
Google Cloud SDK tools, like so:

```bash
gcloud init

# NOTE: Authorize Google's command-line-interface, learn more:
# https://cloud.google.com/sdk/gcloud/reference/auth/login
gcloud auth login

# NOTE: Authorize Google client libraries, learn more:
# https://cloud.google.com/sdk/gcloud/reference/auth/application-default
gcloud auth application-default login
```

## Good job! 🎉
