# Local Setup

This document summarizes how to setup this repository locally for development.

## Prerequisites

This repository only supports macOS with Python 3.6 or higher. You'll also need
[brew](https://brew.sh/), a package manager for macOS.

## 1. Download The Repository

Using `git` clone the repository onto your system:

```bash
git clone --depth=1 --no-single-branch https://github.com/wellsaid-labs/Text-to-Speech.git
```

The reason for using `--depth=1 --no-single-branch` is to reduce the size of the `.git` directory.
On May 20th, these flags would reduce the repository size by 80%
(i.e. 58 megabytes to 12 megabytes).

## 2. Install Python Dependencies

Install Python dependencies, like so:

```bash
# Start a virtual environment, learn more:
# https://realpython.com/python-virtual-environments-a-primer/
python -m venv venv
. venv/bin/activate

# Install Python dependencies
python -m pip install -r requirements.txt --upgrade
```

## 3. Install Additional Dependencies

This repository requires [SoX](http://sox.sourceforge.net/) (Sound eXchange) and
[FFmpeg](https://ffmpeg.org/) for audio preprocessing. They can be installed like so:

```bash
brew install sox
brew install ffmpeg
```

Note that we do not use `ffmpeg` directly instead it is used by one of our
[dependencies](https://librosa.github.io/librosa/install.html#ffmpeg).

This repository also requires [Ninja](https://ninja-build.org/) for compilation. It can be
installed like so:

```bash
brew install ninja
```

### (Optional) [MKL](https://software.intel.com/en-us/mkl)

For optimal signal model inference performance, install [MKL](https://software.intel.com/en-us/mkl)
like so:

```bash
sudo bash src/bin/install_mkl.sh
```

Note that this installation will only work on Linux machines with an Intel CPU.

## 4. Configure Visualization Dependencies

This repository requires [Comet](https://www.comet.ml) for visualization. You'll need to ask
a team member to create you an account.

With your new account, you'll need to create a `.comet.config` file in this repositories root
level directory. The file should contain the `api_key`, `rest_api_key` and `workspace`
configurations. Learn more on
[this web page](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration).

Note that this software tends to trigger Comet's throttling. We are in good standing
with the Comet team; therefore, if you need you can ask for a second API key to ensure their
system does not throttle you.

Note that if you need to file an issue with Comet please visit
[this webpage](https://github.com/comet-ml/issue-tracking).

## You're Done

Verify that your installation was successful by running the test suite:

```bash
pytest
```