<p align="center"><img width="15%" src="logo.png" /></p>

<h1 align="center">Text-to-Speech</h3>

WellSaid Lab's TrueVoice deep neural network architecture.

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg)
[![Build Status](https://travis-ci.com/wellsaid-labs/Text-to-Speech.svg?token=xKbC739Gn2ssU4AStE7z&branch=master)](https://travis-ci.com/wellsaid-labs/Text-to-Speech)

## Get Started

### Prerequisites

This repository only supports Python 3.6 or higher.

### 1. Download The Repository

Using `git` clone the repository onto your system:

```bash
git clone --depth=1 --no-single-branch https://github.com/wellsaid-labs/Text-to-Speech.git
```

The reason for using `--depth=1 --no-single-branch` is to reduce the size of the `.git` directory.
On May 20th, these flags would reduce the repository size by 80%
(i.e. 58 megabytes to 12 megabytes).

### 2. Install Python Dependencies

Install Python dependencies, like so:

```bash
# Start a virtual environment, learn more:
# https://realpython.com/python-virtual-environments-a-primer/
python -m venv venv
. venv/bin/activate

# Install Python dependencies
python -m pip install -r requirements.txt --upgrade
```

### 3. Install Additional Dependencies

This repository requires [SoX](http://sox.sourceforge.net/) (Sound eXchange) and
[FFmpeg](https://ffmpeg.org/) for audio preprocessing. They can be installed like so:

```bash
apt-get install sox
apt-get install ffmpeg
```

or

```bash
brew install sox
brew install ffmpeg
```

This repository also requires [Ninja](https://ninja-build.org/) for compilation. It can be
installed like so:

```bash
apt-get install ninja-build
```

or

```bash
brew install ninja
```

#### (Optional) [MKL](https://software.intel.com/en-us/mkl)

For optimal signal model inference performance, install [MKL](https://software.intel.com/en-us/mkl)
like so:

```bash
sudo bash src/bin/install_mkl.sh
```

Note that this installation will only work on Linux machines with an Intel CPU.

### 4. Configure Visualization Dependencies

This repository requires [Comet](https://www.comet.ml) for visualization. You'll need to ask
a team member to create you an account.

With your new account, you'll need to create a `.comet.config` file in this repositories root
level directory. The file should contain the `api_key`, `rest_api_key` and `workspace`
configurations. Learn more on
[this web page](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration).

Note that this software tends to trigger Comet's throttling. We are in good standing
with the Comet team; therefore, if you need you can ask for a second API key to ensure their
system does not throttle you.

### You're Done

Verify that your installation was successful by running the test suite:

```bash
python -m pytest
```

### Other Documentation

Additionally, further documentation (such as how to train a model) can be found [here](docs/).
