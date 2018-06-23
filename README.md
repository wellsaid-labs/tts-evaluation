
<p align="center"><img width="35%" src="logo.png" /></p>

<h1 align="center">WellSaid: Tacotron 2</h3>

WellSaid's implementation of Google Brain's Tacotron 2. A deep neural network architecture described in this paper: [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf)

![PyPI - Python Version](https://img.shields.io/badge/python-3.5%2C%203.6-blue.svg)
[![Build Status](https://travis-ci.com/AI2Incubator/Tacotron-2.svg?token=xKbC739Gn2ssU4AStE7z&branch=master)](https://travis-ci.com/AI2Incubator/Tacotron-2)

## Installation

This section discusses various dependencies that need to be installed for this repository.

### 1. Clone

Pull the GitHub repository with submodules like so:

    git clone --recurse-submodules git@github.com:AI2Incubator/Tacotron-2.git

This may not work, if so try other strategies listed [here](https://stackoverflow.com/questions/3796927/how-to-git-clone-including-submodules).

### 2. Install Python Dependencies

Make sure you have Python 3.5+ with pip. Install most of the dependencies with the PIP package
manager like so:

    python -m pip install -r requirements.txt

Finally, follow the "Get Started" guide on [pytorch.org](pytorch.org) to install ``torch``.

### 3. Install SoX

This repository requires [SoX](http://sox.sourceforge.net/) (Sound eXchange) for audio preprocessing which can be installed,
like so:

    apt-get install sox

or

    brew install sox

### 4. Install NV WaveNet

This repository depends on NVIDIA's WaveNet kernel. Please follow the instructions [here](https://github.com/AI2Incubator/nv-wavenet/tree/master/pytorch) to set up the kernel.

## Train

This section describes commands to run the executables required for training.

#### Feature Model

First things first, preprocess the audio via:

    python src/bin/feature_model/preprocess.py

Train the feature model like so:

    python src/bin/feature_model/train.py

Launch a Tensorboard instance like so:

    tensorboard --logdir=experiments/feature_model

Note that for both executable there are various options accessible with the ``--help`` flag.

#### Signal Model

Before training the signal model, generate training data using the feature model like so:

    python src/bin/signal_model/generate.py

Train the signal model like so:

    python src/bin/signal_model/train.py

Launch a Tensorboard instance like so:

    tensorboard --logdir=experiments/signal_model

Note that for both executable there are various options accessible with the ``--help`` flag.

## Text to Speech

To run system end-to-end first launch Jupyter:

    jupyter notebook

Then, the ``notebooks/Synthesize Speech from Text.ipynb`` notebook runs the system from end-to-end.

## Test

Run unit tests with code coverage like so:

    python -m pytest tests/ --cov=src --cov-report html:coverage --cov-report=term-missing

Run linting like so:

    flake8 src/; flake8 test/;

## Contributing

Thanks for considering contributing!

### Contributing Guide

Read our [contributing guide](https://github.com/AI2Incubator/Tacotron-2/blob/master/docs/CONTRIBUTING.md) to learn about our development process, how to propose bugfixes and improvements, and how to build and test your changes to Tacotron-2.

## Authors

* Michael Petrochuk - petrochukm@allenai.org
