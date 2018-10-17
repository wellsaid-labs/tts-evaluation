
<p align="center"><img width="15%" src="logo.png" /></p>

<h1 align="center">WellSaid Labs: Text-to-speech</h3>

WellSaid's implementation of Google Brain's Tacotron 2 and DeepMind's WaveRNN. A deep neural network
architecture described in these papers:
[Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf) and [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435).

![PyPI - Python Version](https://img.shields.io/badge/python-3.5%2C%203.6-blue.svg)
[![Build Status](https://travis-ci.com/AI2Incubator/WellSaid-Labs-Text-To-Speech.svg?token=xKbC739Gn2ssU4AStE7z&branch=master)](https://travis-ci.com/AI2Incubator/WellSaid-Labs-Text-To-Speech)

## Installation

This section discusses various dependencies that need to be installed for this repository.

### 1. Install Python Dependencies

Make sure you have Python 3.5+ with pip. Install most of the dependencies with the PIP package
manager like so:

    python3 -m pip install -r requirements.txt

Finally, follow the "Get Started" guide on [pytorch.org](pytorch.org) to install ``torch``.

### 2. Install SoX

This repository requires [SoX](http://sox.sourceforge.net/) (Sound eXchange) for audio preprocessing
which can be installed, like so:

    apt-get install sox

or

    brew install sox

### 3. Install ``rsync``, ``lsyncd``, and ``gcloud compute`` (Optional)

To work with GCP, we recommend installing the dependencies listed in ``docs/GCP_WORKFLOW.md``.

## Train

This section describes commands to run the executables required for training.

#### Feature Model

First things first, preprocess the audio via:

    python -m src.bin.train.feature_model.preprocess

Train the feature model like so:

    python3 -m src.bin.train.feature_model -n experiment_name

Generate training data for the signal model like so:

    python3 -m src.bin.train.feature_model.generate -c your_checkpoint

#### Signal Model

Train the signal model like so:

    python3 -m src.bin.train.signal_model -n experiment_name

## Run Text-to-Speech

To run system end-to-end first launch Jupyter:

    jupyter notebook

Then, the ``notebooks/Synthesize Speech from Text.ipynb`` notebook runs the system from end-to-end.

## Test

Run unit tests with code coverage like so:

    python -m pytest tests/ src/ --doctest-modules --cov=src --cov-report html:coverage --cov-report=term-missing

Run linting like so:

    flake8 src/; flake8 test/;

## Contributing

Thanks for considering contributing!

### Contributing Guide

Read our
[contributing guide](https://github.com/AI2Incubator/WellSaid-Labs-Text-To-Speech/blob/master/docs/CONTRIBUTING.md)
to learn about our development process, how to propose bug-fixes and improvements, and how to build
and test your changes to WellSaid-Labs-Text-To-Speech.

## Authors

* Michael Petrochuk - petrochukm@allenai.org
