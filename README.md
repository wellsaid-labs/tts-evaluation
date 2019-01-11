
<p align="center"><img width="15%" src="logo.png" /></p>

<h1 align="center">WellSaid Labs</h3>

WellSaid Lab's core product featuring our TrueVoice deep neural network architecture.

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg)
[![Build Status](https://travis-ci.com/AI2Incubator/WellSaidLabs.svg?token=xKbC739Gn2ssU4AStE7z&branch=master)](https://travis-ci.com/AI2Incubator/WellSaidLabs)

## Installation

This section discusses various dependencies that need to be installed for this repository.

### 1. Install Python Dependencies

Make sure you have Python 3.6.6 with pip. Install most of the dependencies with the PIP package
manager like so:

    python3 -m pip install -r requirements.txt

Finally, follow the "Get Started" guide on [pytorch.org](pytorch.org) to install ``torch``.

### 2. Install SoX

This repository requires [SoX](http://sox.sourceforge.net/) (Sound eXchange) for audio preprocessing
which can be installed, like so:

    apt-get install sox

or

    brew install sox

### 3. Create a Comet account

With your new Comet account create a ``.env`` file with these details:

    COMET_ML_WORKSPACE=blahblah
    COMET_ML_API_KEY=blahblahblah

### 4. Install ``rsync``, ``lsyncd``, and ``gcloud compute`` (Optional)

To work with GCP, we recommend installing the dependencies listed in ``docs/GCP_WORKFLOW.md``.

## Train

This section describes commands to run the executables required for training.

#### Spectrogram Model

Train the spectrogram model like so:

    python -m src.bin.train.spectrogram_model -n experiment_name -p comet_ml_project_name

#### Signal Model

Train the signal model like so:

    python -m src.bin.train.signal_model -n experiment_name -p comet_ml_project_name

## Test

Run unit tests with code coverage like so:

    python -m pytest tests/ src/ --doctest-modules --cov=src --cov-report html:coverage --cov-report=term-missing

Run linting like so:

    flake8 src/; flake8 test/;

## Contributing

Thanks for considering contributing!

### Contributing Guide

Read our
[contributing guide](https://github.com/AI2Incubator/WellSaidLabs/blob/master/docs/CONTRIBUTING.md)
to learn about our development process, how to propose bug-fixes and improvements, and how to build
and test your changes to WellSaidLabs.

## Authors

* Michael Petrochuk - michaelp@allenai.org
