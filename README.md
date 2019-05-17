
<p align="center"><img width="15%" src="logo.png" /></p>

<h1 align="center">WellSaid Labs</h3>

WellSaid Lab's TrueVoice deep neural network architecture.

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg)
[![Build Status](https://travis-ci.com/AI2Incubator/WellSaidLabs.svg?token=xKbC739Gn2ssU4AStE7z&branch=master)](https://travis-ci.com/AI2Incubator/WellSaidLabs)

## Installation

This section discusses various dependencies that need to be installed for this repository.

### 1. Install Python Dependencies

Make sure you have Python 3.7.3 with PIP. Install most of the dependencies with the PIP package
manager like so:

    python3 -m pip install -r requirements.txt

Finally, follow the "Get Started" guide on [pytorch.org](pytorch.org) to install ``torch``.

### 2. Install SoX and FFmpeg

This repository requires [SoX](http://sox.sourceforge.net/) (Sound eXchange) and
[FFmpeg](https://ffmpeg.org/) for audio preprocessing which can be installed, like so:

    apt-get install sox
    apt-get install ffmpeg

or

    brew install sox
    brew install ffmpeg

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


## Repository Layout

```
.
├── /.vscode/                           # Repository VS Code configuration.
├── /data/                              # Default location for caching repository datasets.
├── /docker/                            # Dockerfiles specified for this repository.
                                        # TODO: The Dockerfile are only used in the context of
                                        # the ``service``, consider moving them there.
├── /docs/                              # Additional repository documentation.
├── /experiments/                       # Default location for storing repository experiment data.
├── /node_modules/                      # Default location for storing npm packages.
├── /notebooks/                         # Experimental and unmaintained jupyter notebook scratch work.
├── /src/                               # Repository source code.
│   ├── /bin/                           # Executable scripts.
│   │   ├── /gcp/                       # Executable GCP utility scripts.
│   │   ├── /train/                     # Executable model training scripts.
│   │   ├── /chunk_wav_and_text.py      # Executable to initially create a text-to-speech dataset.
│   │   ├── /evaluate.py                # Executable to evaluate model performance.
│   ├── /datasets/                      # Dataset hooks, schema definitions, and processing
                                        # modules.
│   │   ├── /constants.py               # Dataset schema definitions.
│   │   ├── /hilary.py                  # Hilary dataset.
│   │   ├── /lj_speech.py               # Linda Johnson dataset.
│   │   ├── /m_ailabs.py                # M-AI Labs dataset.
│   │   └── /process.py                 # Dataset example processing modules.
│   ├── /hparams/                       # Global optional parameter configuration.
                                        # TODO: Consider setting 'experiments/' and 'data/' folder
                                        # via hparams.
                                        # TODO: Consider renaming to 'defaults'.
│   │   ├── /configurable_.py           # Module for applying configuration.
│   │   └── /configure.py               # Global optional parameter configuration.
│   ├── /service/                       # Service deployment with Kubernetes.
│   ├── /signal_model/
│   ├── /spectrogram_model/
│   ├── /audio.py                       # Audio processing modules.
│   ├── /distributed.py                 # PyTorch distributed training utilities.
│   ├── /optimizers.py                  # Gradient descent optimizers definitions.
│   ├── /training_context_manager.py    # Training environment context manager.
│   ├── /utils.py                       # Other utilities.
│   └── /visualize.py
├── /tests/                             # Unit test definitions executed by PyTest.
├── /third_party/                       # Third party modules not handled by a package manager.
└── /travis/                            # Continuous integration testing scripts.
```

## Contributing

Thanks for considering contributing!

### Contributing Guide

Read our
[contributing guide](https://github.com/AI2Incubator/WellSaidLabs/blob/master/docs/CONTRIBUTING.md)
to learn about our development process, how to propose bug-fixes and improvements, and how to build
and test your changes to WellSaidLabs.

### Other Documentation

Additionally, further documentation can be found in the
[docs](https://github.com/AI2Incubator/WellSaidLabs/blob/master/docs/) directory.

## Authors

* Michael Petrochuk - michael@wellsaidlabs.com
