# Tacotron 2

Implementation of Google Brain's Tacotron 2. A deep neural network architecture described in this paper: [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)

![PyPI - Python Version](https://img.shields.io/badge/python-3.5%2C%203.6-blue.svg?style=flat-square)
[![Build Status](https://travis-ci.com/AI2Incubator/Tacotron-2.svg?token=xKbC739Gn2ssU4AStE7z&branch=master)](https://travis-ci.com/AI2Incubator/Tacotron-2)

## Basics

Make sure you have Python 3.5+.

### Install Dependencies

Install most of the dependencies with the PIP package manager, like so:

    pip3 install -r requirements.txt

Follow the "Get Started" guide on [pytorch.org](pytorch.org) to install ``torch``.

### WAV to Spectrogram Image

Convert ``.wav`` files to spectrograms, like so:

    export PYTHONPATH=.
    python3 src/spectrogram.py tests/_test_data/lj_speech.wav

Following this command, you'll find the file ``tests/_test_data/lj_speech_spectrogram.png``

## Contributing

Thanks for considering contributing!

### Contributing Guide

Read our [contributing guide](https://github.com/PetrochukM/Chatty_Speech_Models/blob/master/CONTRIBUTING.md) to learn about our development process, how to propose bugfixes and improvements, and how to build and test your changes to Chatty_Speech_Models.


## Authors

* Michael Petrochuk - petrochukm@allenai.org
