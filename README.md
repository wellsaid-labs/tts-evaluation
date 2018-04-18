# Tacotron-2

Implementation of Google Brain's Tacotron-2. A deep neural network architecture described in this paper: [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)

## Basics

Make sure you have Python 3.5+.

### Install Dependencies

Install dependencies with the PIP package manager, like so:

    pip3 install -r requirements.txt

### WAV to Spectrogram Image

Convert ``.wav`` files to spectrograms, like so:

    export PYTHONPATH=.
    python3 src/spectrogram.py tests/_test_data/sample_ljspeech.wav

Following this command, you'll find the file ``tests/_test_data/sample_ljspeech_spectrogram.png``

## Contributing

Thanks for considering contributing!

### Contributing Guide

Read our [contributing guide](https://github.com/PetrochukM/Chatty_Speech_Models/blob/master/CONTRIBUTING.md) to learn about our development process, how to propose bugfixes and improvements, and how to build and test your changes to Chatty_Speech_Models.


## Authors

* Michael Petrochuk - petrochukm@allenai.org
