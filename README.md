# Tacotron 2

Implementation of Google Brain's Tacotron 2. A deep neural network architecture described in this paper: [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)

![PyPI - Python Version](https://img.shields.io/badge/python-3.5%2C%203.6-blue.svg)
[![Build Status](https://travis-ci.com/AI2Incubator/Tacotron-2.svg?token=xKbC739Gn2ssU4AStE7z&branch=master)](https://travis-ci.com/AI2Incubator/Tacotron-2)

## Basics

Make sure you have Python 3.5+.

### Install Dependencies

Install most of the dependencies with the PIP package manager, like so:

    pip3 install -r requirements.txt

Follow the "Get Started" guide on [pytorch.org](pytorch.org) to install ``torch``.

## Research Ideas

* Following [The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation
](https://arxiv.org/abs/1804.09849v2) and [
Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182.pdf) add
embedding dropout.
* Following [The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation
](https://arxiv.org/abs/1804.09849v2) add multi-headed attention.
* Following [Batch Normalization before or after ReLU?](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/)
move the BatchNorm layer following ReLU
* Following [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) consider
incorporating ELMo and tokenizing on words.
* Following [The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation
](https://arxiv.org/abs/1804.09849v2) and [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) add
layer norm to LSTM.
* Similar to ELMo add weights for combining WaveNet layers.
* Try doing a loss function on top of attention following this paper:
https://arxiv.org/pdf/1705.07704.pdf
* https://github.com/NVIDIA/nv-wavenet/issues/5
* Document carefully crafted getting this running guide
* Try to combine the signal model and the feature model into a fully differentiable component
* Try a different loss function for softmax, because large differences matter rather than categories
* Dockerize the requirements for nv-wavenet and resampling
* Try giving a different embedding depending on LJ book or chapter read?
* sudo sh -c "ulimit -n 65536 && exec su $LOGNAME"
* Consider Mean Normalization as suggested here: http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html similar to http://cs231n.github.io/neural-networks-2/ and https://www.google.com/search?q=mean+normalization+spectrogram&oq=mean+normalization+spectrogram&aqs=chrome..69i57.4190j0j1&sourceid=chrome&ie=UTF-8
* INSTALL SOX

## Contributing

Thanks for considering contributing!

### Contributing Guide

Read our [contributing guide](https://github.com/AI2Incubator/Tacotron-2/blob/master/CONTRIBUTING.md) to learn about our development process, how to propose bugfixes and improvements, and how to build and test your changes to Tacotron-2.

## Authors

* Michael Petrochuk - petrochukm@allenai.org
