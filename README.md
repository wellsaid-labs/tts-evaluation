<p align="center"><img width="15%" src="mark.svg" /></p>

<h1 align="center">Text-to-Speech</h3>

WellSaid Lab's TrueVoice deep neural network architecture.

![PyPI - Python Version](https://img.shields.io/badge/python-3.8-blue.svg)

## Get Started

You might be interested in:

- [Learning about our engineering processes](./docs/ENGINEERING_PROCESSES.md)
- [Setting up a local development environment](./docs/LOCAL_SETUP.md)
- [Training a spectrogram or signal model on GCP](./docs/TRAIN_MODEL_GCP.md)
- [Creating a new dataset](./docs/CREATE_DATASET.md)
- [See some tips and tricks](./docs/TIPS_AND_TRICKS.md)
- [Learn how to build our service locally](./docs/BUILD.md)
- [Learn about our deployments & infrastructure](./ops/README.md)

## Architecture: Functional Core, Imperative Shell

Our library (`lib`) is the functional core of our application. Our services are
defined in `run` as part of the imperative shell. The imperative shell interacts
with the the external world.

Learn more:

- https://www.destroyallsoftware.com/talks/boundaries
- https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell
- https://www.javiercasas.com/articles/functional-programming-patterns-functional-core-imperative-shell
- https://github.com/kbilsted/Functional-core-imperative-shell/blob/master/README.md
- https://en.wikipedia.org/wiki/Pure_function
