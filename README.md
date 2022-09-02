<p align="center"><img width="15%" src="mark.svg" /></p>

<h1 align="center">Text-to-Speech</h3>

Welcome to WellSaid Lab's Text-to-Speech platform. This software allows you to: process and evaluate
data; train and deploy models; manage our infrastructure.

## Install

You'll first need to setup your local development environment. Please
[click here](./docs/LOCAL_SETUP.md) to read how to do so.

## Usage

To use this platform, we have documented these workflows:

- [Creating a new dataset](./docs/CREATE_DATASET.md)
- [Training a spectrogram or signal model](./docs/TRAIN_MODEL_GCP.md)
- [Building our service](./docs/BUILD.md)
- [Deploying our service](./ops/run/README.md)

Additionally, there are a number of CLI and Streamlit tools in the `run` folder. Last but not least,
we've documented some [tips and tricks here](./docs/TIPS_AND_TRICKS.md).

## Contributing

To contribute to this platform, please read through these docs:

- [Software design considerations](./docs/SOFTWARE_DESIGN_CONSIDERATIONS.md)
- [Contribution guidelines](./docs/CONTRIBUTING.md)

## Questions?

Here are the core contributors to this repo :clap: Please feel free to reach out to them if you
have a question.

- **Michael Petrochuk**: Michael created the initial code base and has been responsible for it overall
from design to documentation. Michael has implemented a majority of the data processing and modeling
so far.

- **Rhyan Johnson**: Rhyan is a linguist and data engineer. She has built much of our language
processing including non-English support, respellings pronunciation, and text verbalization. She
has also been involved in creating all our datasets so far. She has been integral to training
and evaluating more models than anyone else.

- **Neil Harlow**: Neil, working with Sam Skjonsberg, has implemented the majority of our
infrastructure. Neil and Rhyan frequently collaborate in order to launch additional voices to our
Studio.

*Please feel free to add your self to this list after you have contributed, so people know who
they can talk to!*
