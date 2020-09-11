# Library

The `lib` Python package contains WellSaid's TTS low-level functions for working with:

- `datasets`: This module focuses on downloading, and loading datasets.
- `spectrogram_model`: This module defines a sequence to sequence text-to-spectrogram model.
- `audio`: This module contains audio processing functions.
- `distributed`: This module contains utilities for working with `torch.distributed`.
- `environment`: This module contains functions for introspecting and interacting the environment.
- `optimizers`: This module contains wrappers and extensions to `torch.optim`.
- `signal_model`: This module defines a sequence-to-sequence spectrogram-to-signal model.
- `text`: This module contains text processing functions.
- `utils`: This module contains utility functions.
- `visualize`: This module defines functions for visualization.

The various functions are hooked into `HParams` for configuration.

## Architecture

The architecture is based on a Library Oriented Architecture. Each module is a library of functions
constrained to one ontology domain. The functions are modular. The emphasis of separating the
functionality of a program into independent, interchangeable modules, such that each contains
everything necessary to execute only one aspect of the desired functionality.
