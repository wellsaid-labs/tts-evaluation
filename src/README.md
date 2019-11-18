# Source

The `src` Python package contains WellSaid's TTS source code. In addition to the various
sub-packages each with a `README.md`, this contains these modules:

- `audio`: This module contains audio processing functions.
- `distributed`: This module contains utilities for working with `torch.distributed`.
- `environment`: This module contains utilities and constants related to global state like disk
  allocation and logging.
- `hparams`: This module contains global hyperparameter configuration.
- `optimizers`: This module contains wrappers and extensions to `torch.optim`.
- `visualize`: This module contains utilities related with visualization.
