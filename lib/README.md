# Library

The `lib` Python package contains WellSaid's TTS functional core:

- `datasets`: This module focuses on downloading, and loading datasets.
- `spectrogram_model`: This module defines a sequence to sequence text-to-spectrogram model.
- `audio`: This module contains audio processing functions.
- `distributed`: This module contains utilities for working with `torch.distributed`.
- `environment`: This module contains functions for introspecting and interacting the environment.
- `optimizers`: This module contains extensions to `torch.optim`.
- `signal_model`: This module defines a sequence-to-sequence spectrogram-to-signal model.
- `text`: This module contains text processing functions.
- `utils`: This module contains utility functions.
- `visualize`: This module defines functions for visualization.

The various functions are hooked into `hparams` for configuration.
