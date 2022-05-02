# NOTE: Other `run` modules import `_config`; therefore, they need to be imported first.
from run import _config  # isort: skip
from run import _utils, data, train, utils

# NOTE: `_tts` and `deploy` use every module, so they need to be imported last.
from run import _tts  # isort: skip
from run import deploy  # isort: skip

# NOTE: `_streamlit` is not imported due to side-effects from the import.

__all__ = ["_config", "_utils", "data", "train", "utils", "_tts", "deploy"]
