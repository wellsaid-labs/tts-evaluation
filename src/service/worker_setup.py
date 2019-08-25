import logging
import pprint

from src.environment import set_basic_logging_config
from src.service.worker_config import SIGNAL_MODEL_CHECKPOINT_PATH
from src.service.worker_config import SPEAKER_ID_TO_SPEAKER_ID
from src.service.worker_config import SPECTROGRAM_MODEL_CHECKPOINT_PATH
from src.utils import Checkpoint

set_basic_logging_config()
pretty_printer = pprint.PrettyPrinter()
logger = logging.getLogger(__name__)


def main(speaker_id_to_speaker_id, spectrogram_model_checkpoint, signal_model_checkpoint):
    """ Runs various preprocessing code to build the `docker/worker/Dockerfile` file.

    Args:
        speaker_id_to_speaker_id (dict)
        spectrogram_model_checkpoint (src.utils.Checkpoint)
        signal_model_checkpoint (src.utils.signal_model_checkpoint)
    """
    # Ensure `speaker_id_to_speaker_id` invariants pass
    speaker_encoder = spectrogram_model_checkpoint.input_encoder.speaker_encoder
    speaker_ids = list(speaker_encoder.stoi.values())
    assert all(i in speaker_ids for i in speaker_id_to_speaker_id.values()), (
        'All speaker ids were not found in `speaker_encoder`.')
    logger.info('The available speakers are:\n%s', pretty_printer.pformat(speaker_encoder.stoi))

    # Cache ths signal model inferrer
    signal_model_checkpoint.model.to_inferrer()

    # TODO: The below checkpoint attributes should be statically defined somewhere so that there is
    # some guarantee that these attributes exist.

    # Reduce checkpoint size
    signal_model_checkpoint.optimizer = None
    signal_model_checkpoint.anomaly_detector = None
    spectrogram_model_checkpoint.optimizer = None

    # Remove unnecessary information
    signal_model_checkpoint.comet_ml_project_name = None
    signal_model_checkpoint.comet_ml_experiment_key = None
    spectrogram_model_checkpoint.comet_ml_project_name = None
    spectrogram_model_checkpoint.comet_ml_experiment_key = None

    # NOTE: This overwrites the previous checkpoint.
    spectrogram_model_checkpoint.save()
    signal_model_checkpoint.save()


if __name__ == '__main__':
    signal_model_checkpoint = Checkpoint.from_path(SIGNAL_MODEL_CHECKPOINT_PATH)
    spectrogram_model_checkpoint = Checkpoint.from_path(SPECTROGRAM_MODEL_CHECKPOINT_PATH)
    main(SPEAKER_ID_TO_SPEAKER_ID, spectrogram_model_checkpoint, signal_model_checkpoint)
