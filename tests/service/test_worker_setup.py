from src.service.worker_setup import main
from tests._utils import get_tts_mocks


def test_main():
    """ Smoke test. """
    mocks = get_tts_mocks()
    speaker_encoder = mocks['spectrogram_model_checkpoint'].input_encoder.speaker_encoder
    speaker_id_to_speaker_id = {i: i for i in speaker_encoder.stoi.values()}
    main(speaker_id_to_speaker_id, mocks['spectrogram_model_checkpoint'],
         mocks['signal_model_checkpoint'])
