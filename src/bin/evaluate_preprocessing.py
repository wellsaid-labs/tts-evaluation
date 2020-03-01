""" This is a script to evaluate our audio preprocessing.

Example:

    python -m src.bin.evaluate_preprocessing --audio_path tests/_test_data/test_audio/hilary.wav

"""
import argparse
import logging
import pathlib
import warnings
import pyloudnorm

import torch

from src.audio import amplitude_to_db
from src.audio import db_to_power
from src.audio import framed_rms_from_power_spectrogram
from src.audio import get_audio_metadata
from src.audio import get_num_seconds
from src.audio import griffin_lim
from src.audio import integer_to_floating_point_pcm
from src.audio import power_to_db
from src.audio import read_audio
from src.audio import rms_from_signal
from src.audio import SignalTodBMelSpectrogram
from src.audio import write_audio
from src.environment import TEMP_PATH
from src.hparams import set_hparams
from src.visualize import plot_mel_spectrogram
from src.visualize import plot_spectrogram

logger = logging.getLogger(__name__)


def main(audio_path):
    """
    Args:
        audio_path (str)
    """
    set_hparams()

    warnings.filterwarnings(
        'ignore', module=r'.*hparams', message=r'.*Overwriting configured argument.*')

    metadata = get_audio_metadata(pathlib.Path(audio_path))
    audio = torch.tensor(integer_to_floating_point_pcm(read_audio(audio_path, metadata)))
    peek_level = audio.abs().max()
    rms_level = rms_from_signal(audio.numpy())
    signal_to_db_mel_spectrogram = SignalTodBMelSpectrogram(sample_rate=metadata.sample_rate)
    # TODO: Following the approach in `test_frame_and_non_frame_equality` we could pad the
    # signal before hand for a more accurate calculation of loudness.
    db_mel_spectrogram, db_spectrogram, spectrogram = signal_to_db_mel_spectrogram(
        audio, intermediate=True)
    weighted_frame_rms_level = framed_rms_from_power_spectrogram(db_to_power(db_mel_spectrogram))
    weighted_rms_level = power_to_db(weighted_frame_rms_level.pow(2).mean()).item()
    meter = pyloudnorm.Meter(metadata.sample_rate, 'DeMan')

    logger.info('Audio File Metadata: %s', metadata)
    logger.info('Audio File Length: %fs', get_num_seconds(audio_path))
    logger.info('Peak Level: %f (%f dBFS)', peek_level, amplitude_to_db(peek_level))
    logger.info('RMS Level: %f (%f dBFS)', rms_level, amplitude_to_db(torch.tensor(rms_level)))
    logger.info('RMS ?-Weighted Level: %f dBFS', weighted_rms_level)
    logger.info('RMS K-Weighted Lavel (ITU-R BS.1770-4): %f dBFS',
                meter.integrated_loudness(audio.numpy()))

    db_mel_spectrogram_path = TEMP_PATH / 'db_mel_spectrogram.png'
    plot_mel_spectrogram(
        db_mel_spectrogram, sample_rate=metadata.sample_rate).savefig(db_mel_spectrogram_path)
    logger.info('Wrote `%s`.' % db_mel_spectrogram_path)

    db_spectrogram_path = TEMP_PATH / 'db_spectrogram.png'
    plot_spectrogram(db_spectrogram, sample_rate=metadata.sample_rate).savefig(db_spectrogram_path)
    logger.info('Wrote `%s`.' % db_spectrogram_path)

    spectrogram_path = TEMP_PATH / 'spectrogram.png'
    plot_spectrogram(spectrogram, sample_rate=metadata.sample_rate).savefig(spectrogram_path)
    logger.info('Wrote `%s`.' % spectrogram_path)

    griffin_lim_path = TEMP_PATH / 'griffin_lim.wav'
    waveform = griffin_lim(
        db_mel_spectrogram.numpy(), sample_rate=metadata.sample_rate, use_tqdm=True)
    write_audio(griffin_lim_path, waveform, sample_rate=metadata.sample_rate, overwrite=True)
    logger.info('Wrote `%s`.' % griffin_lim_path)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=True, help='The audio file to process.')
    args = parser.parse_args()
    main(args.audio_path)
