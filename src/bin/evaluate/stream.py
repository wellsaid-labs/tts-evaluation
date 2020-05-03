"""
Generate a stream of audio from a script.

Example:

    $ python -m src.bin.evaluate.stream --signal_model experiments/your/checkpoint.pt \
                                        --spectrogram_model experiments/your/checkpoint.pt \
                                        --script_file_path script.txt \
                                        --speaker 'Frank Bonacquisti'
"""
import argparse
import logging
import pathlib
import subprocess

from hparams import configurable
from hparams import HParam
from hparams import log_config

import torch

# NOTE: Some modules log on import; therefore, we first setup logging.
from src.environment import set_basic_logging_config

set_basic_logging_config()

from src import datasets
from src.environment import SAMPLES_PATH
from src.environment import set_basic_logging_config
from src.hparams import set_hparams
from src.signal_model import generate_waveform
from src.utils import bash_time_label
from src.utils import Checkpoint
from src.utils import RecordStandardStreams

set_basic_logging_config()

logger = logging.getLogger(__name__)


@configurable
def main(file_path,
         speaker,
         signal_model_checkpoint,
         spectrogram_model_checkpoint,
         sample_rate=HParam(),
         destination=SAMPLES_PATH / bash_time_label(),
         stream_filename='stream.mp3',
         preprocessed_text_filename='preprocessed_text.txt',
         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Args:
        file_path (str): The script file path.
        speaker (Speaker): The voice over speaker.
        signal_model_checkpoint (str or None): Checkpoint used to predict a raw waveform
            given a spectrogram.
        spectrogram_model_checkpoint (str or None): Checkpoint used to generate spectrogram
            from text as input to the signal model.
        get_sample_rate (callable, optional): Get the number of samples in a clip per second.
        destination (str, optional): Path to store results.
        stream_filename (str, optional): The stream file name.
        preprocessed_text_filename (str, optional): The file name to store the preprocessed text.
        device (torch.device, optioanl): The device to run inference on.
    """
    destination = pathlib.Path(destination)
    destination.mkdir(exist_ok=False)

    RecordStandardStreams(destination).start()

    log_config()

    text = pathlib.Path(file_path).read_text()

    spectrogram_model = spectrogram_model_checkpoint.model.eval()
    input_encoder = spectrogram_model_checkpoint.input_encoder
    signal_model_checkpoint.exponential_moving_parameter_average.apply_shadow()
    signal_model = signal_model_checkpoint.model.eval()

    logger.info('Number of characters: %d', len(text))
    (destination / preprocessed_text_filename).write_text(input_encoder._preprocess(text))
    text, speaker = input_encoder.encode((text, speaker))
    logger.info('Number of tokens: %d', len(text))

    def get_spectrogram():
        for item in spectrogram_model(text, speaker, is_generator=True, use_tqdm=True):
            # [num_frames, batch_size (optional), frame_channels] →
            # [batch_size (optional), num_frames, frame_channels]
            yield item[1].transpose(0, 1) if item[1].dim() == 3 else item[1]

    with torch.no_grad():
        command = ('ffmpeg -y -f f32le -acodec pcm_f32le -ar %d -ac 1 -i pipe: -b:a 192k %s' %
                   (sample_rate, destination / stream_filename)).split()
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
        for waveform in generate_waveform(signal_model, get_spectrogram()):
            pipe.stdin.write(waveform.cpu().detach().numpy().tobytes())
        pipe.stdin.close()
        pipe.wait()


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_model', type=str, help='Signal model checkpoint to use.')
    parser.add_argument(
        '--spectrogram_model', type=str, help='Spectrogram model checkpoint to use.')
    parser.add_argument('--script_file_path', type=str, help='The script file path.')
    parser.add_argument('--speaker', type=str, help='The voice over speaker.')
    args = parser.parse_args()

    # NOTE: Load early and crash early by ensuring that the checkpoint exists and is not corrupt.
    args.signal_model = Checkpoint.from_path(args.signal_model)
    args.spectrogram_model = Checkpoint.from_path(args.spectrogram_model)
    args.speaker = getattr(datasets, args.speaker.upper().replace(' ', '_'))

    set_hparams()

    main(
        file_path=args.script_file_path,
        speaker=args.speaker,
        signal_model_checkpoint=args.signal_model,
        spectrogram_model_checkpoint=args.spectrogram_model)
