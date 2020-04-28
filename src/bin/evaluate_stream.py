# TODO: Add examples and documentation.
# TODO: Rename to `evaluate_single.py`
import pathlib
import subprocess

import torch
import unidecode

from src import datasets
from src.environment import set_basic_logging_config
from src.environment import SIGNAL_MODEL_EXPERIMENTS_PATH
from src.environment import SPECTROGRAM_MODEL_EXPERIMENTS_PATH
from src.hparams import set_hparams
from src.signal_model import generate_waveform
from src.spectrogram_model import SpectrogramModel
from src.utils import Checkpoint

# TODO: Save a log along with teh sample
torch.set_grad_enabled(False)
set_basic_logging_config()
set_hparams()

# TODO: Make this configurable
text = pathlib.Path('book.txt').read_text()

# TODO: Make this configurable
speaker = datasets.SAM_SCHOLL

# TODO: Make this configurable
spectrogram_model_checkpoint_path = SPECTROGRAM_MODEL_EXPERIMENTS_PATH / 'step_261253.pt'
signal_model_checkpoint_path = SIGNAL_MODEL_EXPERIMENTS_PATH / 'step_335371.pt'
device = torch.device('cpu')

spectrogram_model_checkpoint = Checkpoint.from_path(
    spectrogram_model_checkpoint_path, device=device)
signal_model_checkpoint = Checkpoint.from_path(signal_model_checkpoint_path, device=device)

# TODO: Remove these lines they are for backwards compatibility, in the next release.
num_tokens = spectrogram_model_checkpoint.input_encoder.text_encoder.vocab_size
num_speakers = spectrogram_model_checkpoint.input_encoder.speaker_encoder.vocab_size
state_dict = spectrogram_model_checkpoint.model.state_dict()
spectrogram_model_checkpoint.model = SpectrogramModel(num_tokens, num_speakers)
spectrogram_model_checkpoint.model.load_state_dict(state_dict)

spectrogram_model = spectrogram_model_checkpoint.model.eval()
input_encoder = spectrogram_model_checkpoint.input_encoder
signal_model = signal_model_checkpoint.model.eval()

# TODO: Save attention pictures and spectrogram images along with the text.
# TODO: Save the phonetic spelling.
text = unidecode.unidecode(text)
input_encoder.text_encoder.enforce_reversible = False
preprocessed_text = input_encoder.preprocess_text(text)
processed_text = input_encoder.text_encoder.decode(
    input_encoder.text_encoder.encode(preprocessed_text))
if processed_text != preprocessed_text:
    improper_characters = set(preprocessed_text).difference(set(input_encoder.text_encoder.vocab))
    improper_characters = ', '.join(sorted(list(improper_characters)))
    raise ValueError('Text cannot contain these characters: %s' % improper_characters)

signal_model_checkpoint.exponential_moving_parameter_average.apply_shadow()
for module in signal_model.get_weight_norm_modules():
    torch.nn.utils.remove_weight_norm(module)

# TODO: Use Logger
print('Number of characters: %d' % len(text))

text, speaker = input_encoder.encode((text, speaker))

print('Number of tokens: %d' % len(text))


# TODO: Use TQDM
def get_spectrogram():
    for _, frames, _, _, _ in spectrogram_model(text, speaker, is_generator=True):
        # [num_frames, batch_size (optional), frame_channels] →
        # [batch_size (optional), num_frames, frame_channels]
        yield frames.transpose(0, 1) if frames.dim() == 3 else frames


# TODO: Save samples in `disk/samples/`
# TODO: Write an MP3 file because it can streamed
command = ('ffmpeg -y -f f32le -acodec pcm_f32le -ar 24000 -ac 1 -i pipe: book.wav').split()
pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
for waveform in generate_waveform(signal_model, get_spectrogram()):
    pipe.stdin.write(waveform.cpu().detach().numpy().tobytes())
pipe.stdin.close()
pipe.wait()
