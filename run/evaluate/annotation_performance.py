import pathlib
import time

import en_core_web_sm
import pyloudnorm
import torch
import torchaudio

from lib import datasets
from lib.audio import (
    SignalTodBMelSpectrogram,
    get_audio_metadata,
    read_audio,
    to_floating_point_pcm,
)
from lib.hparams import set_hparams
from lib.text import grapheme_to_phoneme
from lib.utils import Checkpoint, get_functions_with_disk_cache, seconds_to_string

set_hparams()

for function in get_functions_with_disk_cache():
    function.use_disk_cache(False)

nlp = en_core_web_sm.load()

audio_path = "tests/_test_data/test_audio/heather.wav"
text = (
    "According to Corner, Kinichi, and Keats, strategic decision making in organizations "
    "occurs at two levels: individual and aggregate. They developed a model of parallel "
    "strategic decision making."
)

# Get word vectors
start = time.time()
doc = nlp(text)
assert all([sum(token.tensor) != 0 for token in doc])
print("`nlp` ran for: %s" % seconds_to_string(time.time() - start))  # 36ms (Easy to cache)

# Get phonemes
start = time.time()
phonemes = [grapheme_to_phoneme(token.text) for token in doc]
print("Phonemes:", phonemes)
print(
    "`grapheme_to_phoneme` ran for: %s" % seconds_to_string(time.time() - start)
)  # 522ms (Easy to cache)

# Load audio
metadata = get_audio_metadata(pathlib.Path(audio_path))
audio = torch.tensor(to_floating_point_pcm(read_audio(audio_path, metadata)))

# Get spectrogram
signal_to_db_mel_spectrogram = SignalTodBMelSpectrogram(sample_rate=metadata.sample_rate)
start = time.time()
db_mel_spectrogram, db_spectrogram, spectrogram = signal_to_db_mel_spectrogram(
    audio, intermediate=True
)
print("`SignalTodBMelSpectrogram` ran for: %s" % seconds_to_string(time.time() - start))  # 38ms

# Get loudness
start = time.time()
meter = pyloudnorm.Meter(metadata.sample_rate, "DeMan")
print("%f dBFS" % meter.integrated_loudness(audio.numpy()))
print("`integrated_loudness` ran for: %s" % seconds_to_string(time.time() - start))  # 17ms

# Get pitch
start = time.time()
print("audio", audio.shape)
pitch = torchaudio.functional.detect_pitch_frequency(audio, metadata.sample_rate)
print("pitch", len(pitch))
print(
    "`detect_pitch_frequency` ran for: %s" % seconds_to_string(time.time() - start)
)  # 671ms (Problematic, we might not include this...)

spectrogram_model_checkpoint = Checkpoint.from_path(
    "disk/experiments/spectrogram_model/step_649010.pt", torch.device("cpu")
)
SPECTROGRAM_MODEL = spectrogram_model_checkpoint.model.eval()
INPUT_ENCODER = spectrogram_model_checkpoint.input_encoder
text, speaker = INPUT_ENCODER.encode(
    (text, datasets.HEATHER_DOE),
)
start = time.time()
frames, stop_token, alignments = SPECTROGRAM_MODEL(
    text,
    speaker,
    target_frames=db_mel_spectrogram,
    target_lengths=torch.LongTensor([db_mel_spectrogram.shape[0]]),
)
print("Predicted Frames:", frames.shape)
print(
    "`SPECTROGRAM_MODEL` ran for: %s" % seconds_to_string(time.time() - start)
)  # 1s 520ms - 2s 366ms (Problematic)
