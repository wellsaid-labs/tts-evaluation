import torch
import numpy

from src.audio import get_log_mel_spectrogram
from src.audio import read_audio
from src.audio import write_audio
from src.bin.train.signal_model.data_loader import _get_slice
from src.audio import MultiResolutionMelSpectrogramLoss
from src.hparams import set_hparams
from src.optimizers import AutoOptimizer
from src.visualize import plot_spectrogram
from torch.optim import Adam
from torchnlp.utils import collate_tensors
from src.signal_model.mel_gan import Generator

set_hparams()
signal = read_audio('tests/_test_data/bin/test_chunk_wav_and_text/rate(lj_speech,24000).wav')

model = Generator()

optimizer = AutoOptimizer(Adam(params=filter(lambda p: p.requires_grad, model.parameters())))
criterion = MultiResolutionMelSpectrogramLoss()

log_mel_spectrogram, signal = get_log_mel_spectrogram(signal.astype(numpy.float32))

log_mel_spectrogram = torch.from_numpy(log_mel_spectrogram)
signal = torch.from_numpy(signal)

plot_spectrogram(log_mel_spectrogram).savefig('disk/temp/original_spectrogram.png')
write_audio('disk/temp/original_signal.wav', signal)

step = 0
while True:
    example = collate_tensors([_get_slice(log_mel_spectrogram, signal, 64, 0) for _ in range(8)])
    predicted_signal = model(example.spectrogram, pad_input=False)
    spectral_convergence_loss, log_stft_magnitude_loss = criterion(predicted_signal,
                                                                   example.target_signal)
    optimizer.zero_grad()
    (spectral_convergence_loss + log_stft_magnitude_loss).backward()
    optimizer.step()
    step += 1

    print('step', step)
    print('spectral_convergence_loss', spectral_convergence_loss)
    print('log_stft_magnitude_loss', log_stft_magnitude_loss)
    print('-' * 100)

    if step % 10 == 0:
        write_audio('disk/temp/predicted_signal_%d.wav' % step, predicted_signal[0])
        write_audio('disk/temp/original_signal_%d.wav' % step, example.target_signal[0])
        plot_spectrogram(example.spectrogram[0]).savefig('disk/temp/original_spectrogram_%d.png' %
                                                         step)

        predicted_signal = model(log_mel_spectrogram)
        write_audio('disk/temp/new_full_predicted_signal_%d.wav' % step, predicted_signal)
