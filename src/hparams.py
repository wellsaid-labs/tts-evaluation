from torch import nn

import torch
import librosa
import IPython

from src.utils.configurable import add_config
from src.utils.configurable import configurable


def set_hparams():
    """ Using the ``configurable`` module set the hyperparameters for the source code.
    """

    torch.optim.Adam.__init__ = configurable(torch.optim.Adam.__init__)
    nn.modules.batchnorm._BatchNorm.__init__ = configurable(
        nn.modules.batchnorm._BatchNorm.__init__)
    add_config({
        # NOTE: `momentum=0.01` to match Tensorflow defaults
        'torch.nn.modules.batchnorm._BatchNorm.__init__': {
            'momentum': 0.01,
        },
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999
        'torch.optim.adam.Adam.__init__': {
            'betas': (0.9, 0.999),
            'amsgrad': True,
            'lr': 10**-3
        }
    })

    # SOURCE (Tacotron 2):
    # The convolutional layers in the network are regularized using dropout [25] with probability
    # 0.5, and LSTM layers are regularized using zoneout [26] with probability 0.1
    convolution_dropout = 0.5
    lstm_dropout = 0.1

    # Hidden size of the feature representation generated by encoder.
    encoder_hidden_size = 512

    # SOURCE (Tacotron 2):
    # 80 channel mel filterbank spanning
    frame_channels = 80

    # SOURCE (Tacotron 2):
    # The prediction from the previous time step is first passed through a small
    # pre-net containing 2 fully connected layers of 256 hidden ReLU units.
    pre_net_hidden_size = 256

    # SOURCE (Tacotron 2):
    # Attention probabilities are computed after projecting inputs and location
    # features to 128-dimensional hidden representations.
    attention_hidden_size = 128

    # SOURCE (Tacotron 1):
    # We use 24 kHz sampling rate for all experiments.
    sample_rate = 24000

    get_log_mel_spectrogram = {
        'sample_rate': sample_rate,
        # SOURCE (Tacotron 2):
        # mel spectrograms are computed through a shorttime Fourier transform (STFT)
        # using a 50 ms frame size, 12.5 ms frame hop, and a Hann window function.
        'frame_size': 1200,  # 50ms * 24,000 / 1000 == 1200
        'frame_hop': 300,  # 12.5ms * 24,000 / 1000 == 300
        'window': 'hann',

        # SOURCE (Tacotron 1):
        # 2048-point Fourier transform
        'fft_length': 2048,

        # SOURCE (Tacotron 2):
        # Prior to log compression, the filterbank output magnitudes are clipped to a
        # minimum value of 0.01 in order to limit dynamic range in the logarithmic
        # domain.
        'min_magnitude': 0.01,
    }

    # SOURCE (Tacotron 2):
    # We transform the STFT magnitude to the mel scale using an 80 channel mel
    # filterbank spanning 125 Hz to 7.6 kHz, followed by log dynamic range
    # compression.
    # NOTE: Following running a SoX 7.6 kHz low-pass filter on a LJ dataset sample at 7.6 kHz,
    # we found that her voice tends to use higher frequencies than 7.6 kHz. We bumped it up to 9.1
    # kHz by looking at a melspectrogram of the sample.
    lower_hertz = 125
    upper_hertz = 9100

    # SOURCE (WaveNet):
    # where −1 < xt < 1 and µ = 255.
    signal_channels = 256  # NOTE: signal_channels = µ + 1

    # SOURCE: Efficient Neural Audio Synthesis
    # The WaveRNN model is a single-layer RNN with a dual softmax layer that is
    # designed to efficiently predict 16-bit raw audio samples.
    bits = 16

    librosa.effects.trim = configurable(librosa.effects.trim)
    IPython.display.Audio.__init__ = configurable(IPython.display.Audio.__init__)

    add_config({
        'librosa.effects.trim': {
            'frame_length': get_log_mel_spectrogram['frame_size'],
            'hop_length': get_log_mel_spectrogram['frame_hop'],
            # NOTE: Manually determined to be a adequate cutoff for Linda Johnson via:
            # ``notebooks/Stripping Silence.ipynb``
            'top_db': 50
        },
        'IPython.lib.display.Audio.__init__.rate': sample_rate,
        'src': {
            'datasets.lj_speech.lj_speech_dataset': {
                'resample': sample_rate,
                # NOTE: ``Signal Loudness Distribution`` notebook shows that LJ Speech is biased
                # concerning the loudness and ``norm=True`` unbiases this. In addition, norm
                # also helps smooth out the distribution in notebook ``Signal Energy Distribution``.
                # While, ``loudness=True`` does not help.
                'norm': True,
                'loudness': False,
                # NOTE: Guard to reduce clipping during resampling
                'guard': False,
                # NOTE: Highpass and lowpass filter to ensure Wav is consistent with Spectrogram.
                'lower_hertz': lower_hertz,
                'upper_hertz': upper_hertz,
            },
            'lr_schedulers.DelayedExponentialLR.__init__': {
                # SOURCE (Tacotron 2):
                # learning rate of 10−3 exponentially decaying to 10−5 starting after 50,000
                # iterations.
                # NOTE: Over email the authors confirmed they ended decay at 100,000 steps
                # NOTE: Authors mentioned this hyperparameter is dependent on the dataset; for
                # the LJ speech dataset, we see that dev and train loss begin to diverge at 20k.
                'epoch_start_decay': 10000,
                'epoch_end_decay': 60000,
                'end_lr': 10**-5,
            },
            'optimizer.Optimizer.__init__': {
                'max_grad_norm': None,
            },
            'audio': {
                # SOURCE (Wavenet):
                # To make this more tractable, we first apply a µ-law companding transformation
                # (ITU-T, 1988) to the data, and then quantize it to 256 possible values
                'mu_law_encode.bins': signal_channels,
                'mu_law_decode.bins': signal_channels,
                'mu_law.bins': signal_channels,
                'read_audio.sample_rate': sample_rate,
                'get_log_mel_spectrogram': get_log_mel_spectrogram,
                'griffin_lim': {
                    'frame_size': get_log_mel_spectrogram['frame_size'],
                    'frame_hop': get_log_mel_spectrogram['frame_hop'],
                    'fft_length': get_log_mel_spectrogram['fft_length'],
                    'window': get_log_mel_spectrogram['window'],
                    'sample_rate': sample_rate,
                    # SOURCE (Tacotron 1):
                    # We found that raising the predicted magnitudes by a power of 1.2 before
                    # feeding to Griffin-Lim reduces artifacts
                    'power': 1.20,
                    # SOURCE (Tacotron 1):
                    # We observed that Griffin-Lim converges after 50 iterations (in fact, about 30
                    # iterations seems to be enough), which is reasonably fast.
                    'iterations': 30,
                },
                'mel_filters': {
                    'fft_length': get_log_mel_spectrogram['fft_length'],
                    'sample_rate': sample_rate,
                    # SOURCE (Tacotron 2):
                    # We transform the STFT magnitude to the mel scale using an 80 channel mel
                    # filterbank spanning 125 Hz to 7.6 kHz, followed by log dynamic range
                    # compression.
                    'num_mel_bins': frame_channels,
                    'lower_hertz': lower_hertz,
                    'upper_hertz': upper_hertz,
                }
            },
            'feature_model': {
                'encoder.Encoder.__init__': {
                    'lstm_dropout': lstm_dropout,
                    'convolution_dropout': convolution_dropout,

                    # SOURCE (Tacotron 2):
                    # Input characters are represented using a learned 512-dimensional character
                    # embedding
                    'embedding_dim': 512,

                    # SOURCE (Tacotron 2):
                    # which are passed through a stack of 3 convolutional layers each containing
                    # 512 filters with shape 5 × 1, i.e., where each filter spans 5 characters
                    'num_convolution_layers': 3,
                    'num_convolution_filters': 512,
                    'convolution_filter_size': 5,

                    # SOURCE (Tacotron 2)
                    # The output of the final convolutional layer is passed into a single
                    # bi-directional [19] LSTM [20] layer containing 512 units (256) in each
                    # direction) to generate the encoded features.
                    'lstm_hidden_size': encoder_hidden_size,  # 512
                    'lstm_layers': 1,
                    'lstm_bidirectional': True,
                },
                'attention.LocationSensitiveAttention.__init__': {
                    'encoder_hidden_size': encoder_hidden_size,

                    # SOURCE (Tacotron 2):
                    # Attention probabilities are computed after projecting inputs and location
                    # features to 128-dimensional hidden representations.
                    'hidden_size': attention_hidden_size,

                    # SOURCE (Tacotron 2):
                    # Location features are computed using 32 1-D convolution filters of length
                    # 31.
                    'num_convolution_filters': 32,
                    'convolution_filter_size': 31,
                },
                'decoder.AutoregressiveDecoder.__init__': {
                    'frame_channels': frame_channels,
                    'pre_net_hidden_size': pre_net_hidden_size,
                    'encoder_hidden_size': encoder_hidden_size,
                    'lstm_dropout': lstm_dropout,
                    'attention_context_size': attention_hidden_size,

                    # SOURCE (Tacotron 2):
                    # The prenet output and attention context vector are concatenated and
                    # passed through a stack of 2 uni-directional LSTM layers with 1024 units.
                    'lstm_hidden_size': 1024,
                },
                'pre_net.PreNet.__init__': {
                    'frame_channels': frame_channels,

                    # SOURCE (Tacotron 2):
                    # The prediction from the previous time step is first passed through a small
                    # pre-net containing 2 fully connected layers of 256 hidden ReLU units.
                    'num_layers': 2,
                    'hidden_size': pre_net_hidden_size,

                    # SOURCE (Tacotron 2):
                    # In order to introduce output variation at inference time, dropout with
                    # probability 0.5 is applied only to layers in the pre-net of the
                    # autoregressive decoder.
                    'dropout': 0.5,
                },
                'post_net.PostNet.__init__': {
                    'frame_channels': frame_channels,
                    'convolution_dropout': convolution_dropout,

                    # SOURCE (Tacotron 2):
                    # Finally, the predicted mel spectrogram is passed
                    # through a 5-layer convolutional post-net which predicts a residual
                    # to add to the prediction to improve the overall reconstruction
                    'num_convolution_layers': 5,

                    # SOURCE (Tacotron 2):
                    # Each post-net layer is comprised of 512 filters with shape 5 × 1 with
                    # batch normalization, followed by tanh activations on all but the final
                    # layer
                    'num_convolution_filters': 512,
                    'convolution_filter_size': 5,
                },
                'model.SpectrogramModel.__init__': {
                    'encoder_hidden_size': encoder_hidden_size,
                    'frame_channels': frame_channels,
                }
            },
            'signal_model': {
                'residual_block.ResidualBlock.__init__': {
                    # Tacotron and Parallel WaveNet use kernel size of 3 to increase their receptive
                    # field. However, nv-Wavenet only supports a kernel size of 2.
                    # ISSUE: https://github.com/NVIDIA/nv-wavenet/issues/21
                    'kernel_size': 2
                },
                'wave_net.WaveNet.__init__': {
                    'signal_channels': signal_channels,
                    'local_features_size': frame_channels,

                    # SOURCE Parallel WaveNet: (256 block hidden size)
                    # The number of hidden units in the gating layers is 512 (split into two groups
                    # of 256 for the two parts of the activation function (1)).
                    # SOURCE Deep Voice: (64 block hidden size)
                    # Our highest-quality final model uses l = 40 layers, r = 64 residual channels,
                    # and s = 256 skip channels.
                    'block_hidden_size': 64,
                    'skip_size': 256,

                    # SOURCE Tacotron 2: (From their ablation studies)
                    # Total Layers: 24 | Num Cycles: 4 |  Dilation cycle size: 6
                    # NOTE: We increase the cycle size to 8 to increase the receptive field to 766
                    # samples. Unfortunatly, we cannot increase the kernel size.
                    'num_layers': 28,
                    'cycle_size': 7,
                    'upsample_chunks': 4,

                    # SOURCE: Tacotron 2
                    # only 2 upsampling layers are used in the conditioning stack instead of 3
                    # layers.
                    # SOURCE: Tacotron 2 Author Google Chat
                    # We upsample 4x with the layers and then repeat each value 75x
                    'upsample_convs': [4],
                    'upsample_repeat': 75,
                },
                'wave_rnn.WaveRNN.__init__': {
                    'local_features_size': frame_channels,

                    # SOURCE: Efficient Neural Audio Synthesis
                    # The WaveRNN model is a single-layer RNN with a dual softmax layer that is
                    # designed to efficiently predict 16-bit raw audio samples.
                    'bits': bits,

                    # SOURCE: Efficient Neural Audio Synthesis
                    # We see that the WaveRNN with 896 units achieves NLL scores comparable to those
                    # of the largest WaveNet model
                    'hidden_size': 896,

                    # SOURCE: Tacotron 2
                    # only 2 upsampling layers are used in the conditioning stack instead of 3
                    # layers.
                    # SOURCE: Tacotron 2 Author Google Chat
                    # We upsample 4x with the layers and then repeat each value 75x
                    'upsample_convs': [4],
                    'upsample_repeat': 75,
                }
            },
            'bin.signal_model.train.Trainer.__init__.sample_rate': sample_rate,
            'utils.utils': {
                'plot_waveform.sample_rate': sample_rate,
                'plot_log_mel_spectrogram': {
                    'sample_rate': sample_rate,
                    'frame_hop': get_log_mel_spectrogram['frame_hop'],
                    'lower_hertz': lower_hertz,
                    'upper_hertz': upper_hertz,
                },
                'split_signal.bits': bits,
                'combine_signal.bits': bits,
            }
        }
    })
