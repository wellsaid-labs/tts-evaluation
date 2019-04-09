import logging

from torch import nn

import torch

from src.hparams import add_config
from src.hparams import configurable

from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam

logger = logging.getLogger(__name__)


def _set_anomaly_detection():
    # NOTE: Prevent circular dependency
    from src.utils import AnomalyDetector
    add_config({
        'src.bin.train.signal_model.trainer.Trainer.__init__.min_rollback': 2,
        'src.utils.AnomalyDetector.__init__': {
            # NOTE: Determined empirically with the notebook:
            # ``notebooks/Detecting Anomalies.ipynb``
            'sigma': 6,
            'beta': 0.98,
            'type_': AnomalyDetector.TYPE_HIGH,
        }
    })


def _set_audio_processing():
    # SOURCE (Tacotron 1):
    # We use 24 kHz sampling rate for all experiments.
    sample_rate = 24000

    # SOURCE (Tacotron 2):
    # mel spectrograms are computed through a shorttime Fourier transform (STFT)
    # using a 50 ms frame size, 12.5 ms frame hop, and a Hann window function.
    frame_size = int(50 * sample_rate / 1000)  # NOTE: Frame size in samples
    frame_hop = int(12.5 * sample_rate / 1000)
    window = 'hann'

    fft_length = 2048

    # SOURCE (Tacotron 2):
    # Prior to log compression, the filterbank output magnitudes are clipped to a
    # minimum value of 0.01 in order to limit dynamic range in the logarithmic
    # domain.
    min_magnitude = 0.01

    # SOURCE (Tacotron 2):
    # We transform the STFT magnitude to the mel scale using an 80 channel mel
    # filterbank spanning 125 Hz to 7.6 kHz, followed by log dynamic range
    # compression.
    # SOURCE (Tacotron 2 Author):
    # Google mentioned they settled on [20, 12000] with 128 filters in Google Chat.
    frame_channels = 128
    hertz_bounds = {'lower_hertz': None, 'upper_hertz': None}

    # SOURCE: Efficient Neural Audio Synthesis
    # The WaveRNN model is a single-layer RNN with a dual softmax layer that is
    # designed to efficiently predict 16-bit raw audio samples.
    bits = 16

    try:
        import librosa
        librosa.effects.trim = configurable(librosa.effects.trim)
        librosa.output.write_wav = configurable(librosa.output.write_wav)
        add_config({
            'librosa.effects.trim': {
                'frame_length': frame_size,
                'hop_length': frame_hop,
                # NOTE: Manually determined to be a adequate cutoff for Linda Johnson via:
                # ``notebooks/Stripping Silence.ipynb``
                'top_db': 50
            },
            'librosa.output.write_wav.sr': sample_rate})
    except ImportError:
        logger.info('Ignoring optional `librosa` configurations.')

    try:
        import IPython
        IPython.display.Audio.__init__ = configurable(IPython.display.Audio.__init__)
        add_config({'IPython.lib.display.Audio.__init__.rate': sample_rate})
    except ImportError:
        logger.info('Ignoring optional `IPython` configurations.')

    try:
        import scipy.io.wavfile
        scipy.io.wavfile.write = configurable(scipy.io.wavfile.write)
        add_config({'scipy.io.wavfile.write.rate': sample_rate})
    except ImportError:
        logger.info('Ignoring optional `scipy` configurations.')

    add_config({
        'src.datasets.lj_speech.lj_speech_dataset': {
            'resample': sample_rate,
            # NOTE: ``Signal Loudness Distribution`` notebook shows that LJ Speech is biased
            # concerning the loudness and ``norm=True`` unbiases this. In addition, norm
            # also helps smooth out the distribution in notebook ``Signal Energy Distribution``.
            # While, ``loudness=True`` does not help.
            'norm': True,
            # NOTE: Guard to reduce clipping during resampling
            'guard': True,
        },
        'src.datasets.hilary.hilary_dataset.resample': sample_rate,
        'src.datasets.m_ailabs.m_ailabs_speech_dataset.resample': sample_rate,
        'src.audio': {
            # SOURCE (Wavenet):
            # To make this more tractable, we first apply a µ-law companding transformation
            # (ITU-T, 1988) to the data, and then quantize it to 256 possible values
            'read_audio.sample_rate': sample_rate,
            # NOTE: Practically, `frame_rate` is equal to `sample_rate`. However, the terminology is
            # more appropriate because `sample_rate` is ambiguous. In a multi-channel scenario, each
            # channel has its own set of samples. It's unclear if `sample_rate` depends on the
            # number of channels, scaling linearly per channel.
            'build_wav_header.frame_rate': sample_rate,
            'get_log_mel_spectrogram': {
                'sample_rate': sample_rate,
                'frame_size': frame_size,
                'frame_hop': frame_hop,
                'window': window,
                'fft_length': fft_length,
                'min_magnitude': min_magnitude,
            },
            'griffin_lim': {
                'frame_size': frame_size,
                'frame_hop': frame_hop,
                'fft_length': fft_length,
                'window': window,
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
            '_mel_filters': {
                'fft_length': fft_length,
                # SOURCE (Tacotron 2):
                # We transform the STFT magnitude to the mel scale using an 80 channel mel
                # filterbank spanning 125 Hz to 7.6 kHz, followed by log dynamic range
                # compression.
                'num_mel_bins': frame_channels,
                **hertz_bounds
            },
            'split_signal.bits': bits,
            'combine_signal.bits': bits,
        },
        'src.visualize': {
            'CometML.<locals>.log_audio.sample_rate': sample_rate,
            'plot_waveform.sample_rate': sample_rate,
            'plot_spectrogram': {
                'sample_rate': sample_rate,
                'frame_hop': frame_hop,
                'y_axis': 'mel',
                **hertz_bounds
            },
        }
    })

    return frame_channels, frame_hop, bits


def _set_model_size(frame_channels, bits):

    # SOURCE (Tacotron 2):
    # The prediction from the previous time step is first passed through a small
    # pre-net containing 2 fully connected layers of 256 hidden ReLU units.
    pre_net_hidden_size = 256

    # SOURCE (Tacotron 2):
    # Attention probabilities are computed after projecting inputs and location
    # features to 128-dimensional hidden representations.
    attention_hidden_size = 128

    add_config({
        'src': {
            'spectrogram_model': {
                'encoder.Encoder.__init__': {
                    # SOURCE (Tacotron 2):
                    # Input characters are represented using a learned 512-dimensional character
                    # embedding
                    'token_embedding_dim': 512,

                    # SOURCE (Transfer Learning from Speaker Verification to Multispeaker
                    #         Text-To-Speech Synthesis):
                    # The paper mentions their proposed model uses a 256 dimension embedding.
                    'speaker_embedding_dim': 256,

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
                    'lstm_hidden_size': 512,
                    'lstm_layers': 1,
                    'lstm_bidirectional': True,

                    # SOURCE (Tacotron 2)
                    # Attention probabilities are computed after projecting inputs and location
                    # features to 128-dimensional hidden representations.
                    'out_dim': attention_hidden_size,
                },
                'attention.LocationSensitiveAttention.__init__': {
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
                    'pre_net_hidden_size': pre_net_hidden_size,
                    'attention_hidden_size': attention_hidden_size,

                    # SOURCE (Tacotron 2):
                    # The prenet output and attention context vector are concatenated and
                    # passed through a stack of 2 uni-directional LSTM layers with 1024 units.
                    'lstm_hidden_size': 1024,
                },
                'pre_net.PreNet.__init__': {
                    # SOURCE (Tacotron 2):
                    # The prediction from the previous time step is first passed through a small
                    # pre-net containing 2 fully connected layers of 256 hidden ReLU units.
                    'num_layers': 2,
                    'hidden_size': pre_net_hidden_size,
                },
                'post_net.PostNet.__init__': {
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
                'model.SpectrogramModel.__init__.frame_channels': frame_channels,
            },
            'signal_model.wave_rnn.WaveRNN.__init__': {
                'local_features_size': frame_channels,

                # SOURCE: Efficient Neural Audio Synthesis
                # The WaveRNN model is a single-layer RNN with a dual softmax layer that is
                # designed to efficiently predict 16-bit raw audio samples.
                'bits': bits,

                # SOURCE: Efficient Neural Audio Synthesis
                # We see that the WaveRNN with 896 units achieves NLL scores comparable to
                # those of the largest WaveNet model
                'hidden_size': 896,

                # SOURCE: Efficient Neural Audio Synthesis Author
                # The author suggested adding 3 - 5 convolutions on top of WaveRNN.
                # SOURCE:
                # https://github.com/pytorch/examples/blob/master/super_resolution/model.py
                # Upsampling layer is inspired by super resolution
                'upsample_kernels': [(5, 5), (3, 3), (3, 3), (3, 3)],

                # SOURCE: Tacotron 2
                # only 2 upsampling layers are used in the conditioning stack instead of 3
                # layers.
                # SOURCE: Tacotron 2 Author Google Chat
                # We upsample 4x with the layers and then repeat each value 75x
                'upsample_num_filters': [64, 64, 32, 10],
                'upsample_repeat': 30
            }
        }
    })


def set_hparams():
    """ Using the ``configurable`` module set the hyperparameters for the source code.
    """
    frame_channels, frame_hop, bits = _set_audio_processing()
    _set_anomaly_detection()
    _set_model_size(frame_channels, bits)

    torch.optim.Adam.__init__ = configurable(torch.optim.Adam.__init__)
    nn.modules.batchnorm._BatchNorm.__init__ = configurable(
        nn.modules.batchnorm._BatchNorm.__init__)
    add_config({
        # NOTE: `momentum=0.01` to match Tensorflow defaults
        'torch.nn.modules.batchnorm._BatchNorm.__init__.momentum': 0.01,
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

    # NOTE: Prevent circular dependency
    from src import datasets
    dataset = datasets.lj_speech_dataset

    # SOURCE (Tacotron 2):
    # Finally, the predicted mel spectrogram is passed
    train_signal_model_from_predicted_spectrogram = True

    spectrogram_model_dev_batch_size = 256

    # TODO: Add option to instead of strings to use direct references.
    add_config({
        'src': {
            'spectrogram_model': {
                'encoder.Encoder.__init__': {
                    'lstm_dropout': lstm_dropout,
                    'convolution_dropout': convolution_dropout,
                },
                'decoder.AutoregressiveDecoder.__init__.lstm_dropout': lstm_dropout,
                # SOURCE (Tacotron 2):
                # In order to introduce output variation at inference time, dropout with
                # probability 0.5 is applied only to layers in the pre-net of the
                # autoregressive decoder.
                'pre_net.PreNet.__init__.dropout': 0.5,
                'post_net.PostNet.__init__.convolution_dropout': 0.0
            },
            'bin.evaluate.main.dataset': dataset,
            'datasets.process.compute_spectrograms.batch_size': spectrogram_model_dev_batch_size,
            'bin.train': {
                'spectrogram_model': {
                    '__main__._get_dataset.dataset': dataset,
                    'trainer.Trainer.__init__': {
                        # SOURCE: Tacotron 2
                        # To train the feature prediction network, we apply the standard
                        # maximum-likelihood training procedure (feeding in the correct output
                        # instead of the predicted output on the decoder side, also referred to as
                        # teacher-forcing) with a batch size of 64 on a single GPU.
                        # NOTE: Parameters set after experimentation on a 1 Px100 GPU.
                        'train_batch_size': 64,
                        'dev_batch_size': spectrogram_model_dev_batch_size,

                        # SOURCE (Tacotron 2):
                        # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999
                        'optimizer': Adam,

                        # SOURCE (Tacotron 2 Author):
                        # The author confirmed they used BCE loss in Google Chat.
                        'criterion_stop_token': BCEWithLogitsLoss,

                        # SOURCE: Tacotron 2
                        # We minimize the summed mean squared error (MSE) from before and after the
                        # post-net to aid convergence.
                        'criterion_spectrogram': MSELoss,
                    },
                },
                'signal_model': {
                    '__main__._get_dataset.dataset': dataset,
                    'trainer.Trainer.__init__': {
                        # SOURCE (Tacotron 2):
                        # We train with a batch size of 128 distributed across 32 GPUs with
                        # synchronous updates, using the Adam optimizer with β1 = 0.9, β2 = 0.999, 
                        # eps = 10−8 and a fixed learning rate of 10−4
                        # NOTE: Parameters set after experimentation on a 4 Px100 GPU.
                        'train_batch_size': 64,
                        'dev_batch_size': 256,

                        'use_predicted': train_signal_model_from_predicted_spectrogram,

                        # These are not directly mentioned in the paper; however are popular choices
                        # as of Jan 2019 for a sequence to sequence task.
                        'criterion': CrossEntropyLoss,
                        'optimizer': Adam
                    },
                    'data_loader.DataLoader.__init__': {
                        # SOURCE: Efficient Neural Audio Synthesis
                        # The WaveRNN models are trained on sequences of 960 audio samples
                        'spectrogram_slice_size': int(900 / frame_hop),
                        # TODO: This should depend on an upsample property.
                        'spectrogram_slice_pad': 5,
                    },
                }
            },
            # NOTE: Window size smoothing parameter is not super sensative.
            'optimizers.AutoOptimizer.__init__.window_size': 128,
        }
    })
