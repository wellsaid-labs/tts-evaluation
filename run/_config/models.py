import logging

import torch
import torch.nn
from hparams import HParams, add_config

import lib
import run
from run._config.audio import NUM_FRAME_CHANNELS

logger = logging.getLogger(__name__)


def configure():
    """Configure spectrogram and signal model."""
    # SOURCE (Tacotron 2):
    # Attention probabilities are computed after projecting inputs and location
    # features to 128-dimensional hidden representations.
    encoder_output_size = 128

    # SOURCE (Tacotron 2):
    # Specifically, generation completes at the first frame for which this
    # probability exceeds a threshold of 0.5.
    stop_threshold = 0.5

    # NOTE: These values can be increased as needed, they preemtively allocate model
    # parameters.
    # TODO: After "grapheme to phoneme" is deprecated consider setting these automatically.
    max_tokens = 1000
    max_speakers = 100
    max_sessions = 10000

    # NOTE: Configure the model sizes.
    config = {
        run._models.spectrogram_model.wrapper.SpectrogramModel.__init__: HParams(
            max_tokens=max_tokens,
            max_speakers=max_speakers,
            max_sessions=max_sessions,
        ),
        run._models.spectrogram_model.encoder.Encoder.__init__: HParams(
            # SOURCE (Tacotron 2):
            # Input characters are represented using a learned 512-dimensional character embedding
            # ...
            # which are passed through a stack of 3 convolutional layers each containing
            # 512 filters with shape 5 × 1, i.e., where each filter spans 5 characters
            # ...
            # The output of the final convolutional layer is passed into a single bi-directional
            # [19] LSTM [20] layer containing 512 units (256) in each direction) to generate the
            # encoded features.
            hidden_size=512,
            # SOURCE (Tacotron 2):
            # which are passed through a stack of 3 convolutional layers each containing
            # 512 filters with shape 5 × 1, i.e., where each filter spans 5 characters
            num_conv_layers=3,
            conv_filter_size=5,
            # SOURCE (Tacotron 2)
            # The output of the final convolutional layer is passed into a single
            # bi-directional [19] LSTM [20] layer containing 512 units (256) in each
            # direction) to generate the encoded features.
            lstm_layers=2,
            out_size=encoder_output_size,
        ),
        run._models.spectrogram_model.attention.Attention.__init__: HParams(
            # SOURCE (Tacotron 2):
            # Location features are computed using 32 1-D convolution filters of length 31.
            # SOURCE (Tacotron 2):
            # Attention probabilities are computed after projecting inputs and location
            # features to 128-dimensional hidden representations.
            hidden_size=128,
            conv_filter_size=9,
            # NOTE: The alignment between text and speech is monotonic; therefore, the attention
            # progression should reflect that. The `window_length` ensures the progression is
            # limited.
            # NOTE: Comet visualizes the metric "attention_std", and this metric represents the
            # number of characters the model is attending too at a time. That metric can be used
            # to set the `window_length`.
            window_length=9,
            avg_frames_per_token=1.4555,
        ),
        run._models.spectrogram_model.decoder.Decoder.__init__: HParams(
            encoder_output_size=encoder_output_size,
            # SOURCE (Tacotron 2):
            # The prediction from the previous time step is first passed through a small
            # pre-net containing 2 fully connected layers of 256 hidden ReLU units.
            pre_net_size=256,
            # SOURCE (Tacotron 2):
            # The prenet output and attention context vector are concatenated and
            # passed through a stack of 2 uni-directional LSTM layers with 1024 units.
            lstm_hidden_size=1024,
        ),
        run._models.spectrogram_model.pre_net.PreNet.__init__: HParams(
            # SOURCE (Tacotron 2):
            # The prediction from the previous time step is first passed through a small
            # pre-net containing 2 fully connected layers of 256 hidden ReLU units.
            num_layers=2
        ),
        run._models.spectrogram_model.model.SpectrogramModel.__init__: HParams(
            num_frame_channels=NUM_FRAME_CHANNELS,
            # SOURCE (Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech
            #         Synthesis):
            # The paper mentions their proposed model uses a 256 dimension embedding.
            # NOTE: See https://github.com/wellsaid-labs/Text-to-Speech/pull/258 to learn more about
            # this parameter.
            seq_meta_embed_size=128,
        ),
        run._models.signal_model.wrapper.SpectrogramDiscriminator.__init__: HParams(
            max_speakers=max_speakers,
            max_sessions=max_sessions,
        ),
        run._models.signal_model.wrapper.SignalModel.__init__: HParams(
            max_speakers=max_speakers,
            max_sessions=max_sessions,
        ),
        run._models.signal_model.model.SignalModel.__init__: HParams(
            seq_meta_embed_size=128,
            frame_size=NUM_FRAME_CHANNELS,
            hidden_size=32,
            max_channel_size=512,
        ),
        # NOTE: We found this hidden size to be effective on Comet in April 2020.
        run._models.signal_model.SpectrogramDiscriminator.__init__: HParams(
            seq_meta_embed_size=128,
            hidden_size=512,
        ),
    }
    add_config(config)

    # NOTE: Configure the model regularization.
    config = {
        # SOURCE (Tacotron 2):
        # In order to introduce output variation at inference time, dropout with probability 0.5 is
        # applied only to layers in the pre-net of the autoregressive decoder.
        run._models.spectrogram_model.pre_net.PreNet.__init__: HParams(dropout=0.5),
        run._models.spectrogram_model.attention.Attention.__init__: HParams(dropout=0.1),
        run._models.spectrogram_model.decoder.Decoder.__init__: HParams(stop_net_dropout=0.5),
        # NOTE: This dropout approach proved effective in Comet in March 2020.
        run._models.spectrogram_model.encoder.Encoder.__init__: HParams(
            dropout=0.1, seq_meta_embed_dropout=0.1
        ),
    }
    add_config(config)

    config = {
        # NOTE: Window size smoothing parameter is not sensitive.
        lib.optimizers.AdaptiveGradientNormClipper.__init__: HParams(window_size=128, norm_type=2),
        # NOTE: The `beta` parameter is not sensitive.
        lib.optimizers.ExponentialMovingParameterAverage.__init__: HParams(beta=0.9999),
        run._models.signal_model.model.SignalModel.__init__: HParams(
            # SOURCE https://en.wikipedia.org/wiki/%CE%9C-law_algorithm:
            # For a given input x, the equation for μ-law encoding is where μ = 255 in the North
            # American and Japanese standards.
            mu=255,
        ),
        run._models.spectrogram_model.model.SpectrogramModel.__init__: HParams(
            # NOTE: The spectrogram values range from -50 to 50. Thie scalar rescales the
            # spectrogram to a more reasonable range for deep learning.
            output_scalar=10.0,
            stop_threshold=stop_threshold,
        ),
    }
    add_config(config)

    # NOTE: PyTorch and Tensorflow parameterize `BatchNorm` differently, learn more:
    # https://stackoverflow.com/questions/48345857/batchnorm-momentum-convention-pytorch?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    config = {
        # NOTE: `momentum=0.01` to match Tensorflow defaults.
        torch.nn.modules.batchnorm._BatchNorm.__init__: HParams(momentum=0.01),
        # NOTE: BERT uses `eps=1e-12` for `LayerNorm`, see here:
        # https://github.com/huggingface/transformers/blob/master/src/transformers/configuration_bert.py
        torch.nn.LayerNorm.__init__: HParams(eps=1e-12),
    }
    add_config(config)
