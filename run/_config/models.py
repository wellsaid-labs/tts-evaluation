import logging

import config as cf
import torch
import torch.nn

import lib
import run
from run import _config

logger = logging.getLogger(__name__)


def configure(overwrite: bool = False):
    """Configure spectrogram and signal model."""
    # SOURCE (Tacotron 2):
    # Specifically, generation completes at the first frame for which this
    # probability exceeds a threshold of 0.5.
    stop_threshold = 0.5

    # NOTE: These values can be increased as needed, they preemtively allocate model
    # parameters.
    max_tokens = 1000
    max_anno_vector_size = 2
    annos = [("loudness_vector", "loudness_mask"), ("tempo_vector", "tempo_mask")]
    max_speakers = len(set(s.label for s in _config.data.DATASETS.keys()))
    max_dialects = len(set(s.dialect for s in _config.data.DATASETS.keys()))
    max_styles = len(set(s.style for s in _config.data.DATASETS.keys()))
    max_languages = len(set(s.language for s in _config.data.DATASETS.keys()))

    # NOTE: Configure the model sizes.
    config = {
        run._models.spectrogram_model.encoder.Encoder: cf.Args(
            # SOURCE (Tacotron 2):
            # which are passed through a stack of 3 convolutional layers each containing
            # 512 filters with shape 5 × 1, i.e., where each filter spans 5 characters
            num_layers=4,
            conv_filter_size=3,
            num_conv_block_layers=2,
        ),
        run._models.spectrogram_model.attention.Attention: cf.Args(
            # SOURCE (Tacotron 2):
            # Location features are computed using 32 1-D convolution filters of length 31.
            conv_filter_size=9,
            # NOTE: The alignment between text and speech is monotonic; therefore, the attention
            # progression should reflect that. The `window_len` ensures the progression is
            # limited.
            # NOTE: Comet visualizes the metric "attention_std", and this metric represents the
            # number of characters the model is attending too at a time. That metric can be used
            # to set the `window_len`.
            window_len=9,
            # NOTE: This value was computed with a reference frame size of 4096, and it scales
            # linearly with frame size.
            avg_frames_per_token=1.45 * (4096 / _config.audio.FRAME_SIZE),
        ),
        run._models.spectrogram_model.decoder.Decoder: cf.Args(
            # SOURCE (Tacotron 2):
            # The prenet output and attention context vector are concatenated and
            # passed through a stack of 2 uni-directional LSTM layers with 1024 units.
            hidden_size=1024,
        ),
        run._models.spectrogram_model.pre_net.PreNet: cf.Args(
            # SOURCE (Tacotron 2):
            # The prediction from the previous time step is first passed through a small
            # pre-net containing 2 fully connected layers of 256 hidden ReLU units.
            num_layers=1
        ),
        run._models.spectrogram_model.wrapper.SpectrogramModelWrapper: cf.Args(
            max_tokens=max_tokens,
            max_speakers=max_speakers,
            max_dialects=max_dialects,
            max_styles=max_styles,
            max_languages=max_languages,
            num_frame_channels=_config.audio.NUM_FRAME_CHANNELS,
            # NOTE: 396 is the size of the spaCy embeddings that are added to the model.
            max_word_vector_size=396,
            max_seq_vector_size=2,
            max_anno_vector_size=max_anno_vector_size,
            annos=annos,
            # SOURCE (Tacotron 2):
            # Attention probabilities are computed after projecting inputs and location
            # features to 128-dimensional hidden representations.
            # SOURCE (Tacotron 2):
            # Input characters are represented using a learned 512-dimensional character embedding
            # ...
            # which are passed through a stack of 3 convolutional layers each containing
            # 512 filters with shape 5 × 1, i.e., where each filter spans 5 characters
            # ...
            # The output of the final convolutional layer is passed into a single bi-directional
            # [19] LSTM [20] layer containing 512 units (256) in each direction) to generate the
            # encoded features.
            encoder_hidden_size=512,
        ),
        run._models.signal_model.wrapper.SignalModelWrapper: cf.Args(
            max_speakers=max_speakers,
            seq_meta_embed_size=128,
            frame_size=_config.audio.NUM_FRAME_CHANNELS,
            hidden_size=32,
            max_channel_size=512,
        ),
        # NOTE: We found this hidden size to be effective on Comet in April 2020.
        run._models.signal_model.model.SpectrogramDiscriminator: cf.Args(
            seq_meta_embed_size=128, hidden_size=512
        ),
    }
    cf.add(config, overwrite)

    # NOTE: Configure the model regularization.
    config = {
        # SOURCE (Tacotron 2):
        # In order to introduce output variation at inference time, dropout with probability 0.5 is
        # applied only to layers in the pre-net of the autoregressive decoder.
        run._models.spectrogram_model.pre_net.PreNet: cf.Args(dropout=0.4),
    }
    cf.add(config, overwrite)

    config = {
        # NOTE: Window size smoothing parameter is not sensitive.
        lib.optimizers.AdaptiveGradientNormClipper: cf.Args(window_size=128, norm_type=2),
        # NOTE: The `beta` parameter is not sensitive.
        lib.optimizers.ExponentialMovingParameterAverage: cf.Args(beta=0.999),
        run._models.signal_model.wrapper.SignalModelWrapper: cf.Args(
            # SOURCE https://en.wikipedia.org/wiki/%CE%9C-law_algorithm:
            # For a given input x, the equation for μ-law encoding is where μ = 255 in the North
            # American and Japanese standards.
            mu=255,
        ),
        run._models.spectrogram_model.wrapper.SpectrogramModelWrapper: cf.Args(
            stop_threshold=stop_threshold
        ),
    }
    cf.add(config, overwrite)

    config = {
        # NOTE: BERT uses `eps=1e-12` for `LayerNorm`, see here:
        # https://github.com/huggingface/transformers/blob/master/src/transformers/configuration_bert.py
        torch.nn.LayerNorm: cf.Args(eps=1e-12),
        torch.nn.GELU: cf.Args(approximate="tanh"),
    }
    cf.add(config, overwrite)
