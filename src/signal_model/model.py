import sys
sys.path.insert(0, 'third_party/nv-wavenet/pytorch')

import logging

from torch import nn

import torch

from src.signal_model.residual_block import ResidualBlock
from src.signal_model.upsample import ConditionalFeaturesUpsample
from src.utils.configurable import configurable
from src.audio import mu_law_decode

logger = logging.getLogger(__name__)


def get_receptive_field_size(layers):
    """ Compute receptive field size.

    Args:
        layers (ResidualBlock): Layers of residual blocks used for Wavenet.

    Returns:
        receptive_field_size (int): Receptive field size in samples.
    """
    return sum([layer.dilation * (layer.kernel_size - 1) for layer in layers]) + 1


class WaveNet(nn.Module):
    """
    Notes:
        * Tacotron 2 authors mention on Google Chat:  "We upsample 4x with the layers and then
          repeat each value 75x".

    Args:
        mu (int): Mu used to encode signal with mu-law encoding. [Parameter ``A - 1`` in NV-WaveNet]
        block_hidden_size (int): Hidden size of each residual block.
        num_layers (int): Number of residual layers to use with Wavenet.
        cycle_size (int): Cycles such that dilation is equal to:
            ``2**(i % cycle_size) {0 <= i < num_layers}``
        upsample_convs (list of int): Size of convolution layers used to upsample local features
            (e.g. 256 frames x 4 x ...).
        upsample_repeat (int): Number of times to repeat frames, another upsampling technique.
        local_features_size (int): Dimensionality of local features.
        upsample_chunks (int): Control the memory used by ``upsample_layers`` by breaking the
            operation up into chunks.
    """

    @configurable
    def __init__(self,
                 mu=255,
                 block_hidden_size=64,
                 skip_size=256,
                 num_layers=24,
                 cycle_size=6,
                 upsample_convs=[4],
                 upsample_repeat=75,
                 local_features_size=80,
                 upsample_chunks=3):
        super().__init__()

        self.cycle_size = cycle_size
        self.block_hidden_size = block_hidden_size
        self.mu = mu
        self.num_layers = num_layers
        self.embed = torch.nn.Conv1d(in_channels=1, out_channels=block_hidden_size, kernel_size=1)
        self.layers = nn.ModuleList([
            ResidualBlock(
                hidden_size=block_hidden_size,
                dilation=2**(i % cycle_size),
                skip_size=skip_size,
                is_last_layer=(num_layers == i + 1)) for i in range(num_layers)
        ])
        self.conditional_features_upsample = ConditionalFeaturesUpsample(
            block_hidden_size=block_hidden_size,
            upsample_repeat=upsample_repeat,
            upsample_convs=upsample_convs,
            local_features_size=local_features_size,
            num_layers=num_layers,
            upsample_chunks=upsample_chunks)
        # TODO: Try using TanH as suggested here:
        # https://github.com/ibab/tensorflow-wavenet/issues/143#issuecomment-253400332
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels=skip_size, out_channels=self.mu + 1, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.mu + 1, out_channels=self.mu + 1, kernel_size=1, bias=False),
            nn.LogSoftmax(dim=1),
        )
        torch.nn.init.xavier_uniform_(self.out[1].weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(
            self.out[3].weight, gain=torch.nn.init.calculate_gain('linear'))
        self.has_new_weights = True  # Whether the weights have been updated since last export
        self.kernel = None

        self.receptive_field_size = get_receptive_field_size(self.layers)
        logger.info('Receptive field size in samples: %d' % self.receptive_field_size)

    def queue_kernel_update(self):
        """ Set a flag to update the kernel, incase the weights have changed.

        NOTE:
            * When using DataParallel, for example, it can be difficult to tell when the weights
              have changed; therefore, we ask for a manual queue.
        """
        self.has_new_weights = True

    def _export(self, dtype, device):  # pragma: no cover
        """
        Args:
            dtype (torch.dtype): Type to use with new tensors.
            device (torch.device): Device to put new tensors.

        Notes:
            * Edit hyperparameters inside ``wavenet_infer.cu`` to match.
            * Make sure to build the CUDA kernel.

        Returns:
            kernel (NVWaveNet): NVIDIA optimize wavenet CUDA kernel.
        """
        import nv_wavenet

        logger.info('Exporting signal model...')

        # This implementation does not use embeded ``embedding_prev``
        embedding_prev = torch.zeros(
            (self.mu + 1, self.block_hidden_size), dtype=dtype, device=device)

        # Compute embedding current by the identity matrix through a conv
        # embedding_curr [self.mu + 1]
        embedding_curr = torch.arange(self.mu + 1, dtype=dtype, device=device)
        # ``self.embed`` is trained on unencoded values
        embedding_curr = mu_law_decode(embedding_curr)
        # [self.mu + 1] → [1, 1, self.mu + 1]
        embedding_curr = embedding_curr.view(1, 1, self.mu + 1)
        # [1, 1, self.mu + 1]  → [1, block_hidden_size, self.mu + 1]
        embedding_curr = self.embed(embedding_curr)
        # [1, block_hidden_size, self.mu + 1]  → [self.mu + 1, block_hidden_size]
        embedding_curr = embedding_curr.squeeze(0).transpose(0, 1)

        kwargs = {
            'embedding_prev': embedding_prev,
            'embedding_curr': embedding_curr.detach(),
            'conv_out_weight': self.out[1].weight.detach(),
            'conv_end_weight': self.out[3].weight.detach(),
            'dilate_weights': [l.dilated_conv.weight.detach() for l in self.layers],
            'dilate_biases': [l.dilated_conv.bias.detach() for l in self.layers],
            'max_dilation': max(l.dilation for l in self.layers),
            # Last residual layer does not matter
            'res_weights': [l.out_conv.weight.detach() for l in self.layers[:-1]],
            'res_biases': [l.out_conv.bias.detach() for l in self.layers[:-1]],
            'skip_weights': [l.skip_conv.weight.detach() for l in self.layers],
            'skip_biases': [l.skip_conv.bias.detach() for l in self.layers],
            'use_embed_tanh': False,  # This implementation does not use embeded ``tanh``
        }

        return nv_wavenet.NVWaveNet(**kwargs)

    def forward(self, local_features, gold_signal=None, implementation=None):
        """
        TODO:
            * Consider using ONNX to compile the model:
              https://discuss.pytorch.org/t/partial-onnx-export/18978

        Args:
            local_features (torch.FloatTensor [batch_size, local_length, local_features_size]):
                Local feature to condition signal generation (e.g. spectrogram).
            gold_signal (torch.FloatTensor [batch_size, signal_length], optional): Mu-law encoded
                signal used for teacher-forcing.
            implementation (nv_wavenet.Impl, optional): An implementation used for inference,
                either: AUTO, SINGLE_BLOCK, DUAL_BLOCK, or PERSISTENT

        Returns:

            predicted_signal: Returns former if ``self.training`` else returns latter.
                * (torch.FloatTensor [batch_size, mu + 1, signal_length]): Categorical
                    distribution over ``mu + 1`` energy levels.
                * (torch.FloatTensor [batch_size, signal_length]): Categorical
                    distribution over ``mu + 1`` energy levels.
                The predicted signal is one time step ahead of the gold signal.
        """
        if gold_signal is None and self.training:
            raise ValueError('Training without teacher forcing is not supported.')

        # [batch_size, local_length, local_features_size] →
        # [batch_size, num_layers, block_hidden_size * 2, signal_length]
        conditional_features = self.conditional_features_upsample(local_features)

        if gold_signal is not None:
            assert conditional_features.shape[3] == gold_signal.shape[1], (
                "Upsampling parameters in tangent with signal shape and local features shape " +
                "must be the same length after upsampling.")

        if gold_signal is None and not self.training:  # pragma: no cover
            assert torch.cuda.is_available(), "Inference only supported for CUDA."
            import nv_wavenet
            if self.has_new_weights:
                # Re export weights if ``self.has_new_weights``
                self.kernel = self._export(conditional_features.dtype, conditional_features.device)
                self.has_new_weights = False
            # [batch_size, num_layers, block_hidden_size * 2, signal_length] →
            # [block_hidden_size * 2, batch_size, num_layers, signal_length]
            conditional_features = conditional_features.permute(2, 0, 1, 3).detach()
            implementation = (nv_wavenet.Impl.AUTO if implementation is None else implementation)
            return self.kernel.infer(cond_input=conditional_features, implementation=implementation)

        # [batch_size, signal_length] → [batch_size, 1, signal_length]
        gold_signal = gold_signal.float().unsqueeze(1)

        # [batch_size, 1, signal_length] → [batch_size, block_hidden_size, signal_length]
        gold_signal_features = self.embed(gold_signal.float())

        del gold_signal

        cumulative_skip = None
        for i, layer in enumerate(self.layers):
            gold_signal_features, skip = layer(
                signal_features=gold_signal_features,
                conditional_features=conditional_features[:, i, :, :])

            if cumulative_skip is None:
                cumulative_skip = skip
            else:
                cumulative_skip += skip

        # Convolution operater expects input_ of the form:
        # cumulative_skip [batch_size, channels (skip_size), signal_length] →
        # [batch_size, mu + 1, signal_length]
        return self.out(cumulative_skip)
