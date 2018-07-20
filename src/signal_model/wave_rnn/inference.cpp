#include <torch/torch.h>
#include <iostream>
#include <vector>

std::vector<at::Tensor> inference(
    at::Tensor condition,
    at::Tensor last_coarse,
    at::Tensor last_fine,
    at::Tensor last_hidden_state,
    at::Tensor coarse_bias,
    at::Tensor coarse_weight,
    at::Tensor fine_bias,
    at::Tensor fine_weight,
    at::Tensor hidden_bias,
    at::Tensor hidden_weight,
    at::Tensor to_bins_coarse_in_bias,
    at::Tensor to_bins_coarse_in_weight,
    at::Tensor to_bins_coarse_out_bias,
    at::Tensor to_bins_coarse_out_weight,
    at::Tensor to_bins_fine_in_bias,
    at::Tensor to_bins_fine_in_weight,
    at::Tensor to_bins_fine_out_bias,
    at::Tensor to_bins_fine_out_weight)
{
  /** Run the Wave RNN inference algorithm.
   *
   * @param condition [batch_size, signal_length, 3, hidden_size] Value to condition gates.
   * @param last_coarse [batch_size, 1] Last predicted coarse value.
   * @param last_fine [batch_size, 1] Last predicted fine value.
   * @param last_hidden_state [batch_size, hidden_size] Last RNN state.
   * @param coarse_bias [1.5 * hidden_size] Bias for coarse projection.
   * @param coarse_weight [1.5 * hidden_size, 2] Weights for coarse projection.
   * @param fine_bias [1.5 * hidden_size] Bias for fine projection.
   * @param fine_weight [1.5 * hidden_size, 3] Weights for coarse projection.
   * @param hidden_bias [3 * hidden_size] Bias for hidden projection.
   * @param hidden_weight [3 * hidden_size, hidden_size] Weights for hidden projection.
   * @param to_bins_coarse_in_bias [0.5 * hidden_size] Input dense layer bias to compute
   *    coarse categorical distribution.
   * @param to_bins_coarse_in_weight [0.5 * hidden_size, 0.5 * hidden_size] Input dense layer
   *    weights to compute coarse categorical distribution.
   * @param to_bins_coarse_out_bias [bins] Ouput dense layer bias to compute coarse categorical
   *    distribution.
   * @param to_bins_coarse_out_weight [bins, 0.5 * hidden_size] Ouput dense layer weights to compute
   *    coarse categorical distribution.
   * @param to_bins_fine_in_bias [0.5 * hidden_size] Input dense layer bias to compute
   *    fine categorical distribution.
   * @param to_bins_fine_in_weight [0.5 * hidden_size, 0.5 * hidden_size] Input dense layer
   *    weights to compute fine categorical distribution.
   * @param to_bins_fine_out_bias [bins] Ouput dense layer bias to compute fine categorical
   *    distribution.
   * @param to_bins_fine_out_weight [bins, 0.5 * hidden_size] Ouput dense layer weights to compute
   *    fine categorical distribution.
   */

  // [batch_size, signal_length, 3, size] → [batch_size, signal_length, size]
  auto chunked_condition = condition.chunk(3, /*dim=*/3);

  // [batch_size, signal_length, size] → [batch_size, signal_length, half_size]
  auto condition_r = chunked_condition[0].chunk(2, /*dim=*/2);
  auto condition_u = chunked_condition[1].chunk(2, /*dim=*/2);
  auto condition_e = chunked_condition[2].chunk(2, /*dim=*/2);

  auto batch_size = condition.size(0);

  // for (int i = 0; i < batch_size; ++i)
  // {
  //   hidden_state_projection = at::addmm()
  // }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("run", &inference, "Run inference for WaveRNN.");
}
