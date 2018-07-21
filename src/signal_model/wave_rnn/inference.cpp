#include <torch/torch.h>
#include <iostream>
#include <vector>

std::vector<at::Tensor> step(
    at::Tensor last_hidden_state,
    at::Tensor input,
    at::Tensor project_input_bias,
    at::Tensor project_input_weight,
    at::Tensor hidden_projection,
    at::Tensor to_bins_one_bias,
    at::Tensor to_bins_one_weight,
    at::Tensor to_bins_two_bias,
    at::Tensor to_bins_two_weight,
    bool argmax = false,
    float temperature = 1.0)
{
  /** Run one step of the Wave RNN inference algorithm.
   *
   * @param last_hidden_state [hidden_size / 2, batch_size] Last RNN state.
   * @param input [input_size, batch_size] Input to RNN.
   * @param project_input_weight [1.5 * hidden_size, input_size] Weights to project input for RNN.
   * @param project_input_bias [1.5 * hidden_size] Bias to project input for RNN.
   * @param hidden_projection [1.5 * hidden_size, batch_size] Hidden state projected to
   *    ``1.5 * hidden_size``.
   * @param to_bins_one_bias [hidden_size / 2] Input dense layer bias to compute
   *    categorical distribution.
   * @param to_bins_one_weight [hidden_size / 2, hidden_size / 2] Input dense layer
   *    weights to compute categorical distribution.
   * @param to_bins_two_bias [bins] Ouput dense layer bias to compute categorical
   *    distribution.
   * @param to_bins_two_weight [bins, hidden_size / 2] Ouput dense layer weights to compute
   *    categorical distribution.
   * @param argamax (bool) If ``True``, during inference sample the most likely sample.
   * @param temperature (float) Temperature to control the variance in softmax predictions.
   * @return last_hidden_state [hidden_size / 2, batch_size] Last RNN state.
   * @return sample [batch_size] Sampled class.
   */
  auto hidden_size = hidden_projection.size(0) * 2 / 3;
  auto batch_size = hidden_projection.size(1);
  auto bins = to_bins_two_bias.size(0);

  // [1.5 * hidden_size, batch_size] + [1.5 * hidden_size, input_size] * [input_size, batch_size] →
  // [1.5 * hidden_size, batch_size]
  auto input_projection = at::addmm(project_input_bias,
                                    project_input_weight,
                                    input);

  // ru stands for reset and update gate
  // [1.5 * hidden_size, batch_size] → [hidden_size, batch_size]
  auto input_ru = at::slice(input_projection, /*dim=*/0, /*start=*/0, /*end=*/hidden_size);
  // [3 * hidden_size, batch_size] → [hidden_size, batch_size]
  auto hidden_ru = at::slice(hidden_projection,
                             /*dim=*/0,
                             /*start=*/0,
                             /*end=*/hidden_size);
  // [hidden_size, batch_size] + [hidden_size, batch_size] → [hidden_size, batch_size]
  auto ru = at::sigmoid(input_ru + hidden_ru);

  // [hidden_size, batch_size] → [hidden_size / 2, batch_size]
  auto reset_gate = at::slice(ru, /*dim=*/0, /*start=*/0, /*end=*/hidden_size / 2);

  // e stands for memory
  // [1.5 * hidden_size, batch_size] → [hidden_size / 2, batch_size]
  auto input_e = at::slice(input_projection,
                           /*dim=*/0,
                           /*start=*/hidden_size,
                           /*end=*/hidden_size * 3 / 2);
  // [3 * hidden_size, batch_size] → [hidden_size / 2, batch_size]
  auto hidden_e = at::slice(hidden_projection,
                            /*dim=*/0,
                            /*start=*/hidden_size,
                            /*end=*/hidden_size * 3 / 2);
  auto next_hidden_state = at::tanh(reset_gate * hidden_e + input_e);

  // [hidden_size, batch_size] → [hidden_size / 2, batch_size]
  auto update_gate = at::slice(ru, /*dim=*/0, /*start=*/hidden_size / 2, /*end=*/hidden_size);
  //  last_hidden_state [hidden_size / 2, batch_size]
  last_hidden_state = (update_gate * last_hidden_state +
                       (1.0 - update_gate) * next_hidden_state);

  // [hidden_size / 2, 1] + [hidden_size / 2, hidden_size / 2] * [hidden_size / 2, batch_size] →
  // [hidden_size / 2, batch_size]
  auto out_one = at::relu(at::addmm(at::unsqueeze(to_bins_one_bias, /*dim=*/1),
                                    to_bins_one_weight,
                                    last_hidden_state));
  // [bins, 1] + [bins, hidden_size / 2] * [hidden_size / 2, batch_size] →
  // [bins, batch_size]
  auto out_two = at::addmm(at::unsqueeze(to_bins_two_bias, /*dim=*/1),
                           to_bins_two_weight,
                           out_one);

  // [bins, batch_size] → [batch_size, bins]
  auto out_transposed = at::transpose(out_two, 0, 1);

  at::Tensor sample;
  if (argmax)
  {
    // [batch_size, bins] → [batch_size]
    sample = std::get<1>(at::max(out_transposed, /*dim=*/1));
  }
  else
  {
    auto posterior = at::softmax(out_transposed / temperature, /*dim=*/1);
    // [batch_size, bins] → [batch_size, 1]
    sample = at::multinomial(posterior, /*num_samples=*/1);
    sample = at::squeeze(sample, /*dim=*/1);
  }

  return {last_hidden_state, sample};
}

std::vector<at::Tensor> inference(
    at::Tensor last_coarse,
    at::Tensor last_fine,
    at::Tensor last_coarse_hidden_state,
    at::Tensor last_fine_hidden_state,
    at::Tensor project_coarse_bias,
    at::Tensor project_coarse_weight,
    at::Tensor project_fine_bias,
    at::Tensor project_fine_weight,
    at::Tensor project_hidden_bias,
    at::Tensor project_hidden_weight,
    at::Tensor to_bins_coarse_in_bias,
    at::Tensor to_bins_coarse_in_weight,
    at::Tensor to_bins_coarse_out_bias,
    at::Tensor to_bins_coarse_out_weight,
    at::Tensor to_bins_fine_in_bias,
    at::Tensor to_bins_fine_in_weight,
    at::Tensor to_bins_fine_out_bias,
    at::Tensor to_bins_fine_out_weight,
    bool argmax = false,
    float temperature = 1.0)
{
  /** Run the Wave RNN inference algorithm.
   *
   * @param last_coarse [batch_size] Last predicted coarse value.
   * @param last_fine [batch_size] Last predicted fine value.
   * @param last_coarse_hidden_state [hidden_size / 2, batch_size] Last corase RNN state.
   * @param last_fine_hidden_state [hidden_size / 2, batch_size] Last fine RNN state.
   * @param project_coarse_bias [1.5 * hidden_size, signal_length, batch_size] Bias for coarse
   *    projection.
   * @param project_coarse_weight [1.5 * hidden_size, 2] Weights for coarse projection.
   * @param project_fine_bias [1.5 * hidden_size, signal_length, batch_size] Bias for fine
   *    projection.
   * @param project_fine_weight [1.5 * hidden_size, 3] Weights for coarse projection.
   * @param project_hidden_bias [3 * hidden_size] Bias for hidden projection.
   * @param project_hidden_weight [3 * hidden_size, hidden_size] Weights for hidden projection.
   * @param to_bins_coarse_in_bias [hidden_size / 2] Input dense layer bias to compute
   *    coarse categorical distribution.
   * @param to_bins_coarse_in_weight [hidden_size / 2, hidden_size / 2] Input dense layer
   *    weights to compute coarse categorical distribution.
   * @param to_bins_coarse_out_bias [bins] Ouput dense layer bias to compute coarse categorical
   *    distribution.
   * @param to_bins_coarse_out_weight [bins, hidden_size / 2] Ouput dense layer weights to compute
   *    coarse categorical distribution.
   * @param to_bins_fine_in_bias [hidden_size / 2] Input dense layer bias to compute
   *    fine categorical distribution.
   * @param to_bins_fine_in_weight [hidden_size / 2, hidden_size / 2] Input dense layer
   *    weights to compute fine categorical distribution.
   * @param to_bins_fine_out_bias [bins] Ouput dense layer bias to compute fine categorical
   *    distribution.
   * @param to_bins_fine_out_weight [bins, hidden_size / 2] Ouput dense layer weights to compute
   *    fine categorical distribution.
   * @return coarse_signal [batch_size, signal_length] Predicted signal coarse bits.
   * @return fine_signal [batch_size, signal_length] Predicted signal fine bits.
   * @return last_coarse_hidden_state [hidden_size / 2, batch_size] Last corase RNN state.
   * @return last_fine_hidden_state [hidden_size / 2, batch_size] Last fine RNN state.
   */
  auto signal_length = project_coarse_bias.size(1);
  auto hidden_size = last_coarse_hidden_state.size(0) * 2;
  auto bins = to_bins_coarse_out_bias.size(0);

  auto coarse_signal = std::vector<at::Tensor>(signal_length);
  auto fine_signal = std::vector<at::Tensor>(signal_length);

  for (int i = 0; i < signal_length; ++i)
  {
    // {[hidden_size / 2, batch_size], [hidden_size / 2, batch_size]} → [size, batch_size]
    auto hidden_state = at::cat({last_coarse_hidden_state, last_fine_hidden_state}, /*dim=*/0);
    // [3 * hidden_size] + [3 * hidden_size, hidden_size] * [hidden_size, batch_size] →
    // [3 * hidden_size, batch_size]
    auto hidden_projection = at::addmm(project_hidden_bias, project_hidden_weight, hidden_state);

    // {[1, batch_size], [1, batch_size]} → [2, batch_size]
    auto coarse_input = at::stack({last_coarse, last_fine}, /*dim=*/0);
    auto coarse_input_scaled = (coarse_input.toType(at::ScalarType::Float) /
                                    ((bins - 1.0) / 2.0) -
                                1.0);

    // [3 * hidden_size, batch_size] → [1.5 * hidden_size, batch_size]
    auto hidden_projection_coarse = at::slice(hidden_projection,
                                              /*dim=*/0,
                                              /*start=*/0,
                                              /*end=*/hidden_size * 3 / 2);
    auto coarse_step = step(last_coarse_hidden_state,
                            coarse_input_scaled,
                            project_coarse_bias.select(/*dim=*/1, /*index=*/i),
                            project_coarse_weight,
                            hidden_projection_coarse,
                            to_bins_coarse_in_bias,
                            to_bins_coarse_in_weight,
                            to_bins_coarse_out_bias,
                            to_bins_coarse_out_weight,
                            argmax,
                            temperature);
    last_coarse_hidden_state = coarse_step[0]; // [hidden_size / 2, batch_size]
    coarse_signal[i] = coarse_step[1];         // [batch_size]
    last_coarse = coarse_step[1];              // [batch_size]

    // {[2, batch_size], [1, batch_size]} → [3, batch_size]
    auto fine_input = at::cat({coarse_input, at::unsqueeze(last_coarse, /*dim=*/0)}, /*dim=*/0);
    auto fine_input_scaled = fine_input.toType(at::ScalarType::Float) / ((bins - 1.0) / 2.0) - 1.0;

    // [3 * hidden_size, batch_size] → [1.5 * hidden_size, batch_size]
    auto hidden_projection_fine = at::slice(hidden_projection,
                                            /*dim=*/0,
                                            /*start=*/hidden_size * 3 / 2,
                                            /*end=*/hidden_size * 3);
    auto fine_step = step(last_fine_hidden_state,
                          fine_input_scaled,
                          project_fine_bias.select(/*dim=*/1, /*index=*/i),
                          project_fine_weight,
                          hidden_projection_fine,
                          to_bins_fine_in_bias,
                          to_bins_fine_in_weight,
                          to_bins_fine_out_bias,
                          to_bins_fine_out_weight,
                          argmax,
                          temperature);
    last_fine_hidden_state = fine_step[0]; // [hidden_size / 2, batch_size]
    fine_signal[i] = fine_step[1];         // [batch_size]
    last_fine = fine_step[1];              // [batch_size]
  }

  return {at::stack(coarse_signal, /*dim=*/1),
          at::stack(fine_signal, /*dim=*/1),
          last_coarse_hidden_state,
          last_fine_hidden_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("run", &inference, "Run inference for WaveRNN.");
}
