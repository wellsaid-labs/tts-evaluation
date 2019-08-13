#include <sleef.h>
#include <iostream>
#include <random>
#include <vector>

// NOTE: PyTorch's TensorIterator provides a nice "just stride / size + a pointer" interface.
#include <ATen/core/TensorAccessor.h>

// NOTE: Resolve PyTorch 1.1 vs. 1.2 compatibility.
#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

using torch::Tensor;

using TA1D = at::TensorAccessor<float, 1>;
using TA2D = at::TensorAccessor<float, 2>;

inline float sum8(__m256 x)
{
  /** Sums the 8 elements of x into a single float.
   *
   * @param x __m256 of 8 floats to be summed
   *          x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
   * @return sum x
   */
  // hiQuad = ( x7, x6, x5, x4 )
  const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
  // loQuad = ( x3, x2, x1, x0 )
  const __m128 loQuad = _mm256_castps256_ps128(x);
  // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
  const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  // loDual = ( -, -, x1 + x5, x0 + x4 )
  const __m128 loDual = sumQuad;
  // hiDual = ( -, -, x3 + x7, x2 + x6 )
  const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
  const __m128 sumDual = _mm_add_ps(loDual, hiDual);
  // lo = ( -, -, -, x0 + x2 + x4 + x6 )
  const __m128 lo = sumDual;
  // hi = ( -, -, -, x1 + x3 + x5 + x7 )
  const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
  const __m128 sum = _mm_add_ss(lo, hi);
  return _mm_cvtss_f32(sum);
}

#ifdef NO_MKL
inline void torch_addmv_out(Tensor &hidden_projection, const Tensor &project_hidden_bias,
                            const Tensor &project_hidden_weight, const Tensor &hidden_state)
{
  /** Computes `hidden_projection = project_hidden_bias + project_hidden_weight @ hidden_state`
   * (`@` represents a matrix-vector product) using with PyTorch.
   *
   * @param (output) hidden_projection [n] 1d output
   * @param project_hidden_bias [n]
   * @param project_hidden_weight [n, k]
   * @param hidden_state [k]
   */
  // [3 * hidden_size] + [3 * hidden_size, hidden_size] * [hidden_size] → [3 *
  // hidden_size]
  hidden_projection.copy_(torch::addmv(project_hidden_bias, project_hidden_weight, hidden_state));
}
#else
#include <mkl/mkl.h>
inline void addmv_out(TA1D &res, const TA1D &b, const TA2D &m, const TA1D &v)
{
  /** Computes res = b + m @ v  (@ = matrix-vector product) using MKL.
   *
   * @param (output) res [n] 1d output
   * @param b [n]
   * @param m [n, k]
   * @param v [k]
   */
  cblas_scopy(m.size(0), b.data(), 1, res.data(), 1);
  cblas_sgemv(CblasRowMajor, CblasNoTrans, m.size(0), m.size(1), 1.0, m.data(),
              /*leading dim=*/m.size(1), v.data(), 1, 1.0, res.data(), 1);
}
#endif

constexpr size_t m256_size = 8; /* 8 floats in __m256 */

inline void addmv_relu_out(TA1D &res, const TA1D &b, const TA2D &m,
                           const TA1D &v)
{
  /** Computes res = relu(b + m.t() @ v)  (@ = matrix-vector product)
   *
   * @param (output) res [n] 1d output
   * @param b [n]
   * @param m [k, n] NOTE: the n-stride is assumed to be contiguous!
   * @param v [k]
   */
  float *lh_data = v.data();
  float *b1_data = b.data();
  float *w1_data = m.data();
  float *r1_data = res.data();

  int64_t h_size = v.size(0);
  int64_t b_size = b.size(0);

#pragma omp parallel for
  for (int64_t i = 0; i < b_size; i += 4 * m256_size)
  {
    __m256 racc0 = _mm256_load_ps(b1_data + i + 0 * m256_size);
    __m256 racc1 = _mm256_load_ps(b1_data + i + 1 * m256_size);
    __m256 racc2 = _mm256_load_ps(b1_data + i + 2 * m256_size);
    __m256 racc3 = _mm256_load_ps(b1_data + i + 3 * m256_size);
    for (int64_t j = 0; j < h_size; j += 1)
    {
      __m256 lh0 = _mm256_set1_ps(lh_data[j]);
      __m256 w1_0_0 = _mm256_load_ps(w1_data + j * b_size + i + 0 * m256_size);
      __m256 w1_1_0 = _mm256_load_ps(w1_data + j * b_size + i + 1 * m256_size);
      __m256 w1_2_0 = _mm256_load_ps(w1_data + j * b_size + i + 2 * m256_size);
      __m256 w1_3_0 = _mm256_load_ps(w1_data + j * b_size + i + 3 * m256_size);

      racc0 = _mm256_fmadd_ps(lh0, w1_0_0, racc0);
      racc1 = _mm256_fmadd_ps(lh0, w1_1_0, racc1);
      racc2 = _mm256_fmadd_ps(lh0, w1_2_0, racc2);
      racc3 = _mm256_fmadd_ps(lh0, w1_3_0, racc3);
    }
    __m256 zero = _mm256_set1_ps(0.);
    racc0 = _mm256_max_ps(racc0, zero);
    racc1 = _mm256_max_ps(racc1, zero);
    racc2 = _mm256_max_ps(racc2, zero);
    racc3 = _mm256_max_ps(racc3, zero);
    _mm256_store_ps(r1_data + i + 0 * m256_size, racc0);
    _mm256_store_ps(r1_data + i + 1 * m256_size, racc1);
    _mm256_store_ps(r1_data + i + 2 * m256_size, racc2);
    _mm256_store_ps(r1_data + i + 3 * m256_size, racc3);
  }
}

inline float addmv_exp_out(TA1D &res, const TA1D &b, const TA2D &m,
                           const TA1D &v)
{
  /** Computes res = exp(b + m.t() @ v)  (@ = matrix-vector product)
   *
   * @param (output) res [n] 1d output
   * @param b [n]
   * @param m [k, t] NOTE: the n-stride is assumed to be contiguous!
   * @param v [k]
   * @return sum of res
   */
  float *lh_data = v.data();
  float *b1_data = b.data();
  float *w1_data = m.data();
  float *r1_data = res.data();

  int64_t h_size = v.size(0);
  int64_t b_size = b.size(0);
  float the_sum = 0;
#pragma omp parallel for reduction(+ \
                                   : the_sum)
  for (int64_t i = 0; i < b_size; i += 4 * (256 / 32) /*m256_size*/)
  {
    __m256 racc0 = _mm256_load_ps(b1_data + i + 0 * m256_size);
    __m256 racc1 = _mm256_load_ps(b1_data + i + 1 * m256_size);
    __m256 racc2 = _mm256_load_ps(b1_data + i + 2 * m256_size);
    __m256 racc3 = _mm256_load_ps(b1_data + i + 3 * m256_size);
    for (int64_t j = 0; j < h_size; j += 1)
    {
      __m256 lh0 = _mm256_set1_ps(lh_data[j]);
      __m256 w1_0_0 = _mm256_load_ps(w1_data + j * b_size + i + 0 * m256_size);
      __m256 w1_1_0 = _mm256_load_ps(w1_data + j * b_size + i + 1 * m256_size);
      __m256 w1_2_0 = _mm256_load_ps(w1_data + j * b_size + i + 2 * m256_size);
      __m256 w1_3_0 = _mm256_load_ps(w1_data + j * b_size + i + 3 * m256_size);

      racc0 = _mm256_fmadd_ps(lh0, w1_0_0, racc0);
      racc1 = _mm256_fmadd_ps(lh0, w1_1_0, racc1);
      racc2 = _mm256_fmadd_ps(lh0, w1_2_0, racc2);
      racc3 = _mm256_fmadd_ps(lh0, w1_3_0, racc3);
    }
    racc0 = Sleef_expf8_u10(racc0);
    racc1 = Sleef_expf8_u10(racc1);
    racc2 = Sleef_expf8_u10(racc2);
    racc3 = Sleef_expf8_u10(racc3);
    _mm256_store_ps(r1_data + i + 0 * m256_size, racc0);
    _mm256_store_ps(r1_data + i + 1 * m256_size, racc1);
    _mm256_store_ps(r1_data + i + 2 * m256_size, racc2);
    _mm256_store_ps(r1_data + i + 3 * m256_size, racc3);
    racc0 = racc0 + racc1 + racc2 + racc3;
    // it would be better to split out the thread parallelism from the for
    // loop...
    the_sum += sum8(racc0);
  }
  return the_sum;
}

float addmv_exp_out_wrapper(Tensor &res, const Tensor &b,
                            const Tensor &m, const Tensor &v)
{
  /** Computes res = relu(b + m.t() @ v)  (@ = matrix-vector product)
   *  Wrapper for Python binding
   *
   * @param (output) res [n] 1d output
   * @param b [n]
   * @param m [k, k] NOTE: the n-stride is assumed to be contiguous!
   * @param v [k]
   */
  auto res_a = res.accessor<float, 1>();
  return addmv_exp_out(res_a, b.accessor<float, 1>(), m.accessor<float, 2>(),
                       v.accessor<float, 1>());
}

static std::default_random_engine generator;
static std::uniform_real_distribution<double> distribution(0.0, 1.0);

int64_t sample_multinomial(const TA1D &unnorm_probs, float sum)
{
  /** Samples from multinomial distribution.
   *
   * @param unnorm_probs [bins] unnormed probability vector
   * @param sum sum of unnorm_probs
   * @return random integer 0..bins-1, with sampled with probabilities
   *         given by unnormed_probs / sum
   */
  float cumsum = 0;
  float r = distribution(generator);
  r *= sum;
  float *probs = unnorm_probs.data();
  auto n_bins = unnorm_probs.size(0);
  for (int64_t i = 0; i < n_bins; i++)
  {
    cumsum += probs[i];
    if (cumsum > r)
    {
      return i;
    }
  }
  return n_bins - 1;
}

int64_t sample_multinomial_wrapper(const Tensor &unnorm_probs, float sum)
{
  /** Samples from multinomial distribution.
   *  Python wrapper.
   *
   * @param unnorm_probs [bins] unnormed probability vector
   * @param sum sum of unnorm_probs
   * @return random integer 0..bins-1, with sampled with probabilities
   *         given by unnormed_probs / sum
   */
  return sample_multinomial(unnorm_probs.accessor<float, 1>(), sum);
}

int64_t get_argmax(const TA1D &x)
{
  /** Argmax of x.
   *
   * @param x [n] vector, must not be empty
   * @return i with x[i] maximal
   */
  int64_t maxidx = 0;
  float max = x[0];
  auto n_bins = x.size(0);
  for (int64_t i = 1; i < n_bins; i++)
  {
    if (x[i] > max)
    {
      max = x[i];
      maxidx = i;
    }
  }
  return maxidx;
}

inline __m256 sigmoid(__m256 a)
{
  /** __m256torized y = 1/(1+exp(-x))
   *
   * @param x __m256
   * @return y = 1/(1+exp(-x))
   */
  // a = x
  a = _mm256_set1_ps(0) - a;               // a = -x
  a = Sleef_expf8_u10(a);                  // a = exp(-x)
  a = _mm256_set1_ps(1) + a;               // a = 1 + exp(-x)
  a = _mm256_div_ps(_mm256_set1_ps(1), a); // a = 1/(1 + exp(-x))
  return a;
}

#ifndef __AVX2__
// safety catch
#error "need avx2";
#endif

// simple, low-overhead profiling facility
// map with cumulated timings
static std::unordered_map<std::string, int64_t> timings;

// Gets the map of function and timings. This can be wrapped by PyBind.
const std::unordered_map<std::string, int64_t> &get_timings()
{
  return timings;
}

int64_t get_time()
{
  // gets the time in nanoseconds. Linux specific.
  struct timespec t
  {
  };
  clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<int64_t>(t.tv_sec) * 1000000000 +
         static_cast<int64_t>(t.tv_nsec);
}

#ifdef DO_PROFILE

// Profiler is an internal class for profiling to enable the PROFILE macro
class Profiler
{
  int64_t start_t;
  const char *name_;

public:
  Profiler(const char *name) : name_(name) { start_t = get_time(); }
  ~Profiler()
  {
    struct timespec t
    {
    };
    clock_gettime(CLOCK_MONOTONIC, &t);
    auto dur = get_time() - start_t;
    auto it = timings.find(name_);
    if (it == timings.end())
    {
      timings.emplace(name_, dur);
    }
    else
    {
      it->second += dur;
    }
  }
};

// Profiling macro. Use as (for void returns)
// PROFILE(name, myfunction(something));
// or (with return value)
// auto res = PROFILE(name, myfunction(something));

#define PROFILE(name, ...) \
  [&] {                    \
    Profiler prof(name);   \
    return __VA_ARGS__;    \
  }()
#else
#define PROFILE(name, ...) [&] { return __VA_ARGS__; }()
#endif

inline void gate_calc_(const TA1D &input, const TA1D &hidden_projection,
                       const TA1D &last_hidden,
                       const TA2D &input_projection_weight,
                       const TA2D &input_projection_bias, int64_t i)
{
  /** Run fused GRU gate calculations. Requires hidden_size to be divisible
   by 8.
   *  With blocks being 1/3 of the size = 0.5 hidden_size
   *  ri = sigmoid(input_projection_weight[block0] @ input +
   input_projection_bias[i][block0] + hidden_projection[block0])
   *  ig = sigmoid(input_projection_weight[block1] @ input +
   input_projection_bias[i][block1] + hidden_projection[block1])
   *  ng =    tanh(input_projection_weight[block1] @ input +
   input_projection_bias[i][block1] + ri*(hidden_projection[block1]))
   *  last_hidden = ig * (lh - ng) + ng (= ig * lh + (1 - ig) * ng))
   *
   * @param input [input_size] Input to RNN. (We have 2 or 3 in mind in terms of
   optimization)
   * @param hidden_projection [1.5 * hidden_size] Hidden state projected to
   *    ``1.5 * hidden_size`` (W_h* hiden_state + b_h*).
   * @param last_hidden [hidden_size / 2] Last RNN state.
   *        Will be overwritten with new state.
   * @param project_input_bias [seq_len, 1.5 * hidden_size] Bias to project
   input for RNN. only the i-th row is used
   * @param project_input_weight [1.5 * hidden_size, input_size] Weights to
   *        project input for RNN.
   * @param i index into input_bias.
   */

  int64_t size = last_hidden.size(0);

  float *input_data = input.data();
  int64_t input_size = input.size(0);
  float *ipb_data =
      input_projection_bias.data() + input_projection_bias.stride(0) * i;
  float *ipw_data = input_projection_weight.data();
  float *hp_data = hidden_projection.data();
  float *lh_data = last_hidden.data();
  for (int64_t d = 0; d < size; d += m256_size)
  {
    __m256 ip0 = _mm256_load_ps(ipb_data + d);
    __m256 ip1 = _mm256_load_ps(ipb_data + d + size);
    __m256 ip2 = _mm256_load_ps(ipb_data + d + 2 * size);
    for (int64_t i = 0; i < input_size; i++)
    {
      __m256 inp = _mm256_set1_ps(input_data[i]);
      __m256 ipw0 = _mm256_load_ps(ipw_data + d + i * 3 * size);
      __m256 ipw1 = _mm256_load_ps(ipw_data + d + size + i * 3 * size);
      __m256 ipw2 = _mm256_load_ps(ipw_data + d + 2 * size + i * 3 * size);
      ip0 = _mm256_fmadd_ps(ipw0, inp, ip0);
      ip1 = _mm256_fmadd_ps(ipw1, inp, ip1);
      ip2 = _mm256_fmadd_ps(ipw2, inp, ip2);
    }
    __m256 hp0 = _mm256_load_ps(hp_data + d);
    __m256 ri0 = sigmoid(ip0 + hp0);

    __m256 hp1 = _mm256_load_ps(hp_data + d + size);
    __m256 ig = sigmoid(ip1 + hp1);

    __m256 hp2 = _mm256_load_ps(hp_data + d + 2 * size);
    __m256 ng = Sleef_tanhf8_u10(_mm256_fmadd_ps(ri0, hp2, ip2));

    __m256 lh = _mm256_load_ps(lh_data + d);
    __m256 h = _mm256_fmadd_ps(ig, (lh - ng), ng);

    _mm256_store_ps(lh_data + d, h);
  }
}

int64_t step(TA1D &last_hidden_state, const TA1D &input,
             const TA2D &project_input_bias, const TA2D &project_input_weight,
             const TA1D &hidden_projection, const TA1D &to_bins_one_bias,
             const TA2D &to_bins_one_weight, const TA1D &to_bins_two_bias,
             const TA2D &to_bins_two_weight, TA1D &out_one, TA1D &out_two,
             int64_t i_step, bool argmax = false)
{
  /** Run one step of the Wave RNN inference algorithm.
   *
   * Side effects:
   * - `last_hidden_state` is modified in-place to be the updated hidden state.
   *
   * @param last_hidden_state [hidden_size / 2] Last RNN state.
   * @param input [input_size] Input to RNN.
   * @param project_input_bias [seq_len, 1.5 * hidden_size] Bias to project
   input for RNN. only the i-th row is used
   * @param project_input_weight [input_size, 1.5 * hidden_size] Weights to
   project input for RNN.
   * @param hidden_projection [1.5 * hidden_size] Hidden state projected to
   *    ``1.5 * hidden_size``.
   * @param to_bins_one_bias [hidden_size / 2] Input dense layer bias to compute
   *    categorical distribution.
   * @param to_bins_one_weight [hidden_size / 2, hidden_size / 2] Input dense
   layer
   *    weights to compute categorical distribution.
   * @param to_bins_two_bias [bins] Ouput dense layer bias to compute
   categorical
   *    distribution.
   * @param to_bins_two_weight [bins, hidden_size / 2] Ouput dense layer weights
   to compute
   *    categorical distribution.
   * @param out_one [hidden_size / 2] Working memory
   * @param out_two [bins]  Working memory
   * @param i (int64_t) Step index for input bias.
   * @param argmax (bool) If ``True``, during inference sample the most likely
   sample.
   * @return sample (int64_t) sampled class.
   */

  PROFILE("gate_calc",
          gate_calc_(input, hidden_projection, last_hidden_state,
                     project_input_weight, project_input_bias, i_step));

  // [hidden_size / 2] + [hidden_size / 2, hidden_size / 2] * [hidden_size / 2]
  // → [hidden_size / 2]
  PROFILE("addmv_out_one_alt",
          addmv_relu_out(out_one, to_bins_one_bias, to_bins_one_weight,
                         last_hidden_state));

  // [bins] + [bins, hidden_size / 2] * [hidden_size / 2] →
  // [bins]
  float sum =
      PROFILE("addmv_out_two", addmv_exp_out(out_two, to_bins_two_bias,
                                             to_bins_two_weight, out_one));
  if (argmax)
  {
    // [bins] → [1]
    return get_argmax(out_two);
  }
  else
  {
    return PROFILE("sample", sample_multinomial(out_two, sum));
  }
}

void check_aligned(const torch::ArrayRef<Tensor> ts)
{
  // checks alignment requirements for data
  for (const auto &t : ts)
  {
    TORCH_CHECK((((uintptr_t)t.data<float>()) & 0x1f) == 0, "Tensor is not aligned to 32-byte boundaries.");
    TORCH_CHECK(t.is_contiguous(), "Tensor is not contiguous.");
  }
}

std::vector<Tensor> inference(
    Tensor last_coarse, Tensor last_fine,
    Tensor hidden_state, Tensor project_coarse_bias,
    Tensor project_coarse_weight, Tensor project_fine_bias,
    Tensor project_fine_weight, Tensor project_hidden_bias,
    Tensor project_hidden_weight, Tensor to_bins_coarse_in_bias,
    Tensor to_bins_coarse_in_weight,
    Tensor to_bins_coarse_out_bias,
    Tensor to_bins_coarse_out_weight, Tensor to_bins_fine_in_bias,
    Tensor to_bins_fine_in_weight, Tensor to_bins_fine_out_bias,
    Tensor to_bins_fine_out_weight, bool argmax = false)
{
  /** Run the Wave RNN inference algorithm.
   *
   * NOTE:
   * - All inputs must be aligned to 32 byte (8 float) boundaries.
   * - `hidden_size` should be divisible by 32.
   * - All inputs need to be contiguous.
   * - PyTorch does this by default when you allocate tensors.
   *
   * @param last_coarse [1] Last predicted coarse value.
   * @param last_fine [1] Last predicted fine value.
   * @param hidden_state [hidden_size] Last RNN hidden state.
   * @param project_coarse_bias [sequence_length, 1.5 * hidden_size] Bias for
   * coarse projection.
   * @param project_coarse_weight [2, 1.5 * hidden_size] Weights for coarse
   * projection.
   * @param project_fine_bias [sequence_length, 1.5 * hidden_size] Bias for fine
   *    projection.
   * @param project_fine_weight [3, 1.5 * hidden_size] Weights for coarse
   * projection.
   * @param project_hidden_bias [3 * hidden_size] Bias for hidden projection.
   * @param project_hidden_weight [3 * hidden_size, hidden_size] Weights for
   * hidden projection.
   * @param to_bins_coarse_in_bias [hidden_size / 2] Input dense layer bias to
   * compute coarse categorical distribution.
   * @param to_bins_coarse_in_weight [hidden_size / 2, hidden_size / 2] Input
   * dense layer weights to compute coarse categorical distribution.
   * @param to_bins_coarse_out_bias [bins] Ouput dense layer bias to compute
   * coarse categorical distribution.
   * @param to_bins_coarse_out_weight [hidden_size / 2, bins] Ouput dense layer
   * weights to compute coarse categorical distribution.
   * @param to_bins_fine_in_bias [hidden_size / 2] Input dense layer bias to
   * compute fine categorical distribution.
   * @param to_bins_fine_in_weight [hidden_size / 2, hidden_size / 2] Input
   * dense layer weights to compute fine categorical distribution.
   * @param to_bins_fine_out_bias [bins] Ouput dense layer bias to compute fine
   * categorical distribution.
   * @param to_bins_fine_out_weight [hidden_size / 2, bins] Ouput dense layer
   * weights to compute fine categorical distribution.
   * @param argmax (bool) If ``True``, during inference sample the most likely
   * sample.
   * @return coarse_signal [sequence_length] Predicted signal coarse bits.
   * @return fine_signal [sequence_length] Predicted signal fine bits.
   * @return hidden_state [hidden_size] Last RNN hidden state.
   */
  auto start_time = get_time();
  auto sequence_length = project_coarse_bias.size(0);
  auto hidden_size = hidden_state.size(0);
  auto bins = to_bins_coarse_out_bias.size(0);

  auto opt = torch::TensorOptions().dtype(at::kLong);
  auto coarse_signal = torch::zeros(sequence_length, opt);
  auto fine_signal = torch::zeros(sequence_length, opt);
  int64_t *coarse_signal_data = coarse_signal.data<int64_t>();
  int64_t *fine_signal_data = fine_signal.data<int64_t>();

  auto inverse_bins = 1.0 / ((bins - 1.0) / 2.0);
  auto input = torch::zeros({3}); // [last_coarse, last_fine, coarse]
  float *input_data = input.data<float>();
  input[0] = last_coarse[0];
  input[1] = last_fine[0];
  input.mul_(inverse_bins).sub_(1);

  check_aligned(
      {hidden_state, project_coarse_bias, project_coarse_weight,
       project_fine_bias, project_fine_weight, project_hidden_bias,
       project_hidden_weight, to_bins_coarse_in_bias, to_bins_coarse_in_weight,
       to_bins_coarse_out_bias, to_bins_coarse_out_weight, to_bins_fine_in_bias,
       to_bins_fine_in_weight, to_bins_fine_out_bias, to_bins_fine_out_weight});

  auto out_one_work = torch::empty_like(to_bins_coarse_in_bias);
  auto out_two_work = torch::empty_like(to_bins_coarse_out_bias);

  // [3] → [2]
  auto coarse_input_scaled = torch::slice(input,
                                          /*dim=*/0,
                                          /*start=*/0,
                                          /*end=*/2);

  // [hidden_size] → [hidden_size / 2]
  auto last_coarse_hidden_state = torch::slice(hidden_state,
                                               /*dim=*/0,
                                               /*start=*/0,
                                               /*end=*/hidden_size / 2);
  auto hidden_projection = torch::empty({3 * hidden_size});
  // [3 * hidden_size] → [1.5 * hidden_size]
  auto hidden_projection_coarse = torch::slice(hidden_projection,
                                               /*dim=*/0,
                                               /*start=*/0,
                                               /*end=*/hidden_size * 3 / 2);
  // [3 * hidden_size] → [1.5 * hidden_size]
  auto hidden_projection_fine = torch::slice(hidden_projection,
                                             /*dim=*/0,
                                             /*start=*/hidden_size * 3 / 2,
                                             /*end=*/hidden_size * 3);
  // [hidden_size] → [hidden_size / 2]
  auto last_fine_hidden_state = torch::slice(hidden_state,
                                             /*dim=*/0,
                                             /*start=*/hidden_size / 2,
                                             /*end=*/hidden_size);

  auto hidden_projection_a = hidden_projection.accessor<float, 1>();
  auto project_hidden_bias_a = project_hidden_bias.accessor<float, 1>();
  auto project_hidden_weight_a = project_hidden_weight.accessor<float, 2>();
  auto hidden_state_a = hidden_state.accessor<float, 1>();
  auto last_fine_hidden_state_a = last_fine_hidden_state.accessor<float, 1>();
  auto input_a = input.accessor<float, 1>();
  auto project_fine_bias_a = project_fine_bias.accessor<float, 2>();
  auto project_fine_weight_a = project_fine_weight.accessor<float, 2>();
  auto hidden_projection_fine_a = hidden_projection_fine.accessor<float, 1>();
  auto to_bins_fine_in_bias_a = to_bins_fine_in_bias.accessor<float, 1>();
  auto to_bins_fine_in_weight_a = to_bins_fine_in_weight.accessor<float, 2>();
  auto to_bins_fine_out_bias_a = to_bins_fine_out_bias.accessor<float, 1>();
  auto to_bins_fine_out_weight_a = to_bins_fine_out_weight.accessor<float, 2>();
  auto last_coarse_hidden_state_a =
      last_coarse_hidden_state.accessor<float, 1>();
  auto coarse_input_scaled_a = coarse_input_scaled.accessor<float, 1>();
  auto project_coarse_bias_a = project_coarse_bias.accessor<float, 2>();
  auto project_coarse_weight_a = project_coarse_weight.accessor<float, 2>();
  auto hidden_projection_coarse_a =
      hidden_projection_coarse.accessor<float, 1>();
  auto to_bins_coarse_in_bias_a = to_bins_coarse_in_bias.accessor<float, 1>();
  auto to_bins_coarse_in_weight_a =
      to_bins_coarse_in_weight.accessor<float, 2>();
  auto to_bins_coarse_out_bias_a = to_bins_coarse_out_bias.accessor<float, 1>();
  auto to_bins_coarse_out_weight_a =
      to_bins_coarse_out_weight.accessor<float, 2>();
  auto out_one_work_a = out_one_work.accessor<float, 1>();
  auto out_two_work_a = out_two_work.accessor<float, 1>();

  for (int i = 0; i < sequence_length; ++i)
  {
    // [3 * hidden_size] + [3 * hidden_size, hidden_size] * [hidden_size] → [3 *
    // hidden_size]
#ifdef NO_MKL
    PROFILE("addmv_hidden_projection",
            torch_addmv_out(hidden_projection, project_hidden_bias,
                            project_hidden_weight, hidden_state));
#else
    PROFILE("addmv_hidden_projection",
            addmv_out(hidden_projection_a, project_hidden_bias_a,
                      project_hidden_weight_a, hidden_state_a));
#endif

    int64_t coarse_sample = step(
        last_coarse_hidden_state_a, coarse_input_scaled_a,
        project_coarse_bias_a, project_coarse_weight_a,
        hidden_projection_coarse_a, to_bins_coarse_in_bias_a,
        to_bins_coarse_in_weight_a, to_bins_coarse_out_bias_a,
        to_bins_coarse_out_weight_a, out_one_work_a, out_two_work_a, i, argmax);
    input_data[2] = static_cast<float>(coarse_sample) * inverse_bins - 1.0;
    coarse_signal_data[i] = coarse_sample;

    int64_t fine_sample = step(
        last_fine_hidden_state_a, input_a, project_fine_bias_a,
        project_fine_weight_a, hidden_projection_fine_a, to_bins_fine_in_bias_a,
        to_bins_fine_in_weight_a, to_bins_fine_out_bias_a,
        to_bins_fine_out_weight_a, out_one_work_a, out_two_work_a, i, argmax);
    fine_signal_data[i] = fine_sample;
    input_data[1] = static_cast<float>(fine_sample) * inverse_bins - 1.0;
    input_data[0] = input_data[2];
  }

  return {coarse_signal, fine_signal, hidden_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("run", &inference, "Run inference for WaveRNN.");
  m.def("addmv_exp_out", &addmv_exp_out_wrapper, "fused addmv and exp");
  m.def("sample_multinomial", &sample_multinomial_wrapper, "multinomial sample");
  m.def("get_timings", &get_timings, "get timings from profiler");
}