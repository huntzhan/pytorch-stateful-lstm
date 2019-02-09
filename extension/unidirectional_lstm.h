#ifndef EXTENSION_UNIDIRECTIONAL_LSTM_H_
#define EXTENSION_UNIDIRECTIONAL_LSTM_H_

#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <string>

namespace cnt {

using LstmStateType = std::tuple<torch::Tensor, torch::Tensor>;
using LstmForwardRetType = std::tuple<torch::Tensor, LstmStateType>;
using LstmForwardMultiLayerRetType = std::tuple<
    std::vector<torch::Tensor>, LstmStateType>;

struct UnidirectionalSingleLayerLstmImpl : torch::nn::Module {
  UnidirectionalSingleLayerLstmImpl(
      // Define the structure.
      int64_t input_size,
      int64_t hidden_size,
      int64_t cell_size,

      // `ture` for forward and `false` for backward.
      bool go_forward,
      // Tensorflow-style truncated BPTT.
      int64_t truncated_bptt,

      // Clip on the cell/hidden states.
      double cell_clip,
      double proj_clip,

      // `recurrent_dropout_type`:
      //   - **0**: No dropout.
      //   - **1**: Variational dropout.
      //   - **2**: Dropconnect.
      int64_t recurrent_dropout_type,
      double recurrent_dropout_probability);

  // Inputs: inputs, batch_sizes, (h_0, c_0)
  //   - **inputs** of shape `(*, input_size)`:
  //     tensor of `PackedSequence.data`.
  //   - **batch_sizes** of shape `(total_timesteps,)`:
  //     tensor of `PackedSequence.batch_sizes`.
  //   - **h_0** of shape `(1, {x | x >= batch}, hidden_size)`.
  //     tensor containing the intial hidden state of LSTM.
  //   - **c_0** of shape `(1, {x | x >= batch}, cell_size)`.
  //     tensor containing the intial cell state of LSTM.
  //
  // Outputs: output, (h_1, c_1)
  //   - **output** of shape
  //     `(*, hidden_size)`:
  //     tensor containing output features and should be used
  //     to initialize `PackedSequence` along with `batch_sizes`.
  //   - **h_1** of shape `(1, {x | x >= batch}, hidden_size)`.
  //     tensor containing the final hidden state of LSTM.
  //   - **c_1** of shape `(1, {x | x >= batch}, cell_size)`.
  //     tensor containing the final cell state of LSTM.
  LstmForwardRetType forward(
      torch::Tensor inputs,
      torch::Tensor batch_sizes,
      LstmStateType initial_state);

  torch::TensorOptions weight_options();

  int64_t input_size_ = -1;
  int64_t hidden_size_ = -1;
  int64_t cell_size_ = -1;

  bool go_forward_ = true;
  int64_t truncated_bptt_ = 0;

  double cell_clip_ = 0.0;
  double proj_clip_ = 0.0;

  int64_t recurrent_dropout_type_ = -1;
  double recurrent_dropout_probability_ = 0.0;

  // Packed linear projections.
  torch::Tensor input_linearity_weight_{};
  torch::Tensor hidden_linearity_weight_{};
  torch::Tensor hidden_linearity_bias_{};
  torch::Tensor proj_linearity_weight_{};
};

TORCH_MODULE(UnidirectionalSingleLayerLstm);

struct UnidirectionalLstmImpl : torch::nn::Module {
  UnidirectionalLstmImpl(
      int64_t num_layers,

      // See the constructor of `UnidirectionalSingleLayerLstm`.
      int64_t input_size,
      int64_t hidden_size,
      int64_t cell_size,
      bool go_forward,
      int64_t truncated_bptt,
      double cell_clip,
      double proj_clip,
      int64_t recurrent_dropout_type,
      double recurrent_dropout_probability,

      bool use_skip_connections);

  // Similar to `UnidirectionalSingleLayerLstm`, but with
  // the results of `num_layers` layers.
  //
  // Inputs: inputs, batch_sizes, (h_0, c_0)
  //   - **inputs** of shape `(*, input_size)`.
  //   - **batch_sizes** of shape `(total_timesteps,)`.
  //   - **h_0** of shape `(num_layers, {x | x >= batch}, hidden_size)`.
  //   - **c_0** of shape `(num_layers, {x | x >= batch}, cell_size)`.
  //
  // Outputs: outputs, (h_1, c_1)
  //   - **outputs**, a list of shape `(*, hidden_size)`.
  //   - **h_1** of shape `(num_layers, {x | x >= batch}, hidden_size)`.
  //   - **c_1** of shape `(num_layers, {x | x >= batch}, cell_size)`.
  LstmForwardMultiLayerRetType forward(
      torch::Tensor inputs,
      torch::Tensor batch_sizes,
      LstmStateType initial_state);

  torch::TensorOptions weight_options();

  int64_t hidden_size_ = -1;
  int64_t cell_size_ = -1;

  int64_t num_layers_ = -1;
  bool use_skip_connections_ = false;

  std::string layer_name_prefix_ = "";
  std::vector<UnidirectionalSingleLayerLstm> layers_ = {};
};

TORCH_MODULE(UnidirectionalLstm);

}  // namespace cnt

#endif  // EXTENSION_UNIDIRECTIONAL_LSTM_H_
