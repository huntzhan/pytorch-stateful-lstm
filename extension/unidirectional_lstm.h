#ifndef EXTENSION_UNIDIRECTIONAL_LSTM_H_
#define EXTENSION_UNIDIRECTIONAL_LSTM_H_

#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <string>

namespace cnt {

using LstmStateType = std::tuple<torch::Tensor, torch::Tensor>;
using LstmForwardRetType = std::tuple<torch::Tensor, LstmStateType>;

struct UnidirectionalSingleLayerLstm : torch::nn::Module {
  UnidirectionalSingleLayerLstm(
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

  // Inputs: inputs, batch_lengths, (h_0, c_0)
  //   - **inputs** of shape `(batch, total_timesteps, input_size)`:
  //     tensor containing inputs that must be sorted by the length
  //     of sequence in descending order.
  //   - **batch_lengths** of shape `(batch,)`:
  //     list containing the lengths of the sequences in batch.
  //   - **h_0** of shape `(1, {x | x >= batch}, hidden_size)`.
  //     tensor containing the intial hidden state of LSTM.
  //   - **c_0** of shape `(1, {x | x >= batch}, cell_size)`.
  //     tensor containing the intial cell state of LSTM.
  //
  // Outputs: output, (h_1, c_1)
  //   - **output** of shape
  //     `(batch, total_timesteps, hidden_size)`:
  //     tensor containing output features.
  //   - **h_1** of shape `(1, {x | x >= batch}, hidden_size)`.
  //     tensor containing the final hidden state of LSTM.
  //   - **c_1** of shape `(1, {x | x >= batch}, cell_size)`.
  //     tensor containing the final cell state of LSTM.
  LstmForwardRetType forward(
      torch::Tensor inputs,
      const std::vector<int> &batch_lengths,
      LstmStateType initial_state);

  // Equivalent to passing zero tensors as the initial states.
  LstmForwardRetType forward(
      torch::Tensor inputs,
      const std::vector<int> &batch_lengths);

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

struct UnidirectionalLstm : torch::nn::Module {
  UnidirectionalLstm(
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
      double recurrent_dropout_probability);

  // Similar to `UnidirectionalSingleLayerLstm`, but with
  // the results of `num_layers` layers.
  //
  // Inputs: inputs, batch_lengths, (h_0, c_0)
  //   - **inputs** of shape `(batch, total_timesteps, input_size)`.
  //   - **batch_lengths** of shape `(batch,)`.
  //   - **h_0** of shape `(num_layers, {x | x >= batch}, hidden_size)`.
  //   - **c_0** of shape `(num_layers, {x | x >= batch}, cell_size)`.
  //
  // Outputs: output, (h_1, c_1)
  //   - **output** of shape
  //     `(num_layers, batch, total_timesteps, hidden_size)`.
  //   - **h_1** of shape `(num_layers, {x | x >= batch}, hidden_size)`.
  //   - **c_1** of shape `(num_layers, {x | x >= batch}, cell_size)`.
  LstmForwardRetType forward(
      torch::Tensor inputs,
      const std::vector<int> &batch_lengths,
      LstmStateType initial_state);

  // Equivalent to passing zero tensors as the initial states.
  LstmForwardRetType forward(
      torch::Tensor inputs,
      const std::vector<int> &batch_lengths);

  int64_t hidden_size_ = -1;
  int64_t cell_size_ = -1;

  int64_t num_layers_ = -1;
  std::string layer_name_prefix_ = "";
  std::vector<std::shared_ptr<UnidirectionalSingleLayerLstm>> layers_ = {};
};

}  // namespace cnt

#endif  // EXTENSION_UNIDIRECTIONAL_LSTM_H_
