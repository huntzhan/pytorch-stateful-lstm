#ifndef EXTENSION_STATEFUL_UNIDIRECTIONAL_LSTM_H_
#define EXTENSION_STATEFUL_UNIDIRECTIONAL_LSTM_H_

#include <vector>
#include "extension/unidirectional_lstm.h"

namespace cnt {

struct StatefulUnidirectionalLstmImpl : torch::nn::Module {
  // See the constructor of `UnidirectionalLstm`.
  StatefulUnidirectionalLstmImpl(
      int64_t num_layers,
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

  // Same as `UnidirectionalLstm.forward(inputs, batch_sizes)`,
  // but with managed hidden/cell state.
  LstmForwardMultiLayerRetType forward(
      torch::Tensor inputs,
      torch::Tensor batch_sizes);

  // Initialize managed states if necessary.
  void prepare_managed_states(int64_t batch_size);

  // Permutate hidden/cell state for ordered inputs.
  void permutate_states(torch::Tensor index);

  // Reset hidden/cell state.
  void reset_states();

  // Accessor.
  torch::Tensor managed_hidden_state();
  torch::Tensor managed_cell_state();

  // For building the initial state.
  int64_t num_layers_ = -1;
  int64_t hidden_size_ = -1;
  int64_t cell_size_ = -1;

  // Of shape `(num_layers, max_batch, hidden_size)`.
  torch::Tensor managed_hidden_state_{};
  // Of shape `(num_layers, max_batch, cell_size)`.
  torch::Tensor managed_cell_state_{};

  UnidirectionalLstm uni_lstm_ = nullptr;
};

TORCH_MODULE(StatefulUnidirectionalLstm);

}  // namespace cnt

#endif  // EXTENSION_STATEFUL_UNIDIRECTIONAL_LSTM_H_
