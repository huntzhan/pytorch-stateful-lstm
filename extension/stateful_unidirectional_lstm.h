#ifndef EXTENSION_STATEFUL_UNIDIRECTIONAL_LSTM_H_
#define EXTENSION_STATEFUL_UNIDIRECTIONAL_LSTM_H_

#include <vector>
#include "extension/unidirectional_lstm.h"

namespace cnt {

struct StatefulUnidirectionalLstm : torch::nn::Module {
  // See the constructor of `UnidirectionalLstm`.
  StatefulUnidirectionalLstm(
      int64_t num_layers,
      int64_t input_size,
      int64_t hidden_size,
      int64_t cell_size,
      bool go_forward,
      int64_t truncated_bptt,
      double cell_clip,
      double proj_clip,
      int64_t recurrent_dropout_type,
      double recurrent_dropout_probability);

  // Same as `UnidirectionalLstm.forward(inputs, batch_lengths)`,
  // but with managed hidden/cell state.
  LstmForwardRetType forward(
      torch::Tensor inputs,
      const std::vector<int> &batch_lengths);

  // Reset hidden/cell state.
  void reset_states();

  // For building the initial state.
  int64_t num_layers_ = -1;
  int64_t hidden_size_ = -1;
  int64_t cell_size_ = -1;

  // Of shape `(num_layers, max_batch, hidden_size)`.
  torch::Tensor managed_hidden_state_{};
  // Of shape `(num_layers, max_batch, cell_size)`.
  torch::Tensor managed_cell_state_{};

  std::shared_ptr<UnidirectionalLstm> uni_lstm_ = nullptr;
};

}  // namespace cnt

#endif  // EXTENSION_STATEFUL_UNIDIRECTIONAL_LSTM_H_
