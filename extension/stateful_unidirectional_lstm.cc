#include "extension/stateful_unidirectional_lstm.h"

namespace cnt {

StatefulUnidirectionalLstm::StatefulUnidirectionalLstm(
      int64_t num_layers,
      int64_t input_size,
      int64_t hidden_size,
      int64_t cell_size,
      bool go_forward,
      int64_t truncated_bptt,
      double cell_clip,
      double proj_clip,
      int64_t recurrent_dropout_type,
      double recurrent_dropout_probability)
      :
      num_layers_(num_layers),
      hidden_size_(hidden_size),
      cell_size_(cell_size) {
  // Construct `UnidirectionalLstm`.
  uni_lstm_ = std::make_shared<UnidirectionalLstm>(
      num_layers,
      input_size,
      hidden_size,
      cell_size,
      go_forward,
      truncated_bptt,
      cell_clip,
      proj_clip,
      recurrent_dropout_type,
      recurrent_dropout_probability);
  // Register.
  register_module(
      "uni_lstm",
      uni_lstm_);
}

LstmForwardRetType StatefulUnidirectionalLstm::forward(
      torch::Tensor inputs,
      const std::vector<int> &batch_lengths) {
  int64_t batch_size = inputs.size(0);
  auto options = torch::dtype(inputs.dtype()).device(inputs.device());

  if (!(managed_hidden_state_.defined() && managed_cell_state_.defined())) {
    // Initialize with zero tensors
    // if the managed state is not defined.
    managed_hidden_state_ = torch::zeros(
        {num_layers_, batch_size, hidden_size_},
        options);
    managed_cell_state_ = torch::zeros(
        {num_layers_, batch_size, cell_size_},
        options);

  } else if (managed_hidden_state_.size(1) < batch_size) {
    // Extend with zero tensors
    // if this batch is larger than the all previous states.
    auto num_states_to_concat = batch_size - managed_hidden_state_.size(1);

    auto hidden_state_zeros = torch::zeros(
        {num_layers_, num_states_to_concat, hidden_size_},
        options);
    managed_hidden_state_ = torch::cat(
        {managed_hidden_state_, hidden_state_zeros},
        1);

    auto cell_state_zeros = torch::zeros(
        {num_layers_, num_states_to_concat, cell_size_},
        options);
    managed_cell_state_ = torch::cat(
        {managed_cell_state_, cell_state_zeros},
        1);
  }

  auto lstm_out = uni_lstm_->forward(
      inputs,
      batch_lengths,
      std::make_tuple(managed_hidden_state_, managed_cell_state_));

  // Update & detach manage state.
  auto final_state = std::get<1>(lstm_out);
  auto hidden_state = std::get<0>(final_state);
  auto cell_state = std::get<1>(final_state);

  managed_hidden_state_ = hidden_state.detach();
  managed_cell_state_ = cell_state.detach();

  return lstm_out;
}

void StatefulUnidirectionalLstm::reset_states() {
  managed_hidden_state_.reset();
  managed_cell_state_.reset();
}

}  // namespace cnt
