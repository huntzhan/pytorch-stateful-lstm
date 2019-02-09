#include "extension/stateful_unidirectional_lstm.h"

namespace cnt {

StatefulUnidirectionalLstmImpl::StatefulUnidirectionalLstmImpl(
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
      bool use_skip_connections)
      :
      num_layers_(num_layers),
      hidden_size_(hidden_size),
      cell_size_(cell_size) {
  // Construct `UnidirectionalLstm`.
  uni_lstm_ = UnidirectionalLstm(
      num_layers,
      input_size,
      hidden_size,
      cell_size,
      go_forward,
      truncated_bptt,
      cell_clip,
      proj_clip,
      recurrent_dropout_type,
      recurrent_dropout_probability,
      use_skip_connections);
  // Register.
  register_module(
      "uni_lstm",
      uni_lstm_);
}

void StatefulUnidirectionalLstmImpl::prepare_managed_states(
    int64_t batch_size) {
  auto options = uni_lstm_->weight_options();
  if (!managed_hidden_state_.defined()
      || managed_hidden_state_.device() != options.device()
      || !managed_cell_state_.defined()
      || managed_cell_state_.device() != options.device()) {
    // Initialize with zero tensors
    // if the managed state is not defined or device not match.
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
}

LstmForwardMultiLayerRetType StatefulUnidirectionalLstmImpl::forward(
    torch::Tensor inputs,
    torch::Tensor batch_sizes) {
  auto batch_sizes_accessor = batch_sizes.accessor<int64_t, 1>();
  int64_t batch_size = batch_sizes_accessor[0];
  prepare_managed_states(batch_size);

  auto lstm_out = uni_lstm_(
      inputs,
      batch_sizes,
      std::make_tuple(managed_hidden_state_, managed_cell_state_));

  // Update & detach manage state.
  auto final_state = std::get<1>(lstm_out);
  auto hidden_state = std::get<0>(final_state);
  auto cell_state = std::get<1>(final_state);

  managed_hidden_state_ = hidden_state.detach();
  managed_cell_state_ = cell_state.detach();

  return lstm_out;
}

void StatefulUnidirectionalLstmImpl::permutate_states(torch::Tensor index) {
  // `index` of shape `(batch_size,)`
  int64_t batch_size = index.size(0);
  prepare_managed_states(batch_size);

  // Permuate states.
  auto permuated_managed_hidden_state =
      managed_hidden_state_.index_select(1, index);
  if (batch_size < managed_hidden_state_.size(1)) {
    permuated_managed_hidden_state = torch::cat(
        {
            permuated_managed_hidden_state,
            managed_hidden_state_.narrow(
                1, batch_size, managed_hidden_state_.size(1) - batch_size),
        },
        1);
  }
  managed_hidden_state_ = permuated_managed_hidden_state;

  auto permuated_managed_cell_state =
      managed_cell_state_.index_select(1, index);
  if (batch_size < managed_cell_state_.size(1)) {
    permuated_managed_cell_state = torch::cat(
        {
            permuated_managed_cell_state,
            managed_cell_state_.narrow(
                1, batch_size, managed_cell_state_.size(1) - batch_size),
        },
        1);
  }
  managed_cell_state_ = permuated_managed_cell_state;
}

void StatefulUnidirectionalLstmImpl::reset_states() {
  managed_hidden_state_.reset();
  managed_cell_state_.reset();
}

torch::Tensor StatefulUnidirectionalLstmImpl::managed_hidden_state() {
  return managed_hidden_state_;
}

torch::Tensor StatefulUnidirectionalLstmImpl::managed_cell_state() {
  return managed_cell_state_;
}

}  // namespace cnt
