#include "extension/unidirectional_lstm.h"
#include <cmath>
#include <stdexcept>

namespace cnt {

void block_orthogonal_recursive(
    const at::IntList sizes,
    const at::IntList split_sizes,
    const double gain,
    const int64_t dim,
    torch::Tensor pre_narrowed_tensor) {
  if (dim == static_cast<int64_t>(split_sizes.size())) {
    // Find a block.
    torch::nn::init::orthogonal_(pre_narrowed_tensor, gain);
    return;
  }

  for (int64_t start = 0; start < sizes[dim]; start += split_sizes[dim]) {
    block_orthogonal_recursive(
        sizes,
        split_sizes,
        gain,
        dim + 1,
        pre_narrowed_tensor.narrow(dim, start, split_sizes[dim]));
  }
}

void block_orthogonal(
    torch::Tensor tensor,
    at::IntList split_sizes,
    double gain = 1.0) {
  // Get the sizes of ``tensor``.
  auto sizes = tensor.sizes();

  // Check ``split_sizes``:
  if (split_sizes.size() != sizes.size()) {
    throw std::invalid_argument("Dimension not match: split_sizes, sizes");
  }
  for (int64_t dim = 0; dim < static_cast<int64_t>(split_sizes.size()); dim++) {
    if (sizes[dim] % split_sizes[dim] != 0) {
      throw std::invalid_argument("Size divisible: split_sizes, sizes");
    }
  }

  // Apply ``torch::nn::init::orthogonal_`` to all combinations of slices.
  block_orthogonal_recursive(
      sizes,
      split_sizes,
      gain,
      0,
      tensor);
}

torch::Tensor get_dropout_mask(
    double dropout_probability,
    torch::Tensor tensor_for_masking) {
  auto binary_mask = torch::rand(tensor_for_masking.sizes())
      .gt(dropout_probability)
      .to(tensor_for_masking.device());
  auto dropout_mask = binary_mask
      .to(torch::dtype(torch::kFloat32))
      .div(1.0 - dropout_probability);
  return dropout_mask;
}

UnidirectionalSingleLayerLstm::UnidirectionalSingleLayerLstm(
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
    input_size_(input_size),
    hidden_size_(hidden_size),
    cell_size_(cell_size),
    go_forward_(go_forward),
    truncated_bptt_(truncated_bptt),
    cell_clip_(fabs(cell_clip)),
    proj_clip_(fabs(proj_clip)),
    recurrent_dropout_type_(recurrent_dropout_type),
    recurrent_dropout_probability_(recurrent_dropout_probability) {
  // Create & bind linear projections.
  input_linearity_weight_ = register_parameter(
      "input_linearity_weight",
      torch::zeros(
          {4 * cell_size_, input_size_},
          torch::dtype(torch::kFloat32)));

  hidden_linearity_weight_ = register_parameter(
      "hidden_linearity_weight",
      torch::zeros(
          {4 * cell_size_, hidden_size_},
          torch::dtype(torch::kFloat32)));
  hidden_linearity_bias_ = register_parameter(
      "hidden_linearity_bias",
      torch::zeros(
          {4 * cell_size_},
          torch::dtype(torch::kFloat32)));

  proj_linearity_weight_ = register_parameter(
      "proj_linearity_weight",
      torch::zeros(
          {hidden_size_, cell_size_},
          torch::dtype(torch::kFloat32)));

  // Reset parameters.
  block_orthogonal(
      input_linearity_weight_.detach(),
      {cell_size_, input_size_});

  block_orthogonal(
      hidden_linearity_weight_.detach(),
      {cell_size_, hidden_size_});
  hidden_linearity_bias_
      .detach()
      .fill_(0.0);
  hidden_linearity_bias_
      .detach()
      .narrow(0, cell_size_, cell_size_).fill_(1.0);

  block_orthogonal(
      proj_linearity_weight_.detach(),
      {hidden_size_, cell_size_});
}

LstmForwardRetType UnidirectionalSingleLayerLstm::forward(
    torch::Tensor inputs,
    const std::vector<int> &batch_lengths,
    LstmStateType initial_state) {
  int64_t batch_size = inputs.size(0);
  int64_t total_timesteps = inputs.size(1);
  auto options = torch::dtype(inputs.dtype()).device(inputs.device());

  // Unpack hidden/cell state.
  auto hidden_state = std::get<0>(initial_state).squeeze(0);
  auto cell_state = std::get<1>(initial_state).squeeze(0);
  if (hidden_state.size(0) < batch_size || cell_state.size(0) < batch_size) {
    throw std::invalid_argument(
        "initial_state can't fullfill inputs.");
  }

  auto output_accumulator = torch::zeros(
      {batch_size, total_timesteps, hidden_size_}, options);

  // Variational dropout mask on hidden state.
  torch::Tensor variational_dropout_mask;
  if (is_training() && recurrent_dropout_type_ == 1) {
    if (recurrent_dropout_probability_ == 0.0) {
      throw std::invalid_argument(
          "Variational: recurrent_dropout_probability not provided.");
    }
    variational_dropout_mask = get_dropout_mask(
        recurrent_dropout_probability_,
        hidden_state);
  }
  // Dropconnect mask on input/hidden linear projection weights.
  torch::Tensor dropconnect_input_mask;
  torch::Tensor dropconnect_hidden_mask;
  if (is_training() && recurrent_dropout_type_ == 2) {
    if (recurrent_dropout_probability_ == 0.0) {
      throw std::invalid_argument(
          "Dropconnect: recurrent_dropout_probability not provided.");
    }
    dropconnect_input_mask = get_dropout_mask(
        recurrent_dropout_probability_,
        input_linearity_weight_);
    dropconnect_hidden_mask = get_dropout_mask(
        recurrent_dropout_probability_,
        hidden_linearity_weight_);
  }

  auto batch_input_linearity_weight = input_linearity_weight_;
  auto batch_hidden_linearity_weight = hidden_linearity_weight_;
  // Apply Dropconnect.
  if (is_training() && recurrent_dropout_type_ == 2) {
    batch_input_linearity_weight =
        input_linearity_weight_ * dropconnect_input_mask;
    batch_hidden_linearity_weight =
        hidden_linearity_weight_ * dropconnect_hidden_mask;
  }

  int64_t batch_upper = go_forward_ ? batch_size - 1 : 0;
  for (int64_t offset = 0; offset < total_timesteps; offset++) {
    // Set timestep_index and batch_upper.
    int64_t timestep_index =
        go_forward_ ? offset : total_timesteps - 1 - offset;

    if (go_forward_) {
      while (batch_upper > 0
            && batch_lengths[batch_upper] <= timestep_index) {
        --batch_upper;
      }
    } else {
      while (batch_upper < batch_size - 1
            && batch_lengths[batch_upper + 1] > timestep_index) {
        ++batch_upper;
      }
    }

    // TF-style TBPTT.
    // See https://r2rt.com/styles-of-truncated-backpropagation.html.
    if (is_training()
          && truncated_bptt_ > 0
          && offset > 0
          && offset % truncated_bptt_ == 0) {
      hidden_state = hidden_state.detach();
      cell_state = cell_state.detach();
    }

    // Slicing hidden/cell/inputs.
    auto cur_hidden = hidden_state.narrow(0, 0, batch_upper + 1);
    auto cur_cell = cell_state.narrow(0, 0, batch_upper + 1);
    auto cur_input =
        inputs.narrow(0, 0, batch_upper + 1).select(1, timestep_index);

    // Compute next hidden/cell state.
    auto proj_input = torch::linear(
        cur_input, batch_input_linearity_weight);
    auto proj_hidden = torch::linear(
        cur_hidden, batch_hidden_linearity_weight, hidden_linearity_bias_);

    auto input_gate = torch::sigmoid(
        proj_input.narrow(1, 0 * cell_size_, cell_size_) +
        proj_hidden.narrow(1, 0 * cell_size_, cell_size_));
    auto forget_gate = torch::sigmoid(
        proj_input.narrow(1, 1 * cell_size_, cell_size_) +
        proj_hidden.narrow(1, 1 * cell_size_, cell_size_));
    auto cell_tilde = torch::tanh(
        proj_input.narrow(1, 2 * cell_size_, cell_size_) +
        proj_hidden.narrow(1, 2 * cell_size_, cell_size_));
    auto output_gate = torch::sigmoid(
        proj_input.narrow(1, 3 * cell_size_, cell_size_) +
        proj_hidden.narrow(1, 3 * cell_size_, cell_size_));

    auto next_cell = input_gate * cell_tilde + forget_gate * cur_cell;
    if (cell_clip_ > 0.0) {
      next_cell.clamp_(-cell_clip_, cell_clip_);
    }

    auto next_hidden = torch::linear(
        output_gate * torch::tanh(next_cell),
        proj_linearity_weight_);
    if (proj_clip_ > 0.0) {
      next_hidden.clamp_(-proj_clip_, proj_clip_);
    }

    // Apply variational dropout.
    if (is_training() && recurrent_dropout_type_ == 1) {
      next_hidden *=
          variational_dropout_mask.narrow(0, 0, batch_upper + 1);
    }

    // Required by gradients propagation.
    hidden_state = hidden_state.clone();
    cell_state = cell_state.clone();
    // Update hidden/cell state.
    hidden_state
      .narrow(0, 0, batch_upper + 1)
      .copy_(next_hidden);
    cell_state
      .narrow(0, 0, batch_upper + 1)
      .copy_(next_cell);

    // Fill hidden state to output.
    output_accumulator
      .narrow(0, 0, batch_upper + 1)
      .select(1, timestep_index)
      .copy_(next_hidden);
  }

  return std::make_tuple(
      output_accumulator,
      std::make_tuple(
          hidden_state.unsqueeze(0),
          cell_state.unsqueeze(0)));
}

LstmForwardRetType UnidirectionalSingleLayerLstm::forward(
    torch::Tensor inputs,
    const std::vector<int> &batch_lengths) {
  int64_t batch_size = inputs.size(0);
  auto options = torch::dtype(inputs.dtype()).device(inputs.device());

  auto hidden_state = torch::zeros({1, batch_size, hidden_size_}, options);
  auto cell_state = torch::zeros({1, batch_size, cell_size_}, options);
  return forward(
      inputs,
      batch_lengths,
      std::make_tuple(hidden_state, cell_state));
}

UnidirectionalLstm::UnidirectionalLstm(
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
    hidden_size_(hidden_size),
    cell_size_(cell_size),
    num_layers_(num_layers),
    layer_name_prefix_(go_forward ? "forward_layer_" : "backward_layer_") {
  auto lstm_input_size = input_size;

  for (int64_t layer_idx = 0; layer_idx < num_layers_; layer_idx++) {
    // Construct.
    auto layer_ptr = std::make_shared<UnidirectionalSingleLayerLstm>(
        lstm_input_size,
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
        layer_name_prefix_ + std::to_string(layer_idx),
        layer_ptr);
    layers_.push_back(layer_ptr);

    lstm_input_size = hidden_size;
  }
}

LstmForwardRetType UnidirectionalLstm::forward(
    torch::Tensor inputs,
    const std::vector<int> &batch_lengths,
    LstmStateType initial_state) {
  // Unpack initial_state.
  auto layers_hidden_state = std::get<0>(initial_state);
  auto layers_cell_state = std::get<1>(initial_state);

  std::vector<torch::Tensor> output_accumulators(num_layers_);
  std::vector<torch::Tensor> hidden_states(num_layers_);
  std::vector<torch::Tensor> cell_states(num_layers_);

  auto layer_inputs = inputs;
  for (int64_t layer_idx = 0; layer_idx < num_layers_; layer_idx++) {
    auto layer_hidden_state = layers_hidden_state
        .select(0, layer_idx).unsqueeze(0);
    auto layer_cell_state = layers_cell_state
        .select(0, layer_idx).unsqueeze(0);

    auto layer_out = layers_[layer_idx]->forward(
        layer_inputs,
        batch_lengths,
        std::make_tuple(
            layer_hidden_state,
            layer_cell_state));

    auto output_accumulator = std::get<0>(layer_out);
    auto final_state = std::get<1>(layer_out);
    auto hidden_state = std::get<0>(final_state);
    auto cell_state = std::get<1>(final_state);

    output_accumulators[layer_idx] = output_accumulator.unsqueeze(0);
    hidden_states[layer_idx] = hidden_state;
    cell_states[layer_idx] = cell_state;

    layer_inputs = output_accumulator;
  }

  return std::make_tuple(
      torch::cat(output_accumulators, 0),
      std::make_tuple(
          torch::cat(hidden_states, 0),
          torch::cat(cell_states, 0)));
}

LstmForwardRetType UnidirectionalLstm::forward(
    torch::Tensor inputs,
    const std::vector<int> &batch_lengths) {
  int64_t batch_size = inputs.size(0);
  auto options = torch::dtype(inputs.dtype()).device(inputs.device());

  auto hidden_state = torch::zeros(
      {num_layers_, batch_size, hidden_size_},
      options);
  auto cell_state = torch::zeros(
      {num_layers_, batch_size, cell_size_},
      options);
  return forward(
      inputs,
      batch_lengths,
      std::make_tuple(hidden_state, cell_state));
}

}  // namespace cnt
