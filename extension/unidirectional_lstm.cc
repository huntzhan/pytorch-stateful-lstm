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

UnidirectionalSingleLayerLstmImpl::UnidirectionalSingleLayerLstmImpl(
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

LstmForwardRetType UnidirectionalSingleLayerLstmImpl::forward(
    torch::Tensor inputs,
    torch::Tensor batch_sizes,
    LstmStateType initial_state) {
  // Reduce the cost of dynamic dispatch.
  auto batch_sizes_accessor = batch_sizes.accessor<int64_t, 1>();

  int64_t total_sequences_size = inputs.size(0);
  int64_t total_timesteps = batch_sizes_accessor.size(0);

  // Unpack hidden/cell state.
  auto initial_hidden_state = std::get<0>(initial_state).squeeze(0);
  auto initial_cell_state = std::get<1>(initial_state).squeeze(0);
  // `batch_sizes_accessor[0]`: the batch size.
  if (initial_hidden_state.size(0) < batch_sizes_accessor[0]
      || initial_cell_state.size(0) < batch_sizes_accessor[0]) {
    throw std::invalid_argument(
        "initial_state can't fullfill inputs.");
  }

  // Initialize the output container.
  std::vector<torch::Tensor> output_accumulator(total_timesteps);
  // Keep the chunks of hidden/cell in forward mode.
  std::vector<torch::Tensor> forward_hiddens;
  std::vector<torch::Tensor> forward_cells;

  // Create aliases for dropout.
  auto batch_input_linearity_weight = input_linearity_weight_;
  auto batch_hidden_linearity_weight = hidden_linearity_weight_;

  // Variational dropout mask on hidden state.
  torch::Tensor variational_dropout_mask;
  if (is_training() && recurrent_dropout_type_ == 1) {
    if (recurrent_dropout_probability_ == 0.0) {
      throw std::invalid_argument(
          "Variational: recurrent_dropout_probability not provided.");
    }
    variational_dropout_mask = get_dropout_mask(
        recurrent_dropout_probability_,
        initial_hidden_state);
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
  // Apply Dropconnect.
  if (is_training() && recurrent_dropout_type_ == 2) {
    batch_input_linearity_weight =
        input_linearity_weight_ * dropconnect_input_mask;
    batch_hidden_linearity_weight =
        hidden_linearity_weight_ * dropconnect_hidden_mask;
  }

  // Temporal hidden/cell.
  int64_t init_batch_size =
      go_forward_ ?
      batch_sizes_accessor[0] :
      batch_sizes_accessor[total_timesteps - 1];
  auto hidden_state = initial_hidden_state.narrow(0, 0, init_batch_size);
  auto cell_state = initial_cell_state.narrow(0, 0, init_batch_size);

  // Loop over each timestep and unpack inputs manually.
  int64_t total_offset = 0;
  for (int64_t ts_offset = 0; ts_offset < total_timesteps; ts_offset++) {
    int64_t timestep_index =
        go_forward_ ? ts_offset : total_timesteps - 1 - ts_offset;
    int64_t batch_size = batch_sizes_accessor[timestep_index];

    // TF-style TBPTT.
    // See https://r2rt.com/styles-of-truncated-backpropagation.html.
    if (is_training()
          && truncated_bptt_ > 0
          && ts_offset > 0
          && ts_offset % truncated_bptt_ == 0) {
      hidden_state = hidden_state.detach();
      cell_state = cell_state.detach();
    }

    // Slicing hidden/cell.
    if (go_forward_) {
      if (timestep_index > 0) {
        // Trim off if necessary.
        auto last_batch_size = batch_sizes_accessor[timestep_index - 1];
        auto dec = last_batch_size - batch_size;
        if (dec > 0) {
          forward_hiddens.push_back(
              hidden_state.narrow(
                  0,
                  last_batch_size - dec,
                  dec));
          forward_cells.push_back(
              cell_state.narrow(
                  0,
                  last_batch_size - dec,
                  dec));
          // Narrows state.
          hidden_state = hidden_state.narrow(0, 0, batch_size);
          cell_state = cell_state.narrow(0, 0, batch_size);
        }
      }
    } else {
      if (timestep_index < total_timesteps - 1) {
        auto last_batch_size = batch_sizes_accessor[timestep_index + 1];
        auto inc = batch_size - last_batch_size;
        if (inc > 0) {
          hidden_state = torch::cat(
              {
                  hidden_state,
                  initial_hidden_state.narrow(0, batch_size - inc, inc),
              },
              0);
          cell_state = torch::cat(
              {
                  cell_state,
                  initial_cell_state.narrow(0, batch_size - inc, inc),
              },
              0);
        }
      }
    }

    // Slicing inputs.
    int64_t input_slice_begin = -1;
    if (go_forward_) {
      input_slice_begin = total_offset;
    } else {
      input_slice_begin = total_sequences_size - total_offset - batch_size;
    }
    auto cur_input = inputs.narrow(0, input_slice_begin, batch_size);

    // Compute next hidden/cell state.
    auto proj_input = torch::linear(
        cur_input, batch_input_linearity_weight);
    auto proj_hidden = torch::linear(
        hidden_state, batch_hidden_linearity_weight, hidden_linearity_bias_);

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

    cell_state = input_gate * cell_tilde + forget_gate * cell_state;
    if (cell_clip_ > 0.0) {
      cell_state.clamp_(-cell_clip_, cell_clip_);
    }

    hidden_state = torch::linear(
        output_gate * torch::tanh(cell_state),
        proj_linearity_weight_);
    if (proj_clip_ > 0.0) {
      hidden_state.clamp_(-proj_clip_, proj_clip_);
    }

    // Apply variational dropout.
    if (is_training() && recurrent_dropout_type_ == 1) {
      hidden_state *=
          variational_dropout_mask.narrow(0, 0, batch_size);
    }

    // Update offset.
    total_offset += batch_size;

    // Fill hidden state to output.
    output_accumulator[timestep_index] = hidden_state;
  }

  if (go_forward_) {
    forward_hiddens.push_back(hidden_state);
    std::reverse(forward_hiddens.begin(), forward_hiddens.end());
    hidden_state = torch::cat(forward_hiddens, 0);

    forward_cells.push_back(cell_state);
    std::reverse(forward_cells.begin(), forward_cells.end());
    cell_state = torch::cat(forward_cells, 0);
  }

  auto state_dec = initial_hidden_state.size(0) - hidden_state.size(0);
  if (state_dec > 0) {
    hidden_state = torch::cat(
        {
            hidden_state,
            initial_hidden_state.narrow(0, hidden_state.size(0), state_dec),
        },
        0);
    cell_state = torch::cat(
        {
            cell_state,
            initial_cell_state.narrow(0, cell_state.size(0), state_dec),
        },
        0);
  }

  return std::make_tuple(
      torch::cat(output_accumulator, 0),
      std::make_tuple(
          hidden_state.unsqueeze(0),
          cell_state.unsqueeze(0)));
}

torch::TensorOptions UnidirectionalSingleLayerLstmImpl::weight_options() {
  return torch::dtype(torch::kFloat32)
      .device(hidden_linearity_weight_.device());
}

UnidirectionalLstmImpl::UnidirectionalLstmImpl(
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
    hidden_size_(hidden_size),
    cell_size_(cell_size),
    num_layers_(num_layers),
    use_skip_connections_(use_skip_connections),
    layer_name_prefix_(go_forward ? "forward_layer_" : "backward_layer_") {
  if (num_layers_ == 0) {
    throw std::invalid_argument("num_layers should >= 1.");
  }

  auto lstm_input_size = input_size;

  for (int64_t layer_idx = 0; layer_idx < num_layers_; layer_idx++) {
    // Construct.
    auto layer = UnidirectionalSingleLayerLstm(
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
        layer);
    layers_.push_back(layer);

    lstm_input_size = hidden_size;
  }
}

LstmForwardMultiLayerRetType UnidirectionalLstmImpl::forward(
    torch::Tensor inputs,
    torch::Tensor batch_sizes,
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

    auto layer_out = layers_[layer_idx](
        layer_inputs,
        batch_sizes,
        std::make_tuple(
            layer_hidden_state,
            layer_cell_state));

    auto output_accumulator = std::get<0>(layer_out);
    auto final_state = std::get<1>(layer_out);
    auto hidden_state = std::get<0>(final_state);
    auto cell_state = std::get<1>(final_state);

    output_accumulators[layer_idx] = output_accumulator;
    hidden_states[layer_idx] = hidden_state;
    cell_states[layer_idx] = cell_state;

    if (layer_idx > 0 && use_skip_connections_) {
      // input_size == hidden_size.
      output_accumulator += layer_inputs;
    }

    layer_inputs = output_accumulator;
  }

  return std::make_tuple(
      output_accumulators,
      std::make_tuple(
          torch::cat(hidden_states, 0),
          torch::cat(cell_states, 0)));
}

torch::TensorOptions UnidirectionalLstmImpl::weight_options() {
  return layers_[0]->weight_options();
}

}  // namespace cnt
