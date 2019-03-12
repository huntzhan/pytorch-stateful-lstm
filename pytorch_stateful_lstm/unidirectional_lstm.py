"""
Python implementation of unidirectional_lstm.cc.
"""
from typing import Tuple, List
import math

import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter


def block_orthogonal_recursive(
        sizes: torch.Size,
        split_sizes: List[int],
        gain: float,
        dim: int,
        tensor: torch.Tensor,
) -> None:
    if dim == tensor.dim():
        torch.nn.init.orthogonal_(tensor, gain)
        return

    for start in range(0, sizes[dim], split_sizes[dim]):
        block_orthogonal_recursive(
                sizes,
                split_sizes,
                gain,
                dim + 1,
                tensor.narrow(dim, start, split_sizes[dim]),
        )


def block_orthogonal(
        tensor: torch.Tensor,
        split_sizes: List[int],
        gain: float = 1.0,
) -> None:
    sizes = tensor.shape

    if len(split_sizes) != len(sizes):
        raise ValueError("Dimension not match: split_sizes, sizes")

    for size_dim, split_size_dim in zip(sizes, split_sizes):
        if size_dim % split_size_dim != 0:
            raise ValueError("Size divisible: split_sizes, sizes")

    block_orthogonal_recursive(
            sizes,
            split_sizes,
            gain,
            0,
            tensor,
    )


def get_dropout_mask(
        dropout_probability: float,
        tensor_for_masking: torch.Tensor,
) -> torch.Tensor:
    binary_mask = torch.rand_like(tensor_for_masking).gt(dropout_probability)
    dropout_mask = binary_mask.to(torch.float).div(1.0 - dropout_probability)
    return dropout_mask


class PyUnidirectionalSingleLayerLstm(torch.nn.Module):  # type: ignore

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            cell_size: int,
            go_forward: bool = True,
            truncated_bptt: int = 0,
            cell_clip: float = 0.0,
            proj_clip: float = 0.0,
            recurrent_dropout_type: int = 0,
            recurrent_dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.go_forward = go_forward
        self.truncated_bptt = truncated_bptt
        self.cell_clip = math.fabs(cell_clip)
        self.proj_clip = math.fabs(proj_clip)
        self.recurrent_dropout_type = recurrent_dropout_type
        self.recurrent_dropout_probability = recurrent_dropout_probability

        self.input_linearity_weight = Parameter(torch.zeros((4 * cell_size, input_size)))
        block_orthogonal(self.input_linearity_weight.detach(), [cell_size, input_size])

        self.hidden_linearity_weight = Parameter(torch.zeros((4 * cell_size, hidden_size)))
        block_orthogonal(self.hidden_linearity_weight.detach(), [cell_size, hidden_size])

        self.hidden_linearity_bias = Parameter(torch.zeros((4 * cell_size,)))
        self.hidden_linearity_bias.detach().narrow(0, cell_size, cell_size).fill_(1.0)

        self.proj_linearity_weight = Parameter(torch.zeros((hidden_size, cell_size)))
        block_orthogonal(self.proj_linearity_weight.detach(), [hidden_size, cell_size])

    def forward(  # pylint: disable=arguments-differ
            self,
            inputs: torch.Tensor,
            batch_sizes: torch.Tensor,
            initial_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        total_sequences_size = inputs.shape[0]
        total_timesteps = batch_sizes.shape[0]

        initial_hidden_state, initial_cell_state = initial_state
        initial_hidden_state = initial_hidden_state.squeeze(0)
        initial_cell_state = initial_cell_state.squeeze(0)

        if initial_hidden_state.shape[0] < batch_sizes[0] \
                or initial_cell_state.shape[0] < batch_sizes[0]:
            raise ValueError("initial_state can't fullfill inputs.")

        output_accumulator = [None] * total_timesteps
        forward_hiddens = []
        forward_cells = []

        batch_input_linearity_weight = self.input_linearity_weight
        batch_hidden_linearity_weight = self.hidden_linearity_weight

        if self.training and self.recurrent_dropout_type == 1:
            if self.recurrent_dropout_probability == 0.0:
                raise ValueError("Variational: recurrent_dropout_probability not provided.")

            variational_dropout_mask = get_dropout_mask(
                    self.recurrent_dropout_probability,
                    initial_hidden_state,
            )

        if self.training and self.recurrent_dropout_type == 2:
            if self.recurrent_dropout_probability == 0.0:
                raise ValueError("Dropconnect: recurrent_dropout_probability not provided.")

            dropconnect_input_mask = get_dropout_mask(
                    self.recurrent_dropout_probability,
                    batch_input_linearity_weight,
            )
            dropconnect_hidden_mask = get_dropout_mask(
                    self.recurrent_dropout_probability,
                    batch_hidden_linearity_weight,
            )

            batch_input_linearity_weight = \
                    batch_input_linearity_weight * dropconnect_input_mask
            batch_hidden_linearity_weight = \
                    batch_hidden_linearity_weight * dropconnect_hidden_mask

        init_batch_size = batch_sizes[0] if self.go_forward else batch_sizes[total_timesteps - 1]
        hidden_state = initial_hidden_state.narrow(0, 0, init_batch_size)
        cell_state = initial_cell_state.narrow(0, 0, init_batch_size)

        total_offset = 0
        for ts_offset in range(0, total_timesteps):
            timestep_index = ts_offset if self.go_forward else total_timesteps - 1 - ts_offset
            batch_size = batch_sizes[timestep_index]

            if self.training \
                    and self.truncated_bptt > 0 \
                    and ts_offset > 0 \
                    and ts_offset % self.truncated_bptt == 0:
                hidden_state = hidden_state.detach()
                cell_state = cell_state.detach()

            if self.go_forward:
                if timestep_index > 0:
                    last_batch_size = batch_sizes[timestep_index - 1]
                    dec = last_batch_size - batch_size
                    if dec > 0:
                        forward_hiddens.append(hidden_state.narrow(0, last_batch_size - dec, dec))
                        forward_cells.append(cell_state.narrow(0, last_batch_size - dec, dec))

                        hidden_state = hidden_state.narrow(0, 0, batch_size)
                        cell_state = cell_state.narrow(0, 0, batch_size)

            else:
                if timestep_index < total_timesteps - 1:
                    last_batch_size = batch_sizes[timestep_index + 1]
                    inc = batch_size - last_batch_size
                    if inc > 0:
                        hidden_state = torch.cat(
                                [
                                        hidden_state,
                                        initial_hidden_state.narrow(0, batch_size - inc, inc),
                                ],
                                dim=0,
                        )
                        cell_state = torch.cat(
                                [
                                        cell_state,
                                        initial_cell_state.narrow(0, batch_size - inc, inc),
                                ],
                                dim=0,
                        )

            if self.go_forward:
                input_slice_begin = total_offset
            else:
                input_slice_begin = total_sequences_size - total_offset - batch_size
            cur_input = inputs.narrow(0, input_slice_begin, batch_size)

            proj_input = F.linear(cur_input, batch_input_linearity_weight)
            proj_hidden = F.linear(
                    hidden_state,
                    batch_hidden_linearity_weight,
                    self.hidden_linearity_bias,
            )

            input_gate = torch.sigmoid(
                    proj_input.narrow(1, 0 * self.cell_size, self.cell_size) +
                    proj_hidden.narrow(1, 0 * self.cell_size, self.cell_size))
            forget_gate = torch.sigmoid(
                    proj_input.narrow(1, 1 * self.cell_size, self.cell_size) +
                    proj_hidden.narrow(1, 1 * self.cell_size, self.cell_size))
            cell_tilde = torch.tanh(
                    proj_input.narrow(1, 2 * self.cell_size, self.cell_size) +
                    proj_hidden.narrow(1, 2 * self.cell_size, self.cell_size))
            output_gate = torch.sigmoid(
                    proj_input.narrow(1, 3 * self.cell_size, self.cell_size) +
                    proj_hidden.narrow(1, 3 * self.cell_size, self.cell_size))

            cell_state = input_gate * cell_tilde + forget_gate * cell_state
            if self.cell_clip > 0.0:
                cell_state.clamp_(-self.cell_clip, self.cell_clip)

            hidden_state = F.linear(
                    output_gate * torch.tanh(cell_state),
                    self.proj_linearity_weight,
            )
            if self.proj_clip > 0.0:
                hidden_state.clamp_(-self.proj_clip, self.proj_clip)

            if self.training and self.recurrent_dropout_type == 1:
                hidden_state *= variational_dropout_mask.narrow(0, 0, batch_size)

            total_offset += batch_size

            output_accumulator[timestep_index] = hidden_state

        if self.go_forward:
            forward_hiddens.append(hidden_state)
            hidden_state = torch.cat(forward_hiddens[::-1], dim=0)

            forward_cells.append(cell_state)
            cell_state = torch.cat(forward_cells[::-1], dim=0)

        state_dec = initial_hidden_state.shape[0] - hidden_state.shape[0]
        if state_dec > 0:
            hidden_state = torch.cat(
                    [
                            hidden_state,
                            initial_hidden_state.narrow(0, hidden_state.shape[0], state_dec),
                    ],
                    dim=0,
            )

        state_dec = initial_cell_state.shape[0] - cell_state.shape[0]
        if state_dec > 0:
            cell_state = torch.cat(
                    [
                            cell_state,
                            initial_cell_state.narrow(0, cell_state.shape[0], state_dec),
                    ],
                    dim=0,
            )

        final_state = (hidden_state.unsqueeze(0), cell_state.unsqueeze(0))
        return torch.cat(output_accumulator, dim=0), final_state


class PyUnidirectionalLstm(torch.nn.Module):  # type: ignore

    def __init__(
            self,
            num_layers: int,
            input_size: int,
            hidden_size: int,
            cell_size: int,
            go_forward: bool = True,
            truncated_bptt: int = 0,
            cell_clip: float = 0.0,
            proj_clip: float = 0.0,
            recurrent_dropout_type: int = 0,
            recurrent_dropout_probability: float = 0.0,
            use_skip_connections: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.use_skip_connections = use_skip_connections

        if num_layers <= 0:
            raise ValueError("num_layers should >= 1.")

        self.layers: List[PyUnidirectionalSingleLayerLstm] = []
        layer_name_prefix = "forward_layer_" if go_forward else "backward_layer_"
        lstm_input_size = input_size
        for layer_idx in range(0, num_layers):
            layer = PyUnidirectionalSingleLayerLstm(
                    lstm_input_size,
                    hidden_size,
                    cell_size,
                    go_forward,
                    truncated_bptt,
                    cell_clip,
                    proj_clip,
                    recurrent_dropout_type,
                    recurrent_dropout_probability,
            )
            self.add_module(
                    layer_name_prefix + str(layer_idx),
                    layer,
            )
            self.layers.append(layer)

            lstm_input_size = hidden_size

    def forward(  # pylint: disable=arguments-differ
            self,
            inputs: torch.Tensor,
            batch_sizes: torch.Tensor,
            initial_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        layers_hidden_state, layers_cell_state = initial_state

        output_accumulators = [None] * self.num_layers
        hidden_states = [None] * self.num_layers
        cell_states = [None] * self.num_layers

        layer_inputs = inputs
        for layer_idx in range(0, self.num_layers):
            layer_hidden_state = layers_hidden_state.select(0, layer_idx).unsqueeze(0)
            layer_cell_state = layers_cell_state.select(0, layer_idx).unsqueeze(0)

            output_accumulator, (hidden_state, cell_state) = self.layers[layer_idx](
                    layer_inputs,
                    batch_sizes,
                    (layer_hidden_state, layer_cell_state),
            )
            output_accumulators[layer_idx] = output_accumulator
            hidden_states[layer_idx] = hidden_state
            cell_states[layer_idx] = cell_state

            if layer_idx > 0 and self.use_skip_connections:
                output_accumulator += layer_inputs
            layer_inputs = output_accumulator

        return output_accumulators, (torch.cat(hidden_states, dim=0), torch.cat(cell_states, dim=0))
