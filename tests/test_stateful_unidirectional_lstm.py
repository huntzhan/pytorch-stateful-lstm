import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import numpy

from pytorch_stateful_lstm import StatefulUnidirectionalLstm


def test_stateful_unidirectional_lstm():
    input_tensor = torch.rand(4, 5, 3)
    input_tensor[1, 4:, :] = 0.
    input_tensor[2, 2:, :] = 0.
    input_tensor[3, 1:, :] = 0.

    inputs = pack_padded_sequence(input_tensor, [5, 4, 2, 1], batch_first=True)

    lstm = StatefulUnidirectionalLstm(
            num_layers=2,
            input_size=3,
            hidden_size=5,
            cell_size=7,
    )

    outputs, lstm_state_1 = lstm(
            inputs.data,
            inputs.batch_sizes,
    )
    output_sequence_1, _batch_sizes = pad_packed_sequence(
            PackedSequence(outputs[1], inputs.batch_sizes),
            batch_first=True,
    )
    assert list(output_sequence_1.size()) == [4, 5, 5]
    assert list(lstm_state_1[0].size()) == [2, 4, 5]
    assert list(lstm_state_1[1].size()) == [2, 4, 7]

    outputs, lstm_state_2 = lstm(
            inputs.data,
            inputs.batch_sizes,
    )
    output_sequence_2, _batch_sizes = pad_packed_sequence(
            PackedSequence(outputs[1], inputs.batch_sizes),
            batch_first=True,
    )
    assert list(output_sequence_2.size()) == [4, 5, 5]
    assert list(lstm_state_2[0].size()) == [2, 4, 5]
    assert list(lstm_state_2[1].size()) == [2, 4, 7]

    numpy.testing.assert_raises(
            AssertionError,
            numpy.testing.assert_array_equal,
            output_sequence_1.data.numpy(),
            output_sequence_2.data.numpy(),
    )
    numpy.testing.assert_raises(
            AssertionError,
            numpy.testing.assert_array_equal,
            lstm_state_1[0].data.numpy(),
            lstm_state_2[0].data.numpy(),
    )
    numpy.testing.assert_raises(
            AssertionError,
            numpy.testing.assert_array_equal,
            lstm_state_1[1].data.numpy(),
            lstm_state_2[1].data.numpy(),
    )

    tmp_inputs = pack_padded_sequence(torch.rand(5, 6, 3), [6, 5, 4, 2, 1], batch_first=True)
    outputs, lstm_state_3 = lstm(
            tmp_inputs.data,
            tmp_inputs.batch_sizes,
    )
    output_sequence_3, _batch_sizes = pad_packed_sequence(
            PackedSequence(outputs[1], tmp_inputs.batch_sizes),
            batch_first=True,
    )
    assert list(output_sequence_3.size()) == [5, 6, 5]
    assert list(lstm_state_3[0].size()) == [2, 5, 5]
    assert list(lstm_state_3[1].size()) == [2, 5, 7]

    tmp_inputs = pack_padded_sequence(torch.rand(4, 6, 3), [6, 4, 2, 1], batch_first=True)
    outputs, lstm_state_4 = lstm(
            tmp_inputs.data,
            tmp_inputs.batch_sizes,
    )
    output_sequence_4, _batch_sizes = pad_packed_sequence(
            PackedSequence(outputs[1], tmp_inputs.batch_sizes),
            batch_first=True,
    )
    numpy.testing.assert_array_equal(
            lstm_state_4[0].data[:, 4:, :].numpy(),
            lstm_state_3[0].data[:, 4:, :].numpy(),
    )
    numpy.testing.assert_array_equal(
            lstm_state_4[1].data[:, 4:, :].numpy(),
            lstm_state_3[1].data[:, 4:, :].numpy(),
    )

    lstm.reset_states()
    outputs, lstm_state_5 = lstm(
            inputs.data,
            inputs.batch_sizes,
    )
    output_sequence_5, _batch_sizes = pad_packed_sequence(
            PackedSequence(outputs[1], inputs.batch_sizes),
            batch_first=True,
    )
    numpy.testing.assert_array_equal(
            output_sequence_1.data.numpy(),
            output_sequence_5.data.numpy(),
    )
    numpy.testing.assert_array_equal(
            lstm_state_1[0].data.numpy(),
            lstm_state_5[0].data.numpy(),
    )
    numpy.testing.assert_array_equal(
            lstm_state_1[1].data.numpy(),
            lstm_state_5[1].data.numpy(),
    )


def test_permutate_states():
    lstm = StatefulUnidirectionalLstm(
            num_layers=2,
            input_size=3,
            hidden_size=5,
            cell_size=7,
    )

    input_tensor = torch.rand(4, 5, 3)
    inputs = pack_padded_sequence(input_tensor, [5, 4, 2, 1], batch_first=True)
    lstm(inputs.data, inputs.batch_sizes)
    hidden_state_1 = lstm.managed_hidden_state()
    assert list(hidden_state_1.shape) == [2, 4, 5]

    lstm.permutate_states(torch.LongTensor([1, 0]))

    hidden_state_2 = lstm.managed_hidden_state()
    assert list(hidden_state_2.shape) == [2, 4, 5]

    numpy.testing.assert_raises(
            AssertionError,
            numpy.testing.assert_array_equal,
            hidden_state_1,
            hidden_state_2,
    )
    numpy.testing.assert_array_equal(
            hidden_state_1.narrow(1, 2, 2),
            hidden_state_2.narrow(1, 2, 2),
    )
