import torch
import numpy

from pytorch_stateful_lstm import StatefulUnidirectionalLstm


def test_stateful_unidirectional_lstm():
    input_tensor = torch.rand(4, 5, 3)
    input_tensor[1, 4:, :] = 0.
    input_tensor[2, 2:, :] = 0.
    input_tensor[3, 1:, :] = 0.

    lstm = StatefulUnidirectionalLstm(
            num_layers=2,
            input_size=3,
            hidden_size=5,
            cell_size=7,
    )

    output_sequence_1, lstm_state_1 = lstm(
            input_tensor,
            [5, 4, 2, 1],
    )
    assert list(output_sequence_1.size()) == [2, 4, 5, 5]
    assert list(lstm_state_1[0].size()) == [2, 4, 5]
    assert list(lstm_state_1[1].size()) == [2, 4, 7]

    output_sequence_2, lstm_state_2 = lstm(
            input_tensor,
            [5, 4, 2, 1],
    )
    assert list(output_sequence_2.size()) == [2, 4, 5, 5]
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

    output_sequence_3, lstm_state_3 = lstm(
            torch.rand(8, 6, 3),
            [5, 4, 2, 1],
    )
    assert list(output_sequence_3.size()) == [2, 8, 6, 5]
    assert list(lstm_state_3[0].size()) == [2, 8, 5]
    assert list(lstm_state_3[1].size()) == [2, 8, 7]

    output_sequence_4, lstm_state_4 = lstm(
            torch.rand(4, 6, 3),
            [5, 4, 2, 1],
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
    output_sequence_5, lstm_state_5 = lstm(
            input_tensor,
            [5, 4, 2, 1],
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
