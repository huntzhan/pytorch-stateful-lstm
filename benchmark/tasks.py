from invoke import task
import numpy
import random
import torch
import time
import json
import glob
from os.path import basename
import statistics

from pytorch_stateful_lstm import (
    UnidirectionalSingleLayerLstm,
    UnidirectionalLstm,
    StatefulUnidirectionalLstm,
)
from allennlp.modules.lstm_cell_with_projection import LstmCellWithProjection
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper


random.seed(13370)
numpy.random.seed(1337)
torch.manual_seed(133)


@task
def uni_sl_lstm(c, input, hidden, cell, batch, timestep, repeat, cuda, output):
    input = int(input)
    hidden = int(hidden)
    cell = int(cell)
    batch = int(batch)
    timestep = int(timestep)
    repeat = int(repeat)

    lstm = UnidirectionalSingleLayerLstm(
            input_size=input,
            hidden_size=hidden,
            cell_size=cell,
    )
    input_tensor = torch.rand(batch, timestep, input)

    initial_hidden_state = torch.ones([1, batch, hidden])
    initial_cell_state = torch.ones([1, batch, cell])

    if cuda == 'cuda':
        lstm.cuda()
        input_tensor = input_tensor.cuda()
        initial_hidden_state = initial_hidden_state.cuda()
        initial_cell_state = initial_cell_state.cuda()

    durations = []

    for idx in range(repeat):
        batch_lengths = [timestep]
        batch_lengths.extend([random.randrange(timestep + 1) for _ in range(batch - 1)])
        batch_lengths = sorted(batch_lengths, reverse=True)

        with torch.no_grad():
            time_start = time.time()
            lstm.forward(
                    input_tensor,
                    batch_lengths,
                    (initial_hidden_state, initial_cell_state),
            )
            durations.append(
                    (idx, time.time() - time_start),
            )

    with open(output, 'w') as fout:
        json.dump(
                {'type': 'uni_sl_lstm', 'cuda': cuda, 'durations': durations},
                fout,
                ensure_ascii=False,
                indent=2,
        )


@task
def allennlp_lstm_cell(c, input, hidden, cell, batch, timestep, repeat, cuda, output):
    input = int(input)
    hidden = int(hidden)
    cell = int(cell)
    batch = int(batch)
    timestep = int(timestep)
    repeat = int(repeat)

    lstm = LstmCellWithProjection(
            input_size=input,
            hidden_size=hidden,
            cell_size=cell,
    )
    input_tensor = torch.rand(batch, timestep, input)

    initial_hidden_state = torch.ones([1, batch, hidden])
    initial_cell_state = torch.ones([1, batch, cell])

    if cuda == 'cuda':
        lstm = lstm.cuda()
        input_tensor = input_tensor.cuda()
        initial_hidden_state = initial_hidden_state.cuda()
        initial_cell_state = initial_cell_state.cuda()

    durations = []
    for idx in range(repeat):
        batch_lengths = [timestep]
        batch_lengths.extend([random.randrange(timestep + 1) for _ in range(batch - 1)])
        batch_lengths = sorted(batch_lengths, reverse=True)

        with torch.no_grad():
            time_start = time.time()
            lstm(
                    input_tensor,
                    batch_lengths,
                    (initial_hidden_state, initial_cell_state),
            )
            durations.append(
                    (idx, time.time() - time_start),
            )

    with open(output, 'w') as fout:
        json.dump(
                {'type': 'allennlp_lstm_cell', 'cuda': cuda, 'durations': durations},
                fout,
                ensure_ascii=False,
                indent=2,
        )


@task
def uni_lstm(c, num_layers, input, hidden, cell, batch, timestep, repeat, cuda, output):
    num_layers = int(num_layers)
    input = int(input)
    hidden = int(hidden)
    cell = int(cell)
    batch = int(batch)
    timestep = int(timestep)
    repeat = int(repeat)

    lstm = StatefulUnidirectionalLstm(
            num_layers=num_layers,
            input_size=input,
            hidden_size=hidden,
            cell_size=cell,
    )
    input_tensor = torch.rand(batch, timestep, input)
    if cuda == 'cuda':
        lstm.cuda()
        input_tensor = input_tensor.cuda()

    durations = []

    for idx in range(repeat):
        batch_lengths = [timestep]
        batch_lengths.extend([random.randrange(timestep + 1) for _ in range(batch - 1)])
        batch_lengths = sorted(batch_lengths, reverse=True)

        with torch.no_grad():
            time_start = time.time()
            lstm.forward(
                    input_tensor,
                    batch_lengths,
            )
            durations.append(
                    (idx, time.time() - time_start),
            )

    with open(output, 'w') as fout:
        json.dump(
                {'type': 'uni_lstm', 'cuda': cuda, 'durations': durations},
                fout,
                ensure_ascii=False,
                indent=2,
        )


@task
def allennlp_seq2seq(c, num_layers, input, hidden, cell, batch, timestep, repeat, cuda, output):
    num_layers = int(num_layers)
    input = int(input)
    hidden = int(hidden)
    cell = int(cell)
    batch = int(batch)
    timestep = int(timestep)
    repeat = int(repeat)

    lstms = []
    lstm_input = input
    for _ in range(num_layers):
        lstms.append(PytorchSeq2SeqWrapper(AugmentedLstm(
                input_size=lstm_input,
                hidden_size=hidden,
                use_highway=False,
                use_input_projection_bias=False,
        ), stateful=True))
        lstm_input = hidden

    input_tensor = torch.rand(batch, timestep, input)
    if cuda == 'cuda':
        input_tensor = input_tensor.cuda()
        lstms = [l.cuda() for l in lstms]

    durations = []
    for idx in range(repeat):
        batch_lengths = [timestep]
        batch_lengths.extend([random.randrange(timestep + 1) for _ in range(batch - 1)])
        batch_lengths = sorted(batch_lengths, reverse=True)

        mask = torch.zeros(batch, timestep, dtype=torch.long)
        for mask_idx, length in enumerate(batch_lengths):
            mask[mask_idx, :length] = 1
        if cuda == 'cuda':
            mask = mask.cuda()

        with torch.no_grad():
            time_start = time.time()
            lstm_input = input_tensor
            for lstm in lstms:
                lstm_input = lstm(
                        lstm_input,
                        mask,
                )
            durations.append(
                    (idx, time.time() - time_start),
            )

    with open(output, 'w') as fout:
        json.dump(
                {'type': 'allennlp_seq2seq', 'cuda': cuda, 'durations': durations},
                fout,
                ensure_ascii=False,
                indent=2,
        )


@task
def plot(c, prefix, keywords, output):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    keywords = eval(keywords)
    name_path_set = []
    for path in glob.glob(prefix):
        name = None
        filename = basename(path)
        for kw_set in keywords:
            match = True
            for kw in kw_set:
                if kw not in filename:
                    match = False
                    break
            if match:
                name = '-'.join(kw_set)
                break
        if name:
            name_path_set.append((name, path))

    print(name_path_set)
    for name, path in name_path_set:
        with open(path) as fin:
            data = json.load(fin)

        x = []
        y = []

        acc = 0.
        for idx, drt in data['durations']:
            acc += drt
            x.append(idx)
            y.append(acc)

        plt.plot(x, y)

        print(name)
        print(statistics.mean(drt for _, drt in data['durations']))

    plt.legend([name for name, _ in name_path_set], loc='upper left')
    plt.xlabel('Batch Iterations')
    plt.ylabel('Duration In Seconds')
    plt.savefig(output)
