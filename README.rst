=====================
pytorch-stateful-lstm
=====================


.. image:: https://img.shields.io/pypi/v/pytorch_stateful_lstm.svg
        :target: https://pypi.python.org/pypi/pytorch_stateful_lstm

.. image:: https://img.shields.io/travis/cnt-dev/pytorch-stateful-lstm.svg
        :target: https://travis-ci.org/cnt-dev/pytorch-stateful-lstm

* Free software: MIT license

Features
--------

Pytorch LSTM implementation powered by Libtorch, and with the support of:

- Hidden/Cell Clip.
- Variational Dropout & DropConnect.
- Managed Initial State.
- Built-in TBPTT.

Benchmark: https://github.com/cnt-dev/pytorch-stateful-lstm/tree/master/benchmark

Install
-------

Prerequisite: `torch>=1.0.0`, supported C++11 compiler (see here_). To install through pip::

    pip install pytorch-stateful-lstm

.. _here: https://github.com/pytorch/pytorch/blob/0bf1383f0a6caa34945feaf19191986d18205251/torch/utils/cpp_extension.py#L169-L181

Usage
-----

Example::

    from pytorch_stateful_lstm import StatefulUnidirectionalLstm

    lstm = StatefulUnidirectionalLstm(
            num_layers=2,
            input_size=3,
            hidden_size=5,
            cell_size=7,
    )

    output_sequence, lstm_state = lstm(
            input_tensor,
            [5, 4, 2, 1],
    )

For the definition of parameters, see https://github.com/cnt-dev/pytorch-stateful-lstm/tree/master/extension.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
