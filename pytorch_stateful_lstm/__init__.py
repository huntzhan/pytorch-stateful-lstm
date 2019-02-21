# -*- coding: utf-8 -*-
"""Top-level package for pytorch-stateful-lstm."""

__author__ = """Hunt Zhan"""
__email__ = 'huntzhan.dev@gmail.com'
__version__ = '1.5.1'

# To avoid `undefined symbol` error.
import torch
# pylint: disable=no-name-in-module
from _pytorch_stateful_lstm import (
        UnidirectionalSingleLayerLstm,
        UnidirectionalLstm,
        StatefulUnidirectionalLstm,
)
