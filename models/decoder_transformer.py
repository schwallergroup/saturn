
"""
Based on the implementation from https://github.com/karpathy/nanoGPT
"""
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from models.vocabulary import Vocabulary, SMILESTokenizer


class DecoderTransformer(nn.Module):
    """
    Implements a decoder-only transformer model (based on GPT-2 architecture).
    """
    # TODO: implement

    def __init__(
        self, 
        voc_size, 
        layer_size: int = 512, 
        num_layers: int = 3, 
        cell_type: str = "lstm", 
        embedding_layer_size: int = 256, 
        dropout: float = 0.0,
        layer_normalization=False
    ):
        """
        Implements a N layer GRU|LSTM cell including an embedding layer and an output linear layer back to the size of the
        vocabulary
        :param voc_size: Size of the vocabulary.
        :param layer_size: Size of each of the RNN layers.
        :param num_layers: Number of RNN layers.
        :param embedding_layer_size: Size of the embedding layer.
        """
        super(DecoderTransformer, self).__init__()

        self._layer_size = layer_size
        self._embedding_layer_size = embedding_layer_size
        self._num_layers = num_layers
        self._cell_type = cell_type.lower()
        self._dropout = dropout
        self._layer_normalization = layer_normalization

        self._embedding = nn.Embedding(voc_size, self._embedding_layer_size)
        if self._cell_type == "gru":
            self._rnn = nn.GRU(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                                dropout=self._dropout, batch_first=True)
        elif self._cell_type == "lstm":
            self._rnn = nn.LSTM(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                                 dropout=self._dropout, batch_first=True)
        else:
            raise ValueError('Value of the parameter cell_type should be "gru" or "lstm"')
        self._linear = nn.Linear(self._layer_size, voc_size)

    def forward(self, input_vector, hidden_state=None):  # pylint: disable=W0221
        """
        Performs a forward pass on the model. Note: you pass the **whole** sequence.
        :param input_vector: Input tensor (batch_size, seq_size).
        :param hidden_state: Hidden state tensor.
        """
        batch_size, seq_size = input_vector.size()
        if hidden_state is None:
            size = (self._num_layers, batch_size, self._layer_size)
            if self._cell_type == "gru":
                hidden_state = torch.zeros(*size)
            else:
                hidden_state = [torch.zeros(*size), torch.zeros(*size)]
        embedded_data = self._embedding(input_vector)  # (batch,seq,embedding)
        output_vector, hidden_state_out = self._rnn(embedded_data, hidden_state)

        if self._layer_normalization:
            output_vector = F.layer_norm(output_vector, output_vector.size()[1:])
        output_vector = output_vector.reshape(-1, self._layer_size)

        output_data = self._linear(output_vector).view(batch_size, seq_size, -1)

        return output_data, hidden_state_out

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        """
        return {
            'dropout': self._dropout,
            'layer_size': self._layer_size,
            'num_layers': self._num_layers,
            'cell_type': self._cell_type,
            'embedding_layer_size': self._embedding_layer_size
        }