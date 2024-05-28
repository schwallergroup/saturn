
"""
Based on the implementation from https://github.com/MolecularAI/reinvent-models.
RNN model using LSTM cells. This is the default model in REINVENT versions 2.0, 3.2, and 4 for de novo small molecule generation.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    """
    Implements a N layer LSTM cell architecture consisting of:
        1. Embedding layer - maps tokenized SMILES to embedding
        2. LSTM cells - takes embedding and returns hidden/cell states and output
        3. Linear layer - maps RNN output back to the size of the vocabulary
        
    :param vocabulary_size: number of tokens in the vocabulary
    :param embedding_dim: dimension of the embedding layer
    :param hidden_dim: dimension of each of the LSTM cells
    :param num_layers: number of LSTM cells
    :param dropout: dropout rate. Would be applied to the outputs of each LSTM cell except the last one
    :param layer_normalization: whether to apply layer normalization to the RNN output

    GRU cells are not implemented, as LSTM cells have generally better performance for the SMILES generation task as explored here: 
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0393-0

    Default parameters with a Vocabulary size of 37 yields a 5,807,909 parameter model.
    """
    def __init__(
        self, 
        vocabulary_size: int, 
        embedding_dim: int = 256,
        hidden_dim: int = 512, 
        num_layers: int = 3, 
        dropout: float = 0.0,
        layer_normalization=False
    ):
        super(RNN, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_normalization = layer_normalization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim
        )
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            dropout=dropout, 
            batch_first=True  # Input and output tensors are (batch, sequence_length, feature)
        )
        self.linear = nn.Linear(hidden_dim, vocabulary_size)

    def forward(
        self, 
        input_vector: torch.Tensor, 
        hidden_state=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # pylint: disable=W0221
        """
        Performs a forward pass on the model. 
        Note: you pass the **whole** sequence.
        
        :param input_vector: Input tensor (batch_size, sequence_size).
        :param hidden_state: Hidden state tensor.
        """
        batch_size, sequence_length = input_vector.size()
        if hidden_state is None:
            size = (self.num_layers, batch_size, self.hidden_dim)
            hidden_state = [torch.zeros(*size, device=self.device), torch.zeros(*size, device=self.device)]
        
        # 1. Vocabulary indices to Embedding
        embedded_vector = self.embedding(input_vector)  # (batch, sequence_length, embedding_dim)

        # 2. Pass through LSTM cells
        output, hidden_state_out = self.rnn(embedded_vector, hidden_state)
        
        # 3. Apply layer normalization (if specified)
        if self.layer_normalization:
            output = F.layer_norm(output, output.size()[1:])

        # 4. Map LSTM output back to Vocabulary size
        output = output.reshape(-1, self.hidden_dim)  # (batch * sequence_length, hidden_dim)
        output = self.linear(output).view(batch_size, sequence_length, -1)

        return output, hidden_state_out

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        """
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout
        }
