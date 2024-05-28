import torch
import torch.nn as nn

from utils.utils import generate_causal_mask


class Decoder(nn.Module):
    """
    Implements a Decoder-only Transformer architecture based on GPT-2.

    :param vocabulary_size: number of tokens in the vocabulary
    :param embedding_dim: dimension of the embedding layer
    :param hidden_dim: dimension of the hidden layer
    :param num_layers: number of Decoder blocks
    :param num_heads: number of heads for Multi-Head Attention (MHA)
    :param dropout: dropout rate

    Default parameters with a Vocabulary size of 37 yields a 6,337,061 parameter model. 
    Using REINVENT's loss function, this model size has been shown to work here:
    https://openreview.net/pdf?id=1B6YKnHYBb
    """
    def __init__(
        self, 
        vocabulary_size: int, 
        embedding_dim: int = 256, 
        hidden_dim: int = 1024, 
        num_layers: int = 8, 
        num_heads: int = 16, 
        dropout: float = 0.0
    ):
        super(Decoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.decoder_blocks = nn.ModuleList([
            DecoderLayer(
                embedding_dim=embedding_dim, 
                num_heads=num_heads, 
                hidden_dim=hidden_dim, 
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(embedding_dim, vocabulary_size)
        
    def forward(
        self, 
        input_vector: torch.Tensor, 
        src_mask=None
    ) -> torch.Tensor:
        batch_size, sequence_length = input_vector.size()

        # 1. Vocabulary indices to Embedding
        x = self.embedding(input_vector)  # (batch, sequence_length, embedding_dim)

        # 2. Add Positional Encoding to Embedding
        x = self.positional_encoding(x)  # (batch, sequence_length, embedding_dim)

        # 3. Apply Dropout
        x = self.dropout_layer(x)

        # 4. Pass through Decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, src_mask)  # (batch, sequence_length, embedding_dim)

        # 5. Map back to Vocabulary size
        x = x.reshape(-1, self.embedding_dim)  # (batch * sequence_length, embedding_dim)
        x = self.linear(x).view(batch_size, sequence_length, -1)  # (batch, sequence_length, vocabulary_size)

        return x

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        """
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout
        }

class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        max_sequence_length=128
    ):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_sequence_length, embedding_dim)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        # Add Positional Encoding to the Embedding
        return x + self.pe[:x.size(1), :]

class DecoderLayer(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        num_heads: int, 
        hidden_dim: int, 
        dropout: float = 0.0
    ):
        super(DecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )  # Takes as input (batch, sequence_length, embedding_dim)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, src_mask=None):
        if src_mask is not None:
            attention_mask = src_mask  # For generation
        else:
            attention_mask = generate_causal_mask(
                size=x.size(1), 
                device=x.device
            ) # (sequence_length, sequence_length)

        # Layer Normalization before Self-Attention
        attention_output, _ = self.self_attention(
            query=self.layer_norm_1(x), 
            key=self.layer_norm_1(x), 
            value=self.layer_norm_1(x), 
            attn_mask=attention_mask
        )  # returns (output, attention_weights)

        x = x + attention_output
        x = x + self.feed_forward(self.layer_norm_2(x))

        return x  # (batch, sequence_length, embedding_dim)
