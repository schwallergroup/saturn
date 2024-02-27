import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    """
    Implements a Decoder-only Transformer architecture based on GPT-2.

    :param vocabulary_size: number of tokens in the vocabulary
    :param embedding_dim: dimension of the embedding layer
    :param hidden_dim: dimension of the hidden layer
    :param num_layers: number of Decoder blocks
    :param num_heads: number of heads for Multi-Head Attention (MHA)
    :param dropout: dropout rate
    """
    def __init__(
        self, 
        vocabulary_size: int, 
        embedding_dim: int = 256, 
        hidden_dim: int = 512, 
        num_layers: int = 3, 
        num_heads: int = 12, 
        dropout: float = 0.0
    ):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
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
        batch_size, seq_size = input_vector.size()
        # get embeddings
        x = self.embedding(input_vector)  # (batch, sequence_length, embedding_dim)
        x = self.positional_encoding(x)  # (batch, sequence_length, embedding_dim)
        # pass through Decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, src_mask)
        # map back to vocabulary size
        x = x.reshape(-1, self.embedding_dim)
        return self.linear(x).view(batch_size, seq_size, -1)

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
        max_sequence_length=256
    ):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_sequence_length, embedding_dim)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

class DecoderLayer(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        num_heads: int, 
        hidden_dim: int, 
        dropout: float = 0.0
    ):
        super(DecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_mask=None):
        attention_mask = None
        if src_mask is not None:
            attention_mask = src_mask.unsqueeze(0)

        x, _ = self.self_attention(x, x, x, attn_mask=attention_mask)  # returns (output, attention_weights)
        x = self.layer_norm_1(x + self.dropout(x))
        x = self.feed_forward(x)
        x = self.layer_norm_2(x + self.dropout(x))
        return x
