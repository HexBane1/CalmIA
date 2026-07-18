"""
Developer A work package: Baseline neural network architecture.

Implements a lightweight autoregressive Transformer decoder for symbolic music
token sequences. This is the class referenced by shared.config.ModelConfig
.model_class_name -- do not rename without updating the shared config and
notifying Developer B.

An LSTM alternative is included at the bottom of this file, commented out, as a
faster-to-train drop-in if the Transformer proves too slow for Week 1 iteration
speed on Robert's dataset.
"""

import math

import torch
import torch.nn as nn

from shared.config import DATA_CONFIG, MODEL_CONFIG


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class MusicSequenceModel(nn.Module):
    """
    Autoregressive Transformer decoder for next-token prediction over the
    event-based music vocabulary defined in shared.vocabulary.

    Forward pass returns raw logits over the vocabulary at every position. Causal
    masking is applied internally so this can be used directly for both teacher-
    forced training and, in developer_b/sampler.py, incremental generation.
    """

    def __init__(
        self,
        vocab_size: int = None,
        d_model: int = None,
        n_heads: int = None,
        n_layers: int = None,
        d_feedforward: int = None,
        dropout: float = None,
        max_sequence_length: int = None,
        pad_token_id: int = None,
    ):
        super().__init__()
        vocab_size = vocab_size or DATA_CONFIG.vocab_size
        d_model = d_model or MODEL_CONFIG.d_model
        n_heads = n_heads or MODEL_CONFIG.n_heads
        n_layers = n_layers or MODEL_CONFIG.n_layers
        d_feedforward = d_feedforward or MODEL_CONFIG.d_feedforward
        dropout = dropout if dropout is not None else MODEL_CONFIG.dropout
        max_sequence_length = max_sequence_length or DATA_CONFIG.max_sequence_length
        self.pad_token_id = pad_token_id if pad_token_id is not None else DATA_CONFIG.pad_token_id

        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_token_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_sequence_length + 8)
        self.embedding_dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Using TransformerEncoder with a causal mask to implement decoder-only
        # (GPT-style) behavior, which avoids the unnecessary cross-attention
        # machinery of nn.TransformerDecoder for an unconditional baseline.
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying between input embedding and output projection is a common
        # and effective regularizer for autoregressive sequence models.
        self.output_projection.weight = self.token_embedding.weight

    @staticmethod
    def _generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # Boolean mask (True = disallowed) so it shares a dtype with the boolean
        # key-padding mask, avoiding a PyTorch deprecation warning that occurs
        # when a float attention mask is combined with a bool padding mask.
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) long tensor of token ids.
            padding_mask: (batch, seq_len) bool tensor, True where padded. Optional.

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.embedding_dropout(x)

        causal_mask = self._generate_causal_mask(seq_len, device)
        x = self.decoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.output_norm(x)
        logits = self.output_projection(x)
        return logits


# ---------------------------------------------------------------------------
# LSTM alternative (commented out). Enable this instead of the Transformer above
# if Week 1 iteration speed on Robert's dataset requires a cheaper baseline.
# Keep the class name MusicSequenceModel and the forward() signature identical
# so no changes are required in train.py or developer_b/.
# ---------------------------------------------------------------------------
#
# class MusicSequenceModel(nn.Module):
#     def __init__(self, vocab_size=None, d_model=None, n_layers=None, dropout=None,
#                  pad_token_id=None, **_ignored):
#         super().__init__()
#         vocab_size = vocab_size or DATA_CONFIG.vocab_size
#         d_model = d_model or MODEL_CONFIG.d_model
#         n_layers = n_layers or MODEL_CONFIG.n_layers
#         dropout = dropout if dropout is not None else MODEL_CONFIG.dropout
#         self.pad_token_id = pad_token_id if pad_token_id is not None else DATA_CONFIG.pad_token_id
#
#         self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_token_id)
#         self.lstm = nn.LSTM(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=n_layers,
#             dropout=dropout if n_layers > 1 else 0.0,
#             batch_first=True,
#         )
#         self.output_projection = nn.Linear(d_model, vocab_size)
#
#     def forward(self, input_ids, padding_mask=None):
#         x = self.token_embedding(input_ids)
#         x, _ = self.lstm(x)
#         return self.output_projection(x)
