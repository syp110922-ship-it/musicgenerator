"""Compact 2-layer Tiny-GRU sequence model.

Sized to stay small enough to train quickly on a CPU and run with low
latency once ported to the browser (WASM, Stage 2): a few hundred thousand
parameters at the default sizes below, well under the ~1.5M parameter
ballpark target in the spec, leaving headroom to grow the hidden size later
without redesigning the pipeline.

Inputs at every step are four small embeddings (genre, pitch-interval
bucket, duration bucket, bar-position bucket) concatenated and projected
down to `hidden_size`, matching the "(genre token, pitch interval, note
length, beat position)" tokenization described in the spec. The beat
position is a *conditioning* feature (derivable deterministically while
generating) rather than something the network predicts, so the two output
heads only predict the next pitch bucket and the next duration bucket.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from .tokens import NUM_DUR_BUCKETS, NUM_GENRES, NUM_PITCH_BUCKETS, NUM_POS_BUCKETS


@dataclass(frozen=True)
class TinyGRUConfig:
    hidden_size: int = 128
    genre_dim: int = 8
    pitch_dim: int = 24
    dur_dim: int = 12
    pos_dim: int = 8
    num_layers: int = 2

    def as_metadata(self) -> dict[str, str]:
        """safetensors metadata values must be strings."""
        return {k: str(v) for k, v in asdict(self).items()} | {
            "num_genres": str(NUM_GENRES),
            "num_pitch_buckets": str(NUM_PITCH_BUCKETS),
            "num_dur_buckets": str(NUM_DUR_BUCKETS),
            "num_pos_buckets": str(NUM_POS_BUCKETS),
        }


class TinyGRU(nn.Module):
    def __init__(self, config: TinyGRUConfig | None = None):
        super().__init__()
        self.config = config or TinyGRUConfig()
        c = self.config

        self.genre_embedding = nn.Embedding(NUM_GENRES, c.genre_dim)
        self.pitch_embedding = nn.Embedding(NUM_PITCH_BUCKETS, c.pitch_dim)
        self.dur_embedding = nn.Embedding(NUM_DUR_BUCKETS, c.dur_dim)
        self.pos_embedding = nn.Embedding(NUM_POS_BUCKETS, c.pos_dim)

        input_dim = c.genre_dim + c.pitch_dim + c.dur_dim + c.pos_dim
        self.input_proj = nn.Linear(input_dim, c.hidden_size)
        self.gru = nn.GRU(c.hidden_size, c.hidden_size, num_layers=c.num_layers, batch_first=True)
        self.pitch_head = nn.Linear(c.hidden_size, NUM_PITCH_BUCKETS)
        self.dur_head = nn.Linear(c.hidden_size, NUM_DUR_BUCKETS)

    def forward(
        self,
        genre_ids: torch.Tensor,     # [B] or [B, T]
        pitch_ids: torch.Tensor,     # [B, T]
        dur_ids: torch.Tensor,       # [B, T]
        pos_ids: torch.Tensor,       # [B, T]
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if genre_ids.dim() == 1:
            genre_ids = genre_ids.unsqueeze(1).expand(-1, pitch_ids.size(1))

        x = torch.cat(
            [
                self.genre_embedding(genre_ids),
                self.pitch_embedding(pitch_ids),
                self.dur_embedding(dur_ids),
                self.pos_embedding(pos_ids),
            ],
            dim=-1,
        )
        x = self.input_proj(x)
        out, hidden = self.gru(x, hidden)
        return self.pitch_head(out), self.dur_head(out), hidden

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
