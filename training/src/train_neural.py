"""Training loop for `TinyGRU` and safetensors export.

Kept deliberately simple (sequence-at-a-time, no padding/batching machinery)
since Stage 1 datasets are small (tens to low-hundreds of short monophonic
tunes per genre) and the model is tiny — this comfortably finishes in
seconds to low minutes on a CPU, matching the spec's "fast training on a
standard CPU" requirement.
"""
from __future__ import annotations

import logging

import torch
from safetensors.torch import save_file
from torch import nn

from .neural_model import TinyGRU, TinyGRUConfig
from .tokens import GENRE_TO_ID

logger = logging.getLogger(__name__)


def _tune_to_tensors(genre_id: int, pitch_seq: list[int], dur_seq: list[int], pos_seq: list[int]):
    genre_t = torch.tensor([genre_id], dtype=torch.long)
    pitch_t = torch.tensor([pitch_seq], dtype=torch.long)
    dur_t = torch.tensor([dur_seq], dtype=torch.long)
    pos_t = torch.tensor([pos_seq], dtype=torch.long)
    return genre_t, pitch_t, dur_t, pos_t


def train_tiny_gru(
    sequences_by_genre: dict[str, list[list[tuple[int, int, int]]]],
    config: TinyGRUConfig | None = None,
    epochs: int = 5,
    learning_rate: float = 2e-3,
    min_tune_length: int = 3,
    seed: int = 0,
) -> TinyGRU:
    """`sequences_by_genre[genre_name]` is a list of tunes; each tune is a
    list of `(pitch_bucket, dur_bucket, pos_bucket)` triples in order."""
    torch.manual_seed(seed)
    model = TinyGRU(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pitch_loss_fn = nn.CrossEntropyLoss()
    dur_loss_fn = nn.CrossEntropyLoss()

    training_tunes: list[tuple[int, list[int], list[int], list[int]]] = []
    for genre_name, tunes in sequences_by_genre.items():
        if genre_name not in GENRE_TO_ID:
            logger.warning("Skipping unknown genre %r during neural training", genre_name)
            continue
        genre_id = GENRE_TO_ID[genre_name]
        for tune in tunes:
            if len(tune) < min_tune_length:
                continue
            pitch_seq = [t[0] for t in tune]
            dur_seq = [t[1] for t in tune]
            pos_seq = [t[2] for t in tune]
            training_tunes.append((genre_id, pitch_seq, dur_seq, pos_seq))

    if not training_tunes:
        raise ValueError("No tunes long enough to train on (need >= %d notes)" % min_tune_length)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for genre_id, pitch_seq, dur_seq, pos_seq in training_tunes:
            genre_t, pitch_t, dur_t, pos_t = _tune_to_tensors(genre_id, pitch_seq, dur_seq, pos_seq)

            input_pitch, input_dur, input_pos = pitch_t[:, :-1], dur_t[:, :-1], pos_t[:, :-1]
            target_pitch, target_dur = pitch_t[:, 1:], dur_t[:, 1:]

            optimizer.zero_grad()
            pitch_logits, dur_logits, _ = model(genre_t, input_pitch, input_dur, input_pos)
            loss = pitch_loss_fn(
                pitch_logits.reshape(-1, pitch_logits.size(-1)), target_pitch.reshape(-1)
            ) + dur_loss_fn(dur_logits.reshape(-1, dur_logits.size(-1)), target_dur.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(
            "epoch %d/%d avg_loss=%.4f (%d tunes)",
            epoch + 1,
            epochs,
            total_loss / len(training_tunes),
            len(training_tunes),
        )

    model.eval()
    return model


def export_safetensors(model: TinyGRU, path: str) -> None:
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, path, metadata=model.config.as_metadata())
