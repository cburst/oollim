# hierarchical.py
"""Backbone-independent hierarchical pronunciation scoring on phone embeddings."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class HierarchicalScoringNetwork(nn.Module):
    def __init__(self, input_width: int, vocabulary_size: int, text_only: bool = False):
        super().__init__()
        if input_width <= 0 or vocabulary_size <= 0:
            raise ValueError("input_width and vocabulary_size must be positive")

        hidden = 64
        self.input_width = input_width
        self.vocabulary_size = vocabulary_size
        self.text_only = text_only

        self.audio_projection = nn.Linear(input_width, hidden)
        self.reference_embedding = nn.Embedding(vocabulary_size, hidden)
        self.phone_gru = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.phone_head = nn.Linear(hidden, 1)
        self.word_gru = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.word_head = nn.Linear(hidden, 2)
        self.utterance_head = nn.Linear(2 * hidden, 3)

    def forward(
        self,
        embeddings: torch.Tensor,
        phone_mask: torch.Tensor,
        phone_to_word: torch.Tensor,
        reference_ids: torch.Tensor,
        text_only: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.input_width:
            raise ValueError("embeddings must have shape [B,P,input_width]")
        batch, phones, _ = embeddings.shape
        shape = (batch, phones)
        if batch == 0 or phones == 0:
            raise ValueError("Every batch and phone dimension must be nonempty")
        if phone_mask.shape != shape or phone_mask.dtype != torch.bool:
            raise ValueError("phone_mask must be boolean [B,P]")
        if phone_to_word.shape != shape or phone_to_word.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise ValueError("phone_to_word must be integer [B,P]")
        if reference_ids.shape != shape or reference_ids.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise ValueError("reference_ids must be integer [B,P]")
        if not (
            embeddings.device == phone_mask.device
            == phone_to_word.device == reference_ids.device
        ):
            raise ValueError("All inputs must be on the same device")

        lengths = phone_mask.sum(dim=1)
        if (lengths == 0).any() or (phone_mask[:, 1:] & ~phone_mask[:, :-1]).any():
            raise ValueError("Each row needs a nonempty, contiguous valid phone prefix")
        if (phone_to_word[~phone_mask] != -1).any():
            raise ValueError("Padded phone_to_word entries must be -1")

        valid_words = phone_to_word[phone_mask]
        if (valid_words < 0).any():
            raise ValueError("Valid phones need nonnegative word IDs")
        starts = torch.zeros_like(phone_mask)
        starts[:, 0] = True
        if (phone_to_word[phone_mask & starts] != 0).any():
            raise ValueError("Word IDs must start at zero")
        following = phone_mask[:, 1:]
        increments = phone_to_word[:, 1:] - phone_to_word[:, :-1]
        if ((increments[following] < 0) | (increments[following] > 1)).any():
            raise ValueError("Word IDs must be contiguous and nondecreasing")

        valid_refs = reference_ids[phone_mask]
        if ((valid_refs < 0) | (valid_refs >= self.vocabulary_size)).any():
            raise ValueError("Valid reference IDs must be in the vocabulary")
        if not torch.isfinite(embeddings[phone_mask]).all():
            raise ValueError("Valid embeddings must be finite")

        use_text_only = self.text_only if text_only is None else text_only
        safe_refs = reference_ids.masked_fill(~phone_mask, 0).long()
        phone_input = self.reference_embedding(safe_refs)
        if not use_text_only:
            safe_audio = torch.where(
                phone_mask.unsqueeze(-1), embeddings, torch.zeros_like(embeddings)
            )
            phone_input = phone_input + self.audio_projection(safe_audio)
        phone_input = phone_input.masked_fill(~phone_mask.unsqueeze(-1), 0)

        packed = pack_padded_sequence(
            phone_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_states, _ = self.phone_gru(packed)
        phone_states, _ = pad_packed_sequence(
            packed_states, batch_first=True, total_length=phones
        )
        phone_scores = self.phone_head(phone_states).squeeze(-1)
        phone_scores = phone_scores.masked_fill(~phone_mask, 0)

        word_lengths = phone_to_word.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1) + 1
        word_count = int(word_lengths.max().item())
        word_mask = (
            torch.arange(word_count, device=embeddings.device).unsqueeze(0)
            < word_lengths.unsqueeze(1)
        )

        indices = phone_to_word.clamp_min(0).long()
        sums = phone_states.new_zeros(batch, word_count, 64)
        sums.scatter_add_(
            1, indices.unsqueeze(-1).expand(-1, -1, 64),
            phone_states.masked_fill(~phone_mask.unsqueeze(-1), 0),
        )
        counts = phone_states.new_zeros(batch, word_count)
        counts.scatter_add_(1, indices, phone_mask.to(phone_states.dtype))
        word_input = sums / counts.clamp_min(1).unsqueeze(-1)

        packed_words = pack_padded_sequence(
            word_input, word_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_word_states, _ = self.word_gru(packed_words)
        word_states, _ = pad_packed_sequence(
            packed_word_states, batch_first=True, total_length=word_count
        )
        word_scores = self.word_head(word_states).masked_fill(
            ~word_mask.unsqueeze(-1), 0
        )

        valid_states = word_states.masked_fill(~word_mask.unsqueeze(-1), 0)
        denominators = word_lengths.to(word_states.dtype).unsqueeze(-1)
        mean = valid_states.sum(dim=1) / denominators
        variance = (
            ((word_states - mean.unsqueeze(1)) ** 2)
            .masked_fill(~word_mask.unsqueeze(-1), 0)
            .sum(dim=1)
            / denominators
        )
        std = torch.sqrt(variance + 1e-6)
        utterance_scores = self.utterance_head(torch.cat((mean, std), dim=-1))

        return {
            "utterance": utterance_scores,
            "phone": phone_scores,
            "word": word_scores,
            "phone_mask": phone_mask,
            "word_mask": word_mask,
        }


def hierarchical_loss(
    outputs: dict[str, torch.Tensor],
    utterance_targets: torch.Tensor,
    phone_targets: torch.Tensor | None = None,
    word_targets: torch.Tensor | None = None,
    auxiliary_weight: float = 0.1,
) -> torch.Tensor:
    """SmoothL1 loss; NaN auxiliary targets are omitted, not replaced."""
    utterance = outputs["utterance"]
    phone = outputs["phone"]
    word = outputs["word"]
    phone_mask = outputs["phone_mask"]
    word_mask = outputs["word_mask"]

    if utterance_targets.shape != utterance.shape or utterance.ndim != 2 or utterance.shape[1] != 3:
        raise ValueError("utterance_targets must match utterance outputs [B,3]")
    if not torch.isfinite(utterance_targets).all():
        raise ValueError("Utterance targets must be present and finite")

    loss = F.smooth_l1_loss(utterance, utterance_targets, beta=0.1)

    for predictions, targets, mask, name in (
        (phone, phone_targets, phone_mask, "phone"),
        (word, word_targets, word_mask.unsqueeze(-1), "word"),
    ):
        if targets is None:
            continue
        if targets.shape != predictions.shape:
            raise ValueError(f"{name} targets must match {name} outputs")
        if torch.isinf(targets).any():
            raise ValueError(f"{name} targets must not contain infinity")
        selected = torch.isfinite(targets) & mask
        if selected.any():
            loss = loss + auxiliary_weight * F.smooth_l1_loss(
                predictions[selected], targets[selected], beta=0.1
            )
    return loss
