"""Replaceable CTC alignment prototype. Spans are tentative, not clinical labels."""
from typing import Protocol
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Alignment:
    status: str
    spans: list
    log_likelihood: float | None
    reason: str | None = None


class Aligner(Protocol):
    def align(self, log_probs: torch.Tensor, expected_ids: list[int], blank_id: int) -> Alignment: ...


class CTCAligner:
    def align(self, log_probs, expected_ids, blank_id):
        if log_probs.ndim != 2 or not torch.isfinite(log_probs).all():
            raise ValueError('Expected finite log-probabilities [frames,vocabulary]')
        t, v = log_probs.shape
        if not 0 <= blank_id < v or not expected_ids or any(x == blank_id or not 0 <= x < v for x in expected_ids):
            return Alignment('unsupported', [], None, 'empty_or_invalid_reference_tokens')
        if t < len(expected_ids) + sum(a == b for a, b in zip(expected_ids, expected_ids[1:])):
            return Alignment('unalignable', [], None, 'insufficient_frames')
        sequence = [blank_id]
        for token in expected_ids:
            sequence.extend([token, blank_id])
        n = len(sequence)
        lp = log_probs.detach().cpu().numpy()
        prev = np.full(n, -np.inf)
        prev[0], prev[1] = lp[0, blank_id], lp[0, expected_ids[0]]
        back = np.full((t, n), -1, dtype=np.int32)
        for frame in range(1, t):
            current = np.full(n, -np.inf)
            for state, token in enumerate(sequence):
                candidates = [state]
                if state > 0:
                    candidates.append(state - 1)
                if state > 1 and token != blank_id and token != sequence[state - 2]:
                    candidates.append(state - 2)
                best = max(candidates, key=lambda s: prev[s])
                current[state] = prev[best] + lp[frame, token]
                back[frame, state] = best
            prev = current
        state = n - 1 if prev[-1] > prev[-2] else n - 2
        score = float(prev[state])
        if not np.isfinite(score):
            return Alignment('unalignable', [], None, 'no_finite_path')
        path = [state]
        for frame in range(t - 1, 0, -1):
            state = int(back[frame, state])
            path.append(state)
        path = np.array(path[::-1])
        spans = []
        for index, token in enumerate(expected_ids):
            frames = np.flatnonzero(path == 2 * index + 1)
            if not len(frames):
                return Alignment('unalignable', [], None, 'missing_reference_token')
            spans.append(dict(token_id=token, start_frame=int(frames[0]), end_frame=int(frames[-1]+1),
                              mean_log_probability=float(lp[frames, token].mean())))
        return Alignment('tentative', spans, score, 'requires_boundary_validation')
