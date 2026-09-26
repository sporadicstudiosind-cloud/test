"""Causal learned compression, owned by the same model as the control core.

State is per request, never stored on the module. Compression is lossy; the
bounded latent bank is not an exact million-token attention window.
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MemoryState:
    slots: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    pending: torch.Tensor | None = None
    pending_valid: torch.Tensor | None = None
    seen: int = 0

    def detach(self):
        return MemoryState(**{k: v.detach() if isinstance(v, torch.Tensor) else v
                              for k, v in vars(self).items()})


class ContextMemory(nn.Module):
    """Recent raw blocks -> learned summaries -> bounded consolidated bank.

    A summary becomes visible only AFTER its entire source block. This is
    causal both in full-sequence training and token-at-a-time cached decoding.
    Learned importance is a soft weighting, never a claim a token is useless.
    """
    def __init__(self, width, slots=64, stride=32, rank=64):
        super().__init__()
        self.capacity, self.stride = slots, stride
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.importance = nn.Linear(rank, 1)
        self.query = nn.Linear(width, rank, bias=False)
        self.key = nn.Linear(width, rank, bias=False)
        self.gate = nn.Linear(width, 1)
        nn.init.constant_(self.gate.bias, -2.)

    def forward(self, inputs, valid, state=None):
        state = state or MemoryState()
        # Return a new state; a failed training forward cannot mutate a session.
        slots, slot_valid = state.slots, state.valid
        pending, pv = state.pending, state.pending_valid
        results, auxiliary = [], []
        offset = 0
        while offset < inputs.shape[1]:
            used = 0 if pending is None else pending.shape[1]
            end = min(inputs.shape[1], offset + self.stride - used)
            x, mask = inputs[:, offset:end], valid[:, offset:end]
            y = x
            if slots is not None:
                # Autocast may return fp16 even for float matmul inputs.
                # Cast the result before the negative mask and softmax.
                score = torch.matmul(self.query(x).float(), self.key(slots).float().transpose(-1, -2)).float()
                score = score / self.query.out_features ** .5
                score = score.masked_fill(~slot_valid[:, None, :], -1e9)
                weights = score.softmax(-1) * slot_valid[:, None, :]
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
                retrieved = torch.matmul(weights.to(slots.dtype), slots)
                y = x + torch.sigmoid(self.gate(x)) * retrieved
            results.append(y)
            pending = x if pending is None else torch.cat((pending, x), 1)
            pv = mask if pv is None else torch.cat((pv, mask), 1)
            if pending.shape[1] == self.stride:
                encoded = torch.tanh(self.down(pending))
                scores = self.importance(encoded).float().squeeze(-1).masked_fill(~pv, -1e9)
                weights = scores.softmax(-1) * pv
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
                summary = self.up((encoded * weights[..., None].to(encoded.dtype)).sum(1, keepdim=True))
                sv = pv.any(1, keepdim=True)
                slots = summary if slots is None else torch.cat((slots, summary), 1)
                slot_valid = sv if slot_valid is None else torch.cat((slot_valid, sv), 1)
                if slots.shape[1] > self.capacity:
                    # Consolidate oldest summaries instead of simply forgetting
                    # them. The first slot gradually becomes a lossy long-term
                    # state; the remaining slots retain finer recent summaries.
                    old = torch.tanh(self.down(slots[:, :2]))
                    old_valid = slot_valid[:, :2]
                    score = self.importance(old).float().squeeze(-1).masked_fill(~old_valid, -1e9)
                    weight = score.softmax(-1) * old_valid
                    weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-9)
                    merged = self.up((old * weight[..., None].to(old.dtype)).sum(1, keepdim=True))
                    slots = torch.cat((merged, slots[:, 2:]), 1)
                    slot_valid = torch.cat((old_valid.any(1, keepdim=True), slot_valid[:, 2:]), 1)
                # Keep local information recoverable; downstream task gradients
                # also train compression through retrieval in subsequent blocks.
                reconstruction = self.up(encoded)
                error = (reconstruction.float() - pending.detach().float()).square().mean(-1)
                auxiliary.append((error * pv).sum() / pv.sum().clamp_min(1))
                pending = pv = None
            offset = end
        loss = torch.stack(auxiliary).mean() if auxiliary else inputs.sum() * 0
        return torch.cat(results, 1), MemoryState(slots, slot_valid, pending, pv,
                                                state.seen + inputs.shape[1]), loss
