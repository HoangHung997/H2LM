"""Memory-bounded SGD step, with an audit for every unique parameter tensor.

Streaming is optimizer-in-backward: no momentum, accumulation, distributed training,
activation recomputation or global clipping. On an error some weights may already have
changed: DISCARD that in-memory state and resume from the last completed checkpoint.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class AuditedSGD:
    def __init__(self, model: nn.Module, lr: float, streaming: bool = False) -> None:
        if type(lr) not in (float, int) or not math.isfinite(lr) or not 0 < lr <= 0.01:
            raise ValueError("Invalid SGD learning rate")
        self.parameters = dict(model.named_parameters())
        if len(self.parameters) != len(list(model.named_parameters(remove_duplicate=False))):
            raise ValueError("Shared parameter aliases are not supported by the streaming audit")
        if any(not p.requires_grad for p in self.parameters.values()):
            raise ValueError("Scale audit requires all parameters to remain trainable")
        self.lr, self.streaming = lr, streaming
        self.audit: dict[str, dict] = {}
        self.handles = []
        if streaming:
            for name, p in self.parameters.items():
                self.handles.append(p.register_post_accumulate_grad_hook(
                    lambda parameter, name=name: self._update(name, parameter)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def _update(self, name: str, p: nn.Parameter) -> None:
        if name in self.audit:
            raise ValueError("Gradient accumulation/reentrant backward is unsupported")
        g = p.grad
        if g is None or not torch.isfinite(g).all():
            raise ValueError(f"Missing/nonfinite gradient: {name}; discard partial step")
        before = p.detach().clone()
        max_gradient = float(g.abs().max())
        p.add_(g, alpha=-self.lr)
        changed = int(torch.count_nonzero(p != before))
        self.audit[name] = {"numel": p.numel(), "grad_max": max_gradient, "updated_elements": changed}
        p.grad = None

    def backward(self, loss: torch.Tensor) -> dict:
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("Expected a finite scalar loss")
        if any(p.grad is not None for p in self.parameters.values()):
            raise ValueError("No gradient accumulation allowed in scale smoke")
        self.audit = {}
        loss.backward()
        if not self.streaming:
            # Check all gradients before the first standard SGD update.
            for name, p in self.parameters.items():
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise ValueError(f"Missing/nonfinite gradient: {name}")
            for name, p in self.parameters.items():
                self._update(name, p)
        if set(self.audit) != set(self.parameters):
            missing = sorted(set(self.parameters) - set(self.audit))
            raise ValueError(f"Parameters outside backward graph: {missing}")
        return {"parameter_tensors_in_backward": len(self.audit),
                "parameters_in_backward": sum(row["numel"] for row in self.audit.values()),
                "tensors_with_nonzero_gradient": sum(row["grad_max"] > 0 for row in self.audit.values()),
                "tensors_with_update": sum(row["updated_elements"] > 0 for row in self.audit.values()),
                "updated_elements": sum(row["updated_elements"] for row in self.audit.values()),
                "by_tensor": self.audit.copy()}
