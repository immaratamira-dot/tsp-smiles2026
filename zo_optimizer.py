"""
zo_optimizer.py — Zero-order optimizer (student-implemented).

Key design choices
------------------
* SPSA (Simultaneous Perturbation Stochastic Approximation) instead of the
  per-parameter central-difference skeleton.  The skeleton requires 2 forward
  passes *per parameter* — for fc.weight alone that is 2 x 51200 = 102400
  calls per step, which blows the entire budget instantly.  SPSA perturbs ALL
  active parameters at once and needs exactly 2 forward passes per step
  regardless of parameter count.

* Adam-style momentum to accelerate convergence within the limited budget.

* Layer curriculum: tune only fc for the first half of steps (head is randomly
  initialised so it needs the most work), then add layer4 block 2 for the
  second half to pick up some backbone signal.
"""

from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """SPSA-based gradient-free optimizer with Adam-style updates."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-3,
        eps: float = 5e-4,
        perturbation_mode: str = "gaussian",
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
        n_batches: int = 128,
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.perturbation_mode = perturbation_mode
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps
        self.n_batches = n_batches

        self.layer_names: list[str] = ["fc.weight", "fc.bias"]
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}
        self._step_count: int = 0

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(f"Layer names not found in model: {missing}")
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        if self.perturbation_mode == "gaussian":
            return torch.randn_like(param)
        else:
            u = torch.rand_like(param) * 2.0 - 1.0
            norm = u.norm()
            return u / norm if norm > 0 else u

    def _ensure_adam_state(self, name: str, param: torch.Tensor) -> None:
        if name not in self._m:
            self._m[name] = torch.zeros_like(param.data)
            self._v[name] = torch.zeros_like(param.data)

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """SPSA: exactly 2 forward passes for ALL parameters combined.

        Perturbs every active parameter simultaneously with independent
        random directions, then estimates the gradient from 2 loss values.
        """
        directions: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for name, param in params.items():
                u = self._sample_direction(param)
                directions[name] = u
                param.data.add_(self.eps * u)

            f_plus = loss_fn()

            for name, param in params.items():
                param.data.sub_(2.0 * self.eps * directions[name])

            f_minus = loss_fn()

            for name, param in params.items():
                param.data.add_(self.eps * directions[name])

        scale = (f_plus - f_minus) / (2.0 * self.eps)
        return {name: scale * directions[name] for name in params}

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Adam update using estimated pseudo-gradients."""
        t = self._step_count
        with torch.no_grad():
            for name, param in params.items():
                self._ensure_adam_state(name, param)
                g = grads[name]
                self._m[name].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                self._v[name].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                m_hat = self._m[name] / (1.0 - self.beta1 ** t)
                v_hat = self._v[name] / (1.0 - self.beta2 ** t)
                param.data.sub_(self.lr * m_hat / (v_hat.sqrt() + self.adam_eps))

    def _update_layer_schedule(self) -> None:
        """Curriculum: unlock layer4 block 2 at the halfway point."""
        phase2_start = self.n_batches // 2
        if self._step_count == phase2_start:
            extra = [
                "layer4.1.conv1.weight",
                "layer4.1.conv2.weight",
                "layer4.1.bn1.weight",
                "layer4.1.bn1.bias",
                "layer4.1.bn2.weight",
                "layer4.1.bn2.bias",
            ]
            valid = dict(self.model.named_parameters())
            additions = [n for n in extra if n in valid]
            self.layer_names = ["fc.weight", "fc.bias"] + additions

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one SPSA optimisation step. Returns loss before update."""
        self._step_count += 1
        self._update_layer_schedule()
        params = self._active_params()

        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)
        return float(loss_before)
