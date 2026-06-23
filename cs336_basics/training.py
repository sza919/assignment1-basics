from __future__ import annotations
from einops import einsum, rearrange
import torch
import numpy as np
from jaxtyping import Float, Int
from collections.abc import Callable
from typing import Optional
import math
import os
import typing

def cross_entropy(logits: Float[Tensor, " batch_size vocab_size"],
                  targets: Int[Tensor, " batch_size"]):
    logits = logits - torch.max(logits, dim = -1, keepdim = True).values
    logits_exp = torch.exp(logits)
    probability = -logits + torch.log(torch.sum(logits_exp, dim = -1, keepdim = True))
    target_nll = torch.gather(
        probability,
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)
    return target_nll.mean()

def learning_rate_from_cosine_annealing(t, lr_min, lr_max, T_warmup, T_final):
    if t < T_warmup:
        return t * lr_max / T_warmup
    elif t > T_final:
        return lr_min
    else:
        return lr_min + 1/2 * (1 + math.cos(math.pi * (t - T_warmup)/(T_final - T_warmup))) * (lr_max - lr_min)
    
import torch

def gradient_clipping(params, max_norm=1e5, eps=1e-6):
    params = [p for p in params if p.grad is not None]

    if len(params) == 0:
        return torch.tensor(0.0)

    with torch.no_grad():
        total_norm = torch.sqrt(
            sum(torch.sum(p.grad ** 2) for p in params)
        )

        clip_coef = max_norm / (total_norm + eps)

        if clip_coef < 1:
            for p in params:
                p.grad.mul_(clip_coef)

    return total_norm


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9,0.999), weight_decay = 0.01, eps = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"alpha": lr, 'beta1': betas[0], 'beta2': betas[1], 'lambda': weight_decay, 'eps': eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        with torch.no_grad():
            for group in self.param_groups:
                alpha = group['alpha']
                beta1 = group['beta1']
                beta2 = group['beta2']
                lam = group['lambda']
                eps = group['eps']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    t = state.get('t', 1)
                    state["m"] = state.get('m', torch.zeros_like(p))
                    state["v"] = state.get('v', torch.zeros_like(p))
                    grad = p.grad
                    p.mul_(1 - alpha * lam)
                    state['m'] = beta1 * state["m"] + (1 - beta1) * grad
                    state['v'] = beta2 * state["v"] + (1 - beta2) * grad ** 2
                    alpha_t = alpha * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)
                    p.addcdiv_(state["m"], state["v"].sqrt() + eps, value=-alpha_t)
                    state['t'] = t + 1
        return loss

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    x: 1D numpy array of token IDs
    returns:
        inputs:  (batch_size, context_length)
        targets: (batch_size, context_length)
    """
    # Need room for context_length inputs plus 1 next-token target
    max_start = len(x) - context_length - 1

    # Random starting indices
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    # Build input and target batches
    inputs = np.stack([x[i : i + context_length] for i in starts])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in starts])

    # Convert to torch tensors and move to device
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets


def save_checkpoint(model: torch.nn.Modulo, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)
    return

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]



if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights])
    for t in range(10):
        opt.zero_grad()              # reset gradients
        loss = (weights ** 2).mean() # scalar loss
        print(loss.cpu().item())
        loss.backward()              # compute weights.grad
        opt.step()                   # update weights