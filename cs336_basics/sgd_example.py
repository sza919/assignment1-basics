from collections.abc import Callable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] 
                # iteration number for this parameter
                t = state.get("t", 0)
                grad = p.grad
                # update parameter in-place
                p.data -= lr / math.sqrt(t + 1) * grad.data
                # save updated state
                state["t"] = t + 1
        return loss


if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=50)

    for t in range(10):
        opt.zero_grad()              # reset gradients
        loss = (weights ** 2).mean() # scalar loss
        print(loss.cpu().item())
        loss.backward()              # compute weights.grad
        opt.step()                   # update weights