from einops import einsum
import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... input, output input -> ... output')
    
class Embedding(torch.nn.Module):
    