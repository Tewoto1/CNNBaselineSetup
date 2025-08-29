import torch
from torch import nn
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__(); self.smoothing = smoothing
    def forward(self, logits, targets):
        n = logits.size(-1)
        logp = logits.log_softmax(dim=-1)
        with torch.no_grad():
            y = torch.zeros_like(logp).fill_(self.smoothing/(n-1))
            y.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(y * logp).sum(dim=1).mean()
