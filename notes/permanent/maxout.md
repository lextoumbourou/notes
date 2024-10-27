---
title: Maxout
date: 2024-10-28 00:00
modified: 2024-10-28 00:00
status: draft
---

A **Maxout** layer works by splitting its input neurons into groups and then taking the highest value from each group, kind of like having multiple paths and always choosing the strongest signal. Think of it like having several different ways to process the same information and then picking the best result from each group, which helps the network learn more effectively than using simple on/off activation functions like ReLU.

```python
class MaxoutLayer(nn.Module):
    def __init__(self, in_features, out_features, num_pieces=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces
        self.linear = nn.Linear(in_features, out_features * num_pieces)
    
    def forward(self, x):
        shape = [x.shape[0], self.out_features, self.num_pieces]
        x = self.linear(x) # B, F
        x = x.view(*shape)  # B, F, P
        x, _ = torch.max(x, -1) # B, F
        return x # B, F
```
