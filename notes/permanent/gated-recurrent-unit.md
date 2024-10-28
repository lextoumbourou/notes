---
title: "Gated Recurrent Unit"
date: "2022-10-07 00:00"
modified: 2024-10-28 00:00
status: draft
---

**Gated Recurrent Units** (GRUs) are a streamlined variant of [Long Short-Term Memory (1997)](../reference/papers/long-short-term-memory-1997.md) (LSTM) networks that use two gates - a reset gate and an update gate - to solve the vanishing gradient problem in traditional RNNs while maintaining efficient training. The reset gate determines how to combine new input with previous memory, while the update gate controls what portion of the previous memory to keep, allowing GRUs to adaptively capture dependencies of different time scales while being computationally lighter than LSTMs due to having fewer parameters and no separate memory cell. GRUs have proven particularly effective in sequence modeling tasks like machine translation and speech processing, where they can often match or exceed LSTM performance despite their simpler architecture.

I'll break this down into both the mathematical formulation and a Python implementation of a GRU.

The mathematical equations for a GRU at time step $t$ are:

**Update Gate (z)**

$$z_t = σ(W_z[h_{t-1}, x_t] + b_z)$$

**Reset Gate (r)**

$$
r_t = \sigma(W_r[h_{t-1}, x_t] + b_r)
$$

**Candidate Hidden State (h̃)**

$$
h̃_t = tanh(W_h[r_t ⊙ h_{t-1}, x_t] + b_h)
$$

**Final Hidden State (h)**

$$
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
$$

Where:
- $\sigma$ is the sigmoid function
- ⊙ represents element-wise multiplication
- $[a,b]$ represents concatenation
- $W$ and $b$ are learnable parameters

Here's a Python implementation using PyTorch:

```python
import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Update gate parameters
        self.w_z = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Reset gate parameters
        self.w_r = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Candidate hidden state parameters
        self.w_h = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, x, h_prev):
        # Concatenate input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)
        
        # Update gate
        z = torch.sigmoid(self.w_z(combined))
        
        # Reset gate
        r = torch.sigmoid(self.w_r(combined))
        
        # Candidate hidden state
        combined_reset = torch.cat((x, r * h_prev), dim=1)
        h_tilde = torch.tanh(self.w_h(combined_reset))
        
        # Final hidden state
        h = (1 - z) * h_prev + z * h_tilde
        
        return h

# Example usage
batch_size = 1
input_size = 10
hidden_size = 20

gru = GRU(input_size, hidden_size)
x = torch.randn(batch_size, input_size)
h = torch.zeros(batch_size, hidden_size)

output = gru(x, h)
print(f"Output shape: {output.shape}")  # [batch_size, hidden_size]
```

The key advantages of this architecture are:

1. Simpler than LSTM (fewer parameters)
2. No separate memory cell (uses hidden state only)
3. Effective at capturing long-term dependencies
4. More computationally efficient than LSTM