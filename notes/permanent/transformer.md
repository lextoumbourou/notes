---
title: Transformer
date: 2022-07-31 00:00
status: draft
---

A Transformer is a sequence-to-sequence [Model Architecture](Model Architecture).

It takes an input sequence, performs an [Embeddings](Embeddings) operation to convert to a [Vector](vector.md) of length 512

We pass our sequence batch through [Self-Attention](Self-Attention) layer. We'll talk about that next.

Then the output of that into a "feed forward" layer, with which we concat with the original input. That's a process known as [Residual Layers](Residual Layers).

```
# This returns a batch of vectors
attended = self.attention(x)

# We take that output and concat with the original output. That's called residual connection.
x = self.norm1(attended + x)

# We pass to a standard linear layer.
fedforward = self.ff(x)

# Then the residuals of that to another layer norm with a residual connection.
x = self.norm2(fedforward + x)

return x
```

That is an entire Transformer block. The paper then puts 6 in a row.

```python
class TransformerBlock(nn.Module):
    def forward(self):
        # This returns a batch of vectors
        attended = self.attention(x)

        # We take that output and concat with the original output. That's called residual connection.
        x = self.norm1(attended + x)

        # We pass to a standard linear layer.
        fedforward = self.ff(x)

        # Then the residuals of that to another layer norm with a residual connection.
        x = self.norm2(fedforward + x)

        return x
```

## Self-Attention

Self-Attention works like this:

```
class Transformer(nn.Module):
    def forward(self):
        tblocks = []
        for i in range(6):
            tblocks.append(TransformerBlock())
```
