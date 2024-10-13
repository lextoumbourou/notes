---
title: Pad Cheatsheet
date: 2024-01-23 00:00
modified: 2024-01-23 00:00
status: draft
---

The [`pad`](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html) function from the PyTorch functionality library is used to pad a Tensor. In other words, add values (usually zeros) around a Tensor.

## Padding Size

The `pad` argument tells the function which sides of the Tensor to pad.

If you only pass two values, they will serve as `left` and `right` padding.

For example, if I have a 1-dimension Tensor of length 5:

```python
> t1d = torch.ones(5)
> t1d.shape, t1d
(torch.Size([5]), tensor([1., 1., 1., 1., 1.]))
```

 I can add a value of 0 to the left as follows:

```python
> p = F.pad(input=t1d, pad=(1, 0), value=0)
> p.shape, p
(torch.Size([6]), tensor([0., 1., 1., 1., 1., 1.]))
```

Or to the right:

```python
> p = F.pad(input=t1d, pad=(0, 1), value=0)
> p.shape, p
(torch.Size([6]), tensor([1., 1., 1., 1., 1., 0.]))
```

Or to both sides:

```python
> p = F.pad(input=t1d, pad=(1, 1), value=0)
> p.shape, p
(torch.Size([7]), tensor([0., 1., 1., 1., 1., 1., 0.]))
```

With a multi-dimensional Tensor, only the final dimension will be affected if you only pass two values to the pad function.

For example, here's a 3d Tensor:

```python
> t3d = torch.empty(1, 5, 10)
> t3d.shape, t3d
(torch.Size([1, 5, 10]),
 tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]]))
```

I can add a column of 0s to the left-hand side:

```python
> p = F.pad(t3d, (1, 0), value=0)
> p.shape, p
(torch.Size([1, 5, 11]),
 tensor([[[0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]]))
```

If I pass four arguments, I can set the padding for the last dimension left and right, but also add a top and bottom, which affects the 2nd dimension (adds row on top or bottom):

```python
> p = F.pad(t3d, (0, 0, 1, 1), value=0)
> p.shape, p
(torch.Size([1, 7, 10]),
 tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]))
```

I can even pad the batch dimension by padding six arguments:
* left
* right
* top
* bottom
* front
* back

```python
> p = F.pad(t3d, (0, 0, 1, 1, 1, 1), value=0)
> p.shape, p
(torch.Size([3, 7, 10]),
 tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
 
         [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
 
         [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]))
```

## Padding Mode

The `mode` argument supports various types of padding.

### Constant

The most common is `constant`, which supports padding a Tensor with a constant value. The constant value can be passed via the `value` argument.

```python
>> p = F.pad(torch.tensor([[1, 2, 3]]), (2, 2),  mode="constant", value=0)
>> p
tensor([[0, 0, 1, 2, 3, 0, 0]])
```

### Reflect

The `reflect` mode pads use a reflection of the input boundary.

```python
>> p = F.pad(torch.tensor([[1, 2, 3]]), (2, 2),  mode="reflect")
>> p
tensor([[3, 2, 1, 2, 3, 2, 1]])
```

### Replicate

The `replicate` mode replicates the input boundary.

```python
>> p = F.pad(torch.tensor([[1, 2, 3]]), (2, 2),  mode="replicate")
>> p
tensor([[1, 1, 1, 2, 3, 3, 3]])
```

### Circular

The `circular` mode adds circular padding to the Tensor.

```python
>> p = F.pad(torch.tensor([[1, 2, 3]]), (2, 2),  mode="circular")
>> p
tensor([[2, 3, 1, 2, 3, 1, 2]])
```
