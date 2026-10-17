---
title: Backpropagation
date: 2024-10-07 00:00
modified: 2026-10-12 14:54
status: hidden
summary: One of the most important algorithms in Deep Learning - the thing that computes the gradient with respect to the loss.
tags:
- MachineLearning
- Backpropagation
category: note
---

Let's talk about **Backpropagation** or "Backprop"

Backprop is the algorithm used to train basically every [Neural Network](neural-network.md). The tl;dr is that it computes the [Gradient](gradient.md) of the loss with respect to every weight in the network.

You:
1. run the input forward
2. measure the error at the output
3. send that error backwards through each layer, using the [Chain Rule](chain-rule.md) to work out how much each weight contributed to it [@rumelhartLearningRepresentationsBackpropagating1986]

It's the [Gradient Descent](gradient-descent.md) then nudges each weight in the direction that lowers the loss.

Since I'm a software developer - and presumably you are too if you're reading this - the best way to make sure I understand a topic is to make sure I can build it from scratch. So in this article, we'll build backpropagation step by step in PyTorch, following one tiny network and one training example the whole way through, then check our answer against PyTorch's own autograd.

## The problem it solves

A network's loss is a single number, but the network might have millions or billions of weights. To learn, every weight needs to know which way to move, and how far, to bring that one number down. Working out each weight's share of the blame is called **credit assignment** [@liNoPropTrainingNeural2025].

The obvious approach is to change each weight a little and see what happens to the loss.

That actually would work, but you'd have to make like one or two forward passes per weight. For a model with a billion weights, that's a billion forward passes for a single training step.

Luckily, thanks to the invention of calculus in the 17th century, we have a way to calculate the gradient for every weight at once. 

The only thing we sacrifice is a bit of memory. Let's take a look.

## Backpropagation Step by Step

We'll use the same setup as the 1986 paper that made backpropagation famous: layers of [Sigmoid](sigmoid-function.md) units and a squared-error loss [@rumelhartLearningRepresentationsBackpropagating1986]. Modern networks use other activations and losses, but the algorithm is identical.

Our network has two inputs, two hidden units and one output, so it has two weight matrices: `W1` (2×2) and `W2` (1×2). Each layer also has a bias, which we'll start at zero. We'll train it on a single example: the input `[1, 0]` should give the output `1`.

### 0. Set up the network

To keep the numbers easy to follow, we pick the starting weights by hand. In practice, they start as small random numbers: the 1986 paper notes that "to break symmetry we start with small random weights", because if every weight began equal, every hidden unit would compute the same thing and receive the same update forever [@rumelhartLearningRepresentationsBackpropagating1986].

```python {run=false}
x = torch.tensor([[1.0, 0.0]])
target = torch.tensor([[1.0]])

W1 = torch.tensor([[0.5, -0.4],
                   [0.3,  0.8]])
b1 = torch.zeros(2)
W2 = torch.tensor([[0.7, -0.6]])
b2 = torch.zeros(1)
```

The weights are stored like PyTorch's `nn.Linear`: one row per output unit and one column per input. So `W1[0]` holds the weights into hidden unit 1, and `W2[0][1]` is the weight from hidden unit 2 to the output.

### 1. Run the forward pass, and keep every layer's input

Each layer multiplies its input by its weights, adds the bias and applies the sigmoid. This is an ordinary forward pass, with one difference that matters: we keep every intermediate value, because the backward pass needs them.

```python {run=false}
z1 = x @ W1.T + b1      # [[0.5000, 0.3000]]
h = torch.sigmoid(z1)   # [[0.6225, 0.5744]]
z2 = h @ W2.T + b2      # [[0.0911]]
y = torch.sigmoid(z2)   # [[0.5227]]
```

`z1` and `z2` are each layer's total input before the sigmoid, and `h` is the hidden layer's output, which is also the input to the second layer. We wanted `1` and got `0.5227`.

### 2. Compute the loss

The 1986 paper defines the error as half the squared difference between the actual and desired outputs, summed over output units [@rumelhartLearningRepresentationsBackpropagating1986]. The half is there to cancel the 2 when we differentiate.

$$L = \tfrac{1}{2}(y - t)^2$$

```python {run=false}
loss = 0.5 * ((y - target) ** 2).sum()   # 0.1139
```

### 3. Compute the error at the output

Now we go backwards. First: how much does the loss change if the output changes? Differentiating the loss gives the difference between what we got and what we wanted. This is equation (4) in the 1986 paper [@rumelhartLearningRepresentationsBackpropagating1986].

$$\frac{\partial L}{\partial y} = y - t$$

```python {run=false}
dy = y - target   # [[-0.4773]]
```

The sign tells us the direction: it's negative, so increasing `y` would lower the loss.

### 4. Go back through the sigmoid

`y` came out of a sigmoid, so to reach the layer's total input `z2` we apply the chain rule: multiply by the sigmoid's slope at that point. The sigmoid has a neat derivative, $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, so the slope is just `y * (1 - y)`, using the output we kept from the forward pass. This is equation (5) in the paper.

$$\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial y} \cdot y(1 - y)$$

```python {run=false}
dz2 = dy * y * (1 - y)   # [[-0.1191]]
```

`y` is near 0.5, where the sigmoid is steepest, so its slope is about 0.25 and the error shrinks to a quarter. A sigmoid near 0 or 1 is almost flat and lets very little error through, which is one source of the vanishing gradients that made deep networks hard to train.

From here on, I'll call `dz2` the **error** at the output layer.

### 5. Compute the weight gradient: error times input

Each weight's gradient is the error at the unit it feeds into, multiplied by the input that flowed along it in the forward pass. This is equation (6) in the paper: $\partial L / \partial w_{ji} = \partial L / \partial z_j \cdot y_i$ [@rumelhartLearningRepresentationsBackpropagating1986].

$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2}^{T} h$$

```python {run=false}
dW2 = dz2.T @ h      # [[-0.0741, -0.0684]]
db2 = dz2.sum(0)     # [-0.1191]
```

The intuition: a weight only matters as much as the signal that passed through it. Hidden unit 1 was slightly more active (0.6225 against 0.5744), so its weight gets slightly more of the blame. A bias is a weight whose input is always 1, so its gradient is just the error.

With a batch of examples, `dz2.T @ h` sums the products over the batch, and `sum(0)` does the same for the bias.

### 6. Send the error down through the same weights

To keep going, we need the error at the hidden layer. Each hidden unit affected the loss through every output unit it connects to, so its error is the sum of those output errors, each weighted by the connection between them. That's equation (7) in the paper, and in matrix form it means multiplying by the same weight matrix, transposed.

$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial z_2} W_2$$

```python {run=false}
dh = dz2 @ W2   # [[-0.0833, 0.0714]]
```

In the forward pass we computed `h @ W2.T`. Going backwards we compute `dz2 @ W2`: the same matrix with the transpose undone, so the error flows back along exactly the connections the signal came forward on.

The two hidden units get errors of opposite sign. Hidden unit 1 has a positive weight (0.7) to the output, so making it more active would raise `y` and lower the loss. Hidden unit 2 has a negative weight (-0.6), so it should become *less* active.

### 7. Repeat for the layer below

Now we're in the same position as step 4, one layer down: an error at a sigmoid's output. So we repeat steps 4 and 5: multiply by the sigmoid's slope, then multiply the error by the stored input.

```python {run=false}
dz1 = dh * h * (1 - h)   # [[-0.0196, 0.0175]]
dW1 = dz1.T @ x          # [[-0.0196, 0.0000],
                         #  [ 0.0175, 0.0000]]
db1 = dz1.sum(0)         # [-0.0196, 0.0175]
```

Look at the second column of `dW1`: both gradients are exactly zero. The second input was `0`, so the weights leaving it played no part in this mistake, and backpropagation assigns them no blame. That's credit assignment at work.

In a deeper network, we'd keep going: step 6 sends the error down another layer, then step 7 again, until we reach the first layer. Every layer reuses the error computed for the layer above it, and that reuse is what makes backpropagation efficient.

### 8. Check against autograd

PyTorch's autograd runs the same algorithm automatically. If we let it compute the gradients for the same network, they match ours to the last decimal place:

```python {run=false}
params = [t.clone().requires_grad_() for t in (W1, b1, W2, b2)]
y_auto = torch.sigmoid(torch.sigmoid(x @ params[0].T + params[1]) @ params[2].T + params[3])
(0.5 * ((y_auto - target) ** 2).sum()).backward()

for mine, auto in zip((dW1, db1, dW2, db2), params):
    print(torch.allclose(mine, auto.grad))   # True, four times
```

When you call `loss.backward()` in PyTorch, this is what happens: autograd records every operation in the forward pass, then walks that record in reverse, applying each operation's local derivative.

### 9. Update the weights

Backpropagation's job ends with the gradients. Using them is [Gradient Descent](gradient-descent.md): move every weight a small step against its gradient. The 1986 paper used exactly this, $\Delta w = -\varepsilon \, \partial E / \partial w$, and also suggested adding momentum, where each step keeps a fraction of the previous one [@rumelhartLearningRepresentationsBackpropagating1986].

```python {run=false}
lr = 1.0
W1 = W1 - lr * dW1
b1 = b1 - lr * db1
W2 = W2 - lr * dW2
b2 = b2 - lr * db2
```

Running the forward pass again with the new weights gives an output of `0.5762`, up from `0.5227`, and the loss falls from `0.1139` to `0.0898`. One step, and every weight that played a part moved in a direction that helped. The weights from the second input stayed put, since their gradients were zero.

## The full algorithm

Here's the whole thing as a small class, trained on [Exclusive-Or](logical-xor.md). XOR is the classic problem a network without hidden units can't solve: no single straight line separates the inputs that should give 1 from those that should give 0. With a hidden layer trained by backpropagation, it can.

```python
import torch


class TwoLayerNet:
    """A 2-layer sigmoid network trained with hand-written backpropagation."""

    def __init__(self, n_in, n_hidden, n_out):
        self.W1 = torch.randn(n_hidden, n_in)
        self.b1 = torch.zeros(n_hidden)
        self.W2 = torch.randn(n_out, n_hidden)
        self.b2 = torch.zeros(n_out)

    def forward(self, x):
        # Keep every layer's input: the backward pass needs them.
        self.x = x
        self.h = torch.sigmoid(x @ self.W1.T + self.b1)
        self.y = torch.sigmoid(self.h @ self.W2.T + self.b2)
        return self.y

    def backward(self, target):
        # Error at the output, then back through the output sigmoid.
        dy = self.y - target
        dz2 = dy * self.y * (1 - self.y)
        # Weight gradient: error times the layer's stored input.
        self.dW2 = dz2.T @ self.h
        self.db2 = dz2.sum(0)
        # Send the error down through the same weights.
        dh = dz2 @ self.W2
        # Repeat for the layer below.
        dz1 = dh * self.h * (1 - self.h)
        self.dW1 = dz1.T @ self.x
        self.db1 = dz1.sum(0)

    def step(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2


X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
T = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  # XOR

torch.manual_seed(2)
net = TwoLayerNet(n_in=2, n_hidden=2, n_out=1)

for epoch in range(5000):
    y = net.forward(X)
    loss = 0.5 * ((y - T) ** 2).sum()
    net.backward(T)
    net.step(lr=2.0)
    if epoch % 1000 == 0:
        print(f"epoch {epoch}: loss {loss.item():.4f}")

print(net.forward(X).round(decimals=2).flatten())  # tensor([0.0100, 0.9900, 0.9900, 0.0100])
```

> [!note]
> Not every starting point works. When I tried seeds 0 to 5, three of them got stuck: the network learned two of the four cases and output 0.5 for the other two. That's a local minimum, and with only two hidden units this small network has nowhere else to go. More hidden units, or a different seed, gets it out.

## Why it's so cheap

The backward pass costs about the same as the forward pass, give or take a small constant. For transformers, the usual rule of thumb is that the backward pass takes about twice the compute of the forward pass, which is where the estimate of $6N$ operations per training token comes from, for a model with $N$ parameters: $2N$ forward and $4N$ backward [@kaplanScalingLawsNeural2020].

This isn't luck. In 1983, Walter Baur and Volker Strassen proved that all the partial derivatives of a function can be computed for at most a small constant multiple of the cost of the function itself [@baurComplexityPartialDerivatives1983]. Andreas Griewank calls this the "cheap gradient principle": the constant doesn't grow with the number of inputs [@griewankWhoInventedReverse2012]. A billion weights cost one backward pass, not a billion forward passes.

## What it costs

Backpropagation does have a price.

- **Memory.** Every intermediate value from the forward pass has to be kept until the backward pass uses it (step 1). The MeZO paper measured fine-tuning with backpropagation at up to 12 times the memory of running the model [@malladiFineTuningLanguageModels2023].
- **Locking.** No layer can update until the whole forward pass, and the backward pass above it, have finished. Jaderberg et al. call this forward, update and backwards locking [@jaderbergDecoupledNeuralInterfaces2016].
- **Brains.** Step 6 sends the error back through the exact same weights used going forward. Lillicrap et al. point out that this "involves a precise, symmetric backward connectivity pattern, which is thought to be impossible in the brain" [@lillicrapRandomSynapticFeedback2016]. The 1986 paper said as much itself: "The learning procedure, in its current form, is not a plausible model of learning in brains" [@rumelhartLearningRepresentationsBackpropagating1986].

## History

The core trick is older than neural networks.

- **1960.** Henry J. Kelley's paper on optimal flight paths computes its gradients by solving an adjoint system backwards from the end of the flight. His worked example is a solar sail travelling from Earth's orbit to Mars [@kelleyGradientTheoryOptimal1960].
- **1970.** Seppo Linnainmaa's master's thesis at the University of Helsinki describes what is now called reverse-mode automatic differentiation. He used it to estimate rounding errors, not to train anything [@linnainmaaAlgoritminKumulatiivinenPyoristysvirhe1970] [@griewankWhoInventedReverse2012].
- **1982.** Paul Werbos proposes applying the method to "neuron modelling", arguing that optimising a network of model neurons "is not intrinsically different from conventional optimization" [@werbosApplicationsAdvancesNonlinear1982].
- **1986.** David Rumelhart, Geoffrey Hinton and Ronald Williams publish *Learning Representations by Back-Propagating Errors* in Nature. Their point was what the method could do: hidden units trained this way "come to represent important features of the task domain", features nobody designed [@rumelhartLearningRepresentationsBackpropagating1986]. This is the paper that made backpropagation famous.
- **1989.** Yann LeCun and colleagues train a convolutional network with backpropagation to read handwritten zip codes [@lecunBackpropagationAppliedHandwritten1989].

Who invented it is still argued over. The Nobel Committee's scientific background for the 2024 physics prize says Rumelhart, Hinton and Williams "reinvented a scheme for this, which had previously been applied to related problems by others" [@nobelcommitteeforphysicsScientificBackgroundNobel2024], and Griewank traces the reverse mode through "many incarnations" since the late 1960s [@griewankWhoInventedReverse2012].

## Training without backpropagation

People have looked for alternatives for decades, mostly for the reasons above: memory, parallelism and biology. The main families are:

- **Random feedback.** Feedback alignment sends the error back through fixed random weights instead of the forward weights, and the forward weights learn to line up with them [@lillicrapRandomSynapticFeedback2016].
- **Local learning.** Each layer or block gets its own objective, so no error crosses between them. Hinton's Forward-Forward algorithm uses two forward passes, one on real data and one on negative data [@hintonForwardForwardAlgorithmPreliminary2022]. [NoProp: Training Neural Networks Without Back-Propagation or Forward-Propagation](../reference/papers/noprop-training-neural-networks-without-back-propgation-or-forward-propagation.md) trains each block to denoise a noisy label, borrowing from diffusion, though it still uses backpropagation inside each block [@liNoPropTrainingNeural2025].
- **Perturbation.** Go back to wiggling, but cleverly. MeZO fine-tunes language models of up to 66 billion parameters with forward passes alone [@malladiFineTuningLanguageModels2023], evolution strategies such as EGGROLL perturb the weights of a whole population [@sarkarEvolutionStrategiesHyperscale2025], and [Dust: pretraining transformers without backpropagation](dust-pretraining-transformers-without-backpropagation.md) perturbs activations at every token, and gets close to backpropagation when pretraining transformers of up to 243M parameters [@dahalDustPretrainingTransformers2026].

Most of these match backpropagation on small benchmarks like MNIST. At scale, they have struggled: in a 2018 study on ImageNet, feedback alignment reached a top-1 error of 93.08% and difference target propagation 98.34%, against 71.43% for backpropagation [@bartunovAssessingScalabilityBiologicallyMotivated2018]. Even Hinton wrote that Forward-Forward "is unlikely to replace backpropagation for applications where power is not an issue" [@hintonForwardForwardAlgorithmPreliminary2022].
