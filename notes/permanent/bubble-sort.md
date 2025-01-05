---
title: Bubble Sort
date: 2023-12-17 00:00
modified: 2023-12-17 00:00
status: draft
tags:
- SortingAlgorithm
---

**Bubble Sort** is another simple sorting algorithm that involves repeatedly swapping elements that are not in order. The name comes from the way smaller elements "*bubble*" to the top of the list.

It can be implemented with a while loop, where adjacent pairs of numbers are swapped until there are no more swaps to do, as using nested for loops, where every position is compared with every other position until a position has no swaps.

It has one advantage in that it's easy to implement and understand. It also doesn't need any additional memory to operate. However, it has a quadratic run time in the average case and won't be useful for large arrays.

## Time Complexity

### $\text{Big-}O$

A sorted list's upper bound time complexity is linear, as it will need at least one full pass to determine no swaps.

| Best       | Average  | Worst    |
| ---------- | -------- | -------- |
| $O(n)$<br> | $O(n^2)$ | $O(n^2)$ |

### $\text{Big-}\Omega$

The lower bound best case is still linear, as we must look at every element once.

| Best       | Average  | Worst    |
| ---------- | -------- | -------- |
| $O(n)$<br> | $O(n^2)$ | $O(n^2)$ |

### $\text{Big-}\Theta$

| Best         | Average  | Worst    |
| ------------ | -------- | -------- |
| $O(n^2)$<br> | $O(n^2)$ | $O(n^2)$ |

## Space Complexity

 Only requires a constant amount of additional memory space for the swap operation: $O(1)$

## Pseudo-code

```pseudo
\begin{algorithm}
\caption{Bubble Sort (While Loop)}
\begin{algorithmic}
    \Function{BubbleSort}{A}
        \State swapped $\gets$ true
        \While{swapped}
            \State swapped $\gets$ false
            \For{$i \gets 0$ to $\text{length}[A] - 2$}
                \If{$A[i] > A[i + 1]$}
                    \State swap $A[i]$ and $A[i + 1]$
                    \State swapped $\gets$ true
                \EndIf
            \EndFor
        \EndWhile
        \State \Return $A$
    \EndFunction
\end{algorithmic}
\end{algorithm}
```

```pseudo
\begin{algorithm}
\caption{Bubble Sort (For Loop)}
\begin{algorithmic}
    \Function{BubbleSort}{A}
        \State $n \gets \text{length}[A]$
        \For{$i \gets 0$ to $n - 1$}
            \For{$j \gets 0$ to $n - i - 2$}
                \If{$A[j] > A[j + 1]$}
                    \State swap $A[j]$ and $A[j + 1]$
                \EndIf
            \EndFor
        \EndFor
        \State \Return $A$
    \EndFunction
\end{algorithmic}
\end{algorithm}
```

See [Sorting Algorithm](../../../permanent/sorting-algorithm.md).