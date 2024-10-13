---
title: Bubble Sort
date: 2023-12-17 00:00
modified: 2023-12-17 00:00
status: draft
---

**Bubble Sort** is another simple sorting algorithm, that involves repeatedly swapping elements that are not in order. The name comes from the way smaller elements "bubble" to the top of the list.

Bubble sort may need multiple passes to sort the entire vector. The maximum number of passes happen when the list is in complete reverse order. In this case, number of required passes is $n - 1$ where n is the number of elements.

```pseudo
\begin{algorithm}

\caption{Bubblesort}

\begin{algorithmic}
    \Function{BubbleSort}{vector}
        \State $n \gets \text{LENGTH}[\text{vector}]$

        \For{$1 \leq i \leq n - 1$}
            \State $count \gets 0$
            
            \For{$1 \leq j \leq n - 1$}
                \If{$\text{vector}[j+1] < \text{vector}[j]$}
                    \State $\text{Swap}(\text{vector}, j, j+1)$
                    \State $count \gets count + 1$
                \EndIf
            \EndFor
            \If{$count = 0$}
                \State \textbf{break}
            \EndIf
        \EndFor
        \State \Return $\text{vector}$
\EndFunction
\end{algorithmic}

\end{algorithm}
```
