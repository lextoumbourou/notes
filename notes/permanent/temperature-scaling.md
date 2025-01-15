---
title: Temperature Scaling
date: 2025-01-14 00:00
modified: 2025-01-15 00:00
tags:
- LargeLanguageModels
- MachineLearning
summary: a parameter that controls how confident Softmax predictions are
---

Temperature scaling controls how "confident" a model is when making predictions by adjusting the sharpness of probability distributions produced by the [Softmax Function](softmax-activation-function.md).

Softmax is a function that converts a neural network's raw outputs (logits) into probabilities that sum to 1. For example, in a dog breed classifier, the model might output logits representing its confidence for different breeds:

|                  | logit | softmax  |
| ---------------- | ----- | -------- |
| **Golden Retriever** | 5.23  | 0.975007 |
| **Labrador**    | 1.54  | 0.024348 |
| **Husky**       | -2.37 | 0.000488 |
| **German Shepherd** | -3.50 | 0.000158 |

The basic Softmax formula is:

$Softmax(logits) = \frac{\exp(logits)}{\Sigma \exp(logits)}$

By introducing a temperature parameter $T$, we can control how "confident" the model is in its predictions:

$Softmax(logits, T) = \frac{\exp(logits/T)}{\Sigma \exp(logits/T)}$

For numerical stability, we apply temperature scaling to logits before Softmax:

```python
def scaled_softmax(logits, temperature=1.0):
    scaled_logits = logits/temperature
    return softmax(scaled_logits)
```

When $T = 1$, we have plain old Softmax, which maintains the original relative differences between probabilities.

$T < 1$ creates a sharper distribution, making the model more confident. The highest probability becomes even higher, and the lower probabilities become even lower. At $T$ approaching 0, it becomes deterministic (100% confident).

When $T > 1$, it creates a flatter distribution, making the model less confident, and the differences between probabilities become smaller; predictions become more evenly distributed.

Try it for yourself to see how adjusting the temperature setting affects the Softmax probabilities in a dog breed classification problem:

<html>
<div style="padding: 1rem; border: 1px solid #e2e8f0; border-radius: 0.5rem; background: white; max-width: 800px">
  <div style="margin-bottom: 1rem;">
    <label style="display: block; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem;">
      Temperature: <span id="tempValue">1.0</span>
    </label>
    <input 
      type="range" 
      id="tempSlider"
      min="0.1" 
      max="10" 
      step="0.1" 
      value="1"
      style="width: 100%;"
    />
  </div>
  <canvas id="probabilityChart" width="600" height="300"></canvas>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
<script>
const logits = [5.23, 1.54, -2.37, -3.50];

function softmax(logits, temperature) {
  const scaled = logits.map(l => l/temperature);
  const expScaled = scaled.map(Math.exp);
  const sum = expScaled.reduce((a, b) => a + b, 0);
  return expScaled.map(exp => exp/sum);
}

function updateChart(temperature) {
  const probs = softmax(logits, temperature);
  const probabilities = probs.map(p => p * 100);
  
  if (window.myChart) {
    window.myChart.destroy();
  }
  
  const ctx = document.getElementById('probabilityChart').getContext('2d');
  window.myChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Golden Retriever', 'Labrador', 'Husky', 'German Shepherd'],
      datasets: [{
        label: 'Probability (%)',
        data: probabilities,
        backgroundColor: '#4299e1',
      }]
    },
    options: {
      responsive: true,
      plugins: {
        tooltip: {
          callbacks: {
            label: function(context) {
              const value = context.raw.toFixed(2);
              const logit = logits[context.dataIndex];
              return [`Probability: ${value}%`, `Logit: ${logit}`];
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Probability (%)'
          }
        }
      }
    }
  });
}

document.addEventListener('DOMContentLoaded', function() {
  const tempSlider = document.getElementById('tempSlider');
  const tempValue = document.getElementById('tempValue');
  
  updateChart(1.0);
  
  tempSlider.addEventListener('input', function(e) {
    const temperature = parseFloat(e.target.value);
    tempValue.textContent = temperature.toFixed(1);
    updateChart(temperature);
  });
});
</script>
</html>

## Temperature Scaling in Language Models

In a [Language Model](language-model.md), which predicts a token at a time based on the previous tokens in a sequence, each token is predicted by creating a Softmax probability distribution across the vocabulary and then randomly sampling from that distribution.

The temperature parameter, therefore, affects how much randomness is injected at inference time.

$T = 0$: Deterministic, always selects the highest probability token. However, in practice, you can only approximate it with a very small temperature (since dividing by zero is undefined). Good for math, coding, and fact-based responses.

$T \approx 0.7$: Balanced between coherence and creativity. Industry standard for chat models. Maintains context while allowing natural variation

$T > 1$: Increases randomness. Can generate more creative/diverse outputs. Risk of incoherent or off-topic responses

Temperature is typically applied during inference only. During training, models use $T = 1$ to learn the true probability distribution of the data. Although in the paper, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), they experiment with using higher temperature during training to help the model distinguish between similar classes of items.