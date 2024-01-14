import numpy as np
import matplotlib.pyplot as plt

DPI = 100
fig_width = 600 / DPI
fig_height = 4


def snake_activation(x, alpha):
    return x + (1 / alpha) * np.sin(alpha * x) ** 2

# Define a range of x values
x_values = np.linspace(-5, 5, 400)

# Different values of alpha
alphas = [0.5, 1, 2, 5]

# Plotting
plt.figure(figsize=(fig_width, fig_height))

for alpha in alphas:
    plt.plot(x_values, snake_activation(x_values, alpha), label=f'α = {alpha}')

plt.title('Snake Activation Function for Different α Values')
plt.xlabel('x')
plt.ylabel('snake(x)')
plt.legend()
plt.grid(True)

file_path = "notes/_media/snake-activate-examples.png"
plt.savefig(file_path, dpi=DPI, bbox_inches='tight')
plt.close()
