import numpy as np
import matplotlib.pyplot as plt

DPI = 100

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

x = np.linspace(-10, 10, 200)

alphas = [0.01, 0.1, 0.2, 0.3]

# Plotting for each alpha value
for alpha in alphas:
    y = leaky_relu(x, alpha)
    plt.plot(x, y, label=f"alpha = {alpha}")

plt.title("Leaky ReLU Function with Different Alpha Values")
plt.xlabel("Input value (x)")
plt.ylabel("Output value (f(x))")
plt.legend()
plt.grid(True)

file_path = "notes/_media/leaky-relu-activate-examples.png"
plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
plt.close()
