import matplotlib.pyplot as plt
import numpy as np

DPI = 100

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-10, 10, 200)
y = relu(x)

# Plotting
plt.figure(figsize=(6, 3))
plt.plot(x, y, label='ReLU Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU Activation Function')
plt.grid(True)
plt.legend()

file_path = "notes/_media/relu-activation-plot.png"
plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
plt.close()
