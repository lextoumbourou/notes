import matplotlib.pyplot as plt
import numpy as np

DPI = 100

a, b, c = 1, -2, 1

x = np.linspace(-10, 10, 400)

y = a * x**2 + b * x + c

plt.figure(figsize=(4, 3))
plt.plot(x, y, label='f(x) = {}x² + {}x + {}'.format(a, b, c))
plt.title('Quadratic Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()

file_path = "notes/_media/quadratic-function.png"
plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
plt.close()
