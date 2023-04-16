from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def fahrenheit_to_celcius(f):
    return (f - 32) * (5 / 9)


def plot_fahrenheit_to_celsius_function():
    x = np.linspace(-100, 100, 400)
    y = fahrenheit_to_celcius(x)

    plt.figure(figsize=(6, 2))
    plt.title("Temperature Conversion: Fahrenheit to Celsius")
    plt.xlabel("Fahrenheit")
    plt.ylabel("Celsius")

    plt.margins(x=0)
    plt.plot(x, y)
    output_file = Path(__file__).resolve().parent.parent.parent / Path(
        "notes/notes/_media/fahrenheit-to-celsius.png"
    )
    plt.savefig(output_file, dpi=300)


plot_fahrenheit_to_celsius_function()
