import numpy as np
import matplotlib.pyplot as plt

# Define the sine wave function
def generate_sine_wave(freq, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return t, np.sin(2 * np.pi * freq * t)

# Parameters
freq = 5  # Frequency of the sine wave in Hz (5 cycles in 1 second)
duration = 1  # Duration in seconds

# Generate "continuous" sine wave
t_cont, y_cont = generate_sine_wave(freq, 10000, duration)  # very high sample rate

# Generate sine wave at different sample rates
t_low, y_low = generate_sine_wave(freq, 15, duration)
t_med, y_med = generate_sine_wave(freq, 150, duration)
t_high, y_high = generate_sine_wave(freq, 1500, duration)

# Define DPI and figure size
DPI = 100  # Set the Dots per Inch for the plot
fig_width = 400 / DPI  # Width of figure in inches to ensure the plot is 400 pixels wide
fig_height = 8  # You can adjust this to your liking

# Plot
plt.figure(figsize=(fig_width, fig_height))

plt.subplot(3, 1, 1)
plt.plot(t_cont, y_cont, label="Continuous Wave")
plt.plot(t_low, y_low, '-o', label="Sample Rate: 15 Hz")
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t_cont, y_cont, label="Continuous Wave")
plt.plot(t_med, y_med, '-o', label="Sample Rate: 150 Hz")
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_cont, y_cont, label="Continuous Wave")
plt.plot(t_high, y_high, '-o', label="Sample Rate: 1500 Hz")
plt.xlabel("Time (seconds)")
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust top parameter for better layout

# Save the plot
file_path = "notes/_media/sample-rate-examples.png"
plt.savefig(file_path, dpi=DPI, bbox_inches='tight')
plt.close()
