import matplotlib.pyplot as plt
import numpy as np

DPI = 100

# Function to convert frequency in Hz to Mel scale
def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)

# Frequency range from 0 to 6000 Hz
frequencies_hz = np.linspace(0, 6000, 500)
mel_values = hz_to_mel(frequencies_hz)

# Creating the plot
plt.figure(figsize=(6, 2))
plt.plot(frequencies_hz, mel_values, label='Hz to Mel')
plt.title('Frequency vs. Mel Scale')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mel Scale')
plt.grid(True)
plt.legend()

file_path = "notes/_media/mel-scale-plot.png"
plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
plt.close()
