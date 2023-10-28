import numpy as np
import matplotlib.pyplot as plt

# Define a function to simulate a sine wave (sound wave)
def sound_wave(frequency=20, amplitude=1, duration=0.5, sample_rate=10000):
    t = np.linspace(0, duration, int(sample_rate))
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return t, wave

# Generate a sound wave with 20Hz frequency
t, wave = sound_wave()

# Convert figure size from inches to pixels: dpi = pixels/inch
dpi_value = 80  # Default dpi value in matplotlib. You can adjust if needed.
pixel_width = 600
inch_width = pixel_width / dpi_value
inch_height = inch_width / 2  # Since the aspect ratio is 2:1.

plt.figure(figsize=(inch_width, inch_height))
plt.plot(t, wave, lw=2)

plt.title("20Hz Sound Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (dB)")
plt.grid(True)
plt.yticks([])  # Hide y-axis numbers

plt.tight_layout()
plt.savefig("notes/_media/sound-wave-diagram.png", dpi=dpi_value)
