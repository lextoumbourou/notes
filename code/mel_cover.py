import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

DPI = 100

audio_path = librosa.example('trumpet')
y, sr = librosa.load(audio_path)

S = librosa.feature.melspectrogram(y=y, sr=sr)

S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(3, 3))

librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)

plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)

plt.savefig("notes/_media/melspectrogram-cover.png", dpi=DPI, bbox_inches='tight', pad_inches=0)
