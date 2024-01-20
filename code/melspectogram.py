import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import soundfile as sf

DPI = 100

audio_path = librosa.example('trumpet')
y, sr = librosa.load(audio_path)

S = librosa.feature.melspectrogram(y=y, sr=sr)

S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(8, 4))
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (trumpet.mp3)')
plt.tight_layout()

audio_file_path = "notes/_media/trumpet_example.mp3"
sf.write(audio_file_path, y, sr)

plt.savefig("notes/_media/melspectrogram-example.png", dpi=DPI)
