import librosa
import matplotlib.pyplot as plt
x, sr = librosa.load(librosa.ex('trumpet'))
y = librosa.feature.melspectrogram(x, sr, n_mels=40, fmax=4000)
y_db = librosa.power_to_db(y, ref=np.max)
hop_dur = 512 / sr  # default hop length is 512
centers = librosa.mel_frequencies(n_mels=40, fmax=4000)
image = spectrum(y_db, hop_dur, centers)
_ = plt.colorbar(image, format='%+2.0f dB')
plt.tight_layout()
