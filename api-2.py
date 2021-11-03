import librosa
import matplotlib.pyplot as plt
x, sr = librosa.load(librosa.ex('trumpet'))
y = librosa.feature.mfcc(x, sr)
hop_dur = 512 / sr  # default hop length is 512
image = cepstrum(y, hop_dur)
cb = plt.colorbar(image)
cb.outline.set_visible(False)
plt.tight_layout()
