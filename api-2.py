import librosa
import matplotlib.pyplot as plt
x, sr = librosa.load(librosa.ex('trumpet'))
y = librosa.feature.mfcc(x, sr)
hop_dur = 512 / sr  # default hop length is 512
image = cepstrum(y, hop_dur)
_ = plt.colorbar(image)
plt.tight_layout()
