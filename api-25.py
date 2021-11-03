import librosa
import matplotlib.pyplot as plt
x, _ = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
_, axs = plt.subplots(2, figsize=(8, 3))
plt.subplots_adjust(hspace=0)
waveform(
    x[0, :],
    text='Left ',  # empty space for same size as 'Right'
    linewidth=0.5,
    background='#389DCD',
    color='#1B5975',
    ax=axs[0],
)
waveform(
    x[1, :],
    text='Right',
    linewidth=0.5,
    background='#CA5144',
    color='#742A23',
    ax=axs[1],
)
