import librosa
x, _ = librosa.load(librosa.ex('trumpet'))
waveform(x, text='Trumpet')
