import librosa
x, _ = librosa.load(librosa.ex('trumpet'))
waveform(x, background='#363636', color='#f6f6f6')
