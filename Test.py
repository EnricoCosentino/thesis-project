from scipy.io import wavfile # Libreria per leggere file WAVE
from scipy.signal import resample
from audio_to_spectrogram_via_STFT import audio_to_spectrogram_via_STFT

samplerate, data = wavfile.read("02 - Moar Ghosts n' Stuff.wav")

data = (data[:,0] + data[:,1])/2
rd = resample(data, len(data)//2) #reduced data

#specData, featureRate = audio_to_spectrogram_via_STFT(data, )