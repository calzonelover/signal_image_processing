import typing
import numpy as np
from scipy.signal import stft

import sounddevice as sd
import soundfile as sf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

AUDIO_FILE = 'water_sound.wav'
L = 100

def get_channel_sound(input, channel: int) -> typing.List[float]:
    return list(map(lambda x: x[channel], input))

if __name__ == "__main__":
    # Inspect audio metadata
    f = sf.SoundFile(AUDIO_FILE)
    N = len(f)
    T = len(f) / f.samplerate
    print("Sample %d"%N)
    print("Sampling rate (f) %d"%f.samplerate)
    print("Duration (T) %.2f second"%T)
    del f

    data, fs = sf.read(AUDIO_FILE, dtype='float32')
    data = get_channel_sound(data, 0)
    signal_ft = stft(data, 1/T)
    # print(signal_ft)
    for y in signal_ft: print(y.shape)
    # sd.play(data, fs)
    # status = sd.wait()

    fig, axs = plt.subplots(3, figsize=(10,8))
    axs[0].plot(data)
    plt.show()

