import librosa
from librosa.core.spectrum import amplitude_to_db
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def SpeSub_1(noisy_wav_file):
    noisy_wav, fs = librosa.load(noisy_wav_file, sr=None)

    return clean_wav
