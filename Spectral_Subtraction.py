import librosa
from librosa.core.spectrum import amplitude_to_db
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def SpeSub_1(noisy_wav_file):
    noisy_wav, fs = librosa.load(noisy_wav_file, sr=None)
    # Calculate the spectrum of a noisy data
    S_noisy = librosa.stft(noisy_wav, n_fft=256, hop_length=128, win_length=256)  # D * T
    D, T = np.shape(S_noisy)
    Mag_noisy = np.abs(S_noisy)
    Phase_noisy = np.angle(S_noisy)
    Power_noisy = Mag_noisy ** 2
    # print(fs)
    # Estimate the energy of the noisy signal
    # Because the noisy signal is unknown, suppose the first 30 frames of the noisy signal is the noise
    Mag_noise = np.mean(np.abs(S_noisy[:, :15]), axis=1, keepdims=True)  # D * 1
    Power_noise = Mag_noise ** 2
    Power_noise = np.tile(Power_noise, [1, T])
    # Power minus
    Power_enhance = Power_noisy - Power_noise
    # Make the energy > 0
    Power_enhance[Power_enhance < 0] = 0
    Mag_enhance = np.sqrt(Power_enhance)

    # Mag minus
    # Mag_enhance = np.sqrt(Power_noisy) - np.sqrt(Power_noise)
    # Mag_enhance[Mag_enhance < 0] = 0

    # Recover the signal
    S_enhance = Mag_enhance*np.exp(1j*Phase_noisy)
    enhance = librosa.istft(S_enhance, hop_length=128, win_length=256)
    sf.write("./clean.wav", enhance, fs)
    # print("Finish!")
    return True



if __name__ == "__main__":
    SpeSub_1('./data/noisy_data.wav')