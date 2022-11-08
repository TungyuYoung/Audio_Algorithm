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
    Mag_noise = np.mean(np.abs(S_noisy[:, :30]), axis=1, keepdims=True)  # D * 1
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
    S_enhance = Mag_enhance * np.exp(1j * Phase_noisy)
    enhance = librosa.istft(S_enhance, hop_length=128, win_length=256)
    sf.write("./clean.wav", enhance, fs)
    # print("Finish!")
    return True


def SS_over_minus(noisy_wav_file):
    alpha = 4
    gamma = 1
    beta = 0.0001
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
    Mag_noise = np.mean(np.abs(S_noisy[:, :30]), axis=1, keepdims=True)  # D * 1
    Power_noise = Mag_noise ** 2
    Power_noise = np.tile(Power_noise, [1, T])
    # Power minus
    Power_enhance = np.power(Power_noisy, gamma) - alpha * np.power(Power_noise, gamma)
    Power_enhance = np.power(Power_enhance, 1 / gamma)
    mask = (Power_enhance >= beta * Power_noise) - 0
    # print(mask.shape)
    Power_enhance = mask * Power_enhance + beta * (1 - mask) * Power_noise
    Mag_enhance = np.sqrt(Power_enhance)
    # Make the energy > 0

    # Mag minus
    # Mag_enhance = np.sqrt(Power_noisy) - np.sqrt(Power_noise)
    # Mag_enhance[Mag_enhance < 0] = 0

    # Recover the signal
    S_enhance = Mag_enhance * np.exp(1j * Phase_noisy)
    enhance = librosa.istft(S_enhance, hop_length=128, win_length=256)
    sf.write("./clean_2.wav", enhance, fs)
    # print("Finish!")
    return True


def SS_Smooth(noisy_wav_file):
    alpha = 4
    gamma = 1
    beta = 0.0001
    noisy_wav, fs = librosa.load(noisy_wav_file, sr=None)
    # Calculate the spectrum of a noisy data
    S_noisy = librosa.stft(noisy_wav, n_fft=256, hop_length=128, win_length=256)  # D * T
    D, T = np.shape(S_noisy)
    Mag_noisy = np.abs(S_noisy)
    Phase_noisy = np.angle(S_noisy)
    Phase_noisy = Mag_noisy ** 2
    Mag_noise = np.mean(np.abs(S_noisy[:, :15]), axis=1, keepdims=True)
    Power_noise = Mag_noise ** 2
    Power_noise = np.tile(Power_noise, [1, T])

    Mag_noisy_new = np.copy(Mag_noisy)

    k = 1
    for t in range(k, T - k):
        Mag_noisy_new[:, t] = np.mean(Mag_noisy[:, t - k: t - k + 1], axis=1)

    Power_noisy = Mag_noisy_new ** 2
    Power_enhance = np.power(Power_noisy, gamma) - alpha * np.power(Power_noise, gamma)
    Power_enhance = np.power(Power_enhance, 1 / gamma)

    mask = (Power_enhance >= beta * Power_noise) - 0
    Power_enhance = mask * Power_enhance + beta * (1 - mask) * Power_noise
    Mag_enhance = np.sqrt(Power_enhance)
    Mag_enhance_new = np.copy(Mag_enhance)

    max_nr = np.max(np.abs(S_noisy[:, 15]) - Mag_noise, axis=1)

    k_ = 1
    for t in range(k_, T - k_):
        index = np.where(Mag_enhance[:, t] < max_nr)[0]
        temp = np.min(Mag_enhance[:, t-k_:t+k_+1], axis=1)
        Mag_enhance_new[index, t] = temp[index]

    S_enhance = Mag_enhance_new * np.exp(1j * Phase_noisy)
    enhance = librosa.istft(S_enhance, hop_length=128, win_length=256)
    sf.write('./clean_3.wav', enhance, fs)
    return True


if __name__ == "__main__":
    SS_Smooth('./data/noisy_data.wav')
