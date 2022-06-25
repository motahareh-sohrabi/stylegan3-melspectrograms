import os
import soundfile
import librosa
import torch

import noisereduce as nr
import numpy as np

def get_wav(file_name, wav_path):
    """ Loads wav dat at index

    :param file_name: File name
    :param wav_path: Base path of wavs
    :return: (np.array of wav data, sr)
    """
    path = os.path.join(wav_path, file_name)
    data, sr = soundfile.read(path)

    # If audio data has 2 channel, just take average of both
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr

def preprocess_audio(wav, sr=16000):
    """ Removes noise from .wav data using noisereduce

    :param wav: np.array of .wav data
    :param sr: Sample rate of sample
    :return: np.array of denoised audio
    """
    reduced_noise = nr.reduce_noise(y=wav, sr=sr)
    trimmed, _ = librosa.effects.trim(reduced_noise, top_db=40)
    return trimmed

def wav_to_mel(wav, melspec, pad_length, pad_val=1e-6):
    """ Converts wav data to mel-spectrogram. The mel is padded to be of size pad_length and converted to log scale

    :param wav: np.array of wav data
    :param melspec: torchaudio MelSpectrogram transform to apply
    :param pad_length: Size of mel-spectrogram to pad to
    :param pad_val: Value to pad with
    :return: np.array of mel-spectrgram
    """
    data = torch.from_numpy(wav)
    mel = melspec(data).detach()
    mel = torch.log(mel+1e-6)/2.0

    if mel.shape[-1] >= pad_length:
        mel = mel[:,:pad_length]
    else:
        mel = torch.cat([mel, torch.ones(mel.shape[0], pad_length - mel.shape[1])*pad_val], dim=1)

    mel = mel.detach().numpy()
    return mel

def mel_to_img(mel, min_val=-6.91, max_val=4.86):
    """ Converts the mel-spectrogram to image scale

    :param mel: np.array of mel-spectrogram
    :param min_val: The minimum value of the mel-spectrogram in log scale
    :param max_val: The maximum value of the mel-spectrogram in log scale
    :return: np.array of mel-spectrogram image in
    """
    mel = (mel - min_val)/(max_val - min_val)
    mel = mel * 255
    mel = mel.astype(np.uint8)
    return mel