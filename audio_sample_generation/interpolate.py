import pickle
import librosa
import soundfile
import os
import numpy as np
import glob
import torch
from tqdm import tqdm

network_path = "/network/scratch/m/marco.jiralerspong/exvo/experiments/128/conditional/00000-stylegan2-conditional-gpus1-batch32-gamma0.0512"
samples_path = "/network/scratch/m/marco.jiralerspong/exvo/experiments/interpolation/young_old"

with open(os.path.join(samples_path, "004800/Amusement/latents.pkl"), "rb") as f:
    latents = pickle.load(f)

latent_young = latents[48]
latent_old = latents[0]

diff = latent_old - latent_young

network = os.path.join(network_path, "network-snapshot-004800.pkl")

with open(network, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

for i in range(9):
    curr_latent = latent_young + (diff * i)/8
    curr_latent = torch.from_numpy(curr_latent).cuda().unsqueeze(0)

    mel = G.synthesis(curr_latent).cpu().numpy()[0]
    mel = (mel - mel.min())/(mel.max() - mel.min())
    mel = mel * (6.907  + 4.859)
    mel = mel - 6.907
    mel = np.exp(2.0*mel)-1e-6

    audio_data = librosa.feature.inverse.mel_to_audio(
        mel, sr=16000, n_fft=512, hop_length=256, norm=1
    )
    audio_data = audio_data.T

    dest = os.path.join(samples_path, f"sample_{i}.wav")
    soundfile.write(dest, audio_data, 16000, format='WAV', subtype='FLOAT')