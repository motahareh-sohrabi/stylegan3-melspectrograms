import pickle
import librosa
import soundfile
import os
import numpy as np
import glob
from tqdm import tqdm

model_path = "/network/scratch/m/marco.jiralerspong/exvo/experiments/128/conditional/00000-stylegan2-conditional-gpus1-batch32-gamma0.0512"
samples_path = "/network/scratch/m/marco.jiralerspong/exvo/experiments/fid/128/final/untruncated"

emotions = ["Awe", "Excitement", "Amusement", "Awkwardness", "Fear", "Horror", "Distress", "Sadness", "Surprise"]
total_samples = 1000
batch_samples = 100

def create_sample_pkl(G, output_path, class_idx):
    samples = np.zeros((total_samples, 1, 128, 128))
    ws = np.zeros((total_samples, 12, 512))

    for i in range(total_samples//batch_samples):
        z = torch.randn([batch_samples, G.z_dim]).cuda()
        c = torch.zeros(batch_samples, 9).cuda()

        for j in range(batch_samples):
            c[j, class_idx] = 1

        w = G.mapping(z, c)
        ws[i * batch_samples: (i+1) * batch_samples] = w.cpu().numpy()
        samples[i * batch_samples: (i+1) * batch_samples] =  G.synthesis(w).cpu().numpy()

    output_path = os.path.join(output_path, emotions[class_idx])
    os.makedirs(output_path, exist_ok=True)
    pkl_path = os.path.join(output_path, f"samples.pkl")
    latent_path = os.path.join(output_path, f"latents.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(samples, f)

    with open(latent_path, "wb") as f:
        pickle.dump(ws, f)

def to_audio(output_path, class_idx, sr=16000, n_fft=512, hop_length=256):
    pkl_path = os.path.join(output_path, emotions[class_idx], f"samples.pkl")

    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)

    for i in tqdm(range(total_samples)):
        dest = os.path.join(output_path, emotions[class_idx], f"{i}.wav")

        mel = samples[i]

        if mel.max() -mel.min() != 0:
            mel = (mel - mel.min())/(mel.max() - mel.min())
        mel = mel * (6.907  + 4.859)
        mel = mel - 6.907
        mel = np.exp(2.0*mel)-1e-6

        audio_data = librosa.feature.inverse.mel_to_audio(
            mel, sr=sr, n_fft=n_fft, hop_length=hop_length, norm=1
        )
        audio_data = audio_data.T
        soundfile.write(dest, audio_data, sr, format='WAV', subtype='FLOAT')


if __name__ == "__main__":
    # TODO add argparse
    create_samples = True
    create_audio = True

    for network in glob.glob(f"{model_path}/*.pkl"):
        name = network.split(".pkl")[0]
        name = name.split("network-snapshot-")[-1]

        if name != "004800": continue

        full_output_path = os.path.join(samples_path, name)
        os.makedirs(full_output_path, exist_ok=True)

        if create_samples:
            import torch
            with open(network, 'rb') as f:
                G = pickle.load(f)['G_ema'].cuda()
            for class_idx in range(len(emotions)):
                create_sample_pkl(G, full_output_path, class_idx)

        if create_audio:
            for class_idx in range(len(emotions)):
                to_audio(full_output_path, class_idx)
