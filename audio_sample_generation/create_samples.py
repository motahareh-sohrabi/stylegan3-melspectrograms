import pickle
import librosa
import soundfile
import os
import numpy as np
import glob
from tqdm import tqdm

def create_sample_pkl(G, output_path):
    samples = np.zeros((1000, 1, 128, 128))

    for i in range(10):
        z = torch.randn([100, G.z_dim]).cuda()
        c = None
        samples[i * 100: (i+1) * 100] = G(z, c).cpu().numpy()

    pkl_path = os.path.join(output_path, f"samples.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(samples, f)

def to_audio(output_path, sr=16000, n_fft=512, hop_length=256):
    pkl_path = os.path.join(output_path, f"samples.pkl")

    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)

    for i in tqdm(range(1000)):
        dest = os.path.join(output_path, f"{i}.wav")

        mel = samples[i]

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
    create_samples = False
    create_audio = True

    experiment_paths = {
        "/network/scratch/m/marco.jiralerspong/exvo/experiments/128/unconditional/00001-stylegan2-unconditional-gpus1-batch32-gamma0.0512": "/network/scratch/m/marco.jiralerspong/exvo/experiments/fid/128/unconditional"
    }

    # experiment_paths = {
    #     "/network/scratch/m/marco.jiralerspong/exvo/experiments/256/unconditional/00000-stylegan2-unconditional_256-gpus1-batch32-gamma0.0512": "/network/scratch/m/marco.jiralerspong/exvo/experiments/samples/128/unconditional"
    # }

    for path, output_path in experiment_paths.items():
        for network in tqdm(glob.glob(f"{path}/*.pkl")):
            name = network.split(".pkl")[0]
            name = name.split("network-snapshot-")[-1]

            if name == ["00000"]: continue
            if name[-3:] == "000": continue

            full_output_path = os.path.join(output_path, name)
            os.makedirs(full_output_path, exist_ok=True)

            if create_samples:
                import torch
                with open(network, 'rb') as f:
                    G = pickle.load(f)['G_ema'].cuda()
                    create_sample_pkl(G, full_output_path)

            if create_audio:
                to_audio(full_output_path)
