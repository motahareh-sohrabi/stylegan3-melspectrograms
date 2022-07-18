import pickle
import yaml
import torch
import numpy as np
from utils import to_audio, create_sample_latent_pkl

if __name__ == "__main__":
    """ SETUP """
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
        mel_config = config["mel_config"]
        emotions = config["emotions"]
        num_interpolations = config["interpolation"]["num_interpolations"]

    with open(config["network_path"], 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()

    latent_1 = config["interpolation"]["latent_1_idx"]
    latent_2 = config["interpolation"]["latent_2_idx"]
    difference = latent_2 - latent_1

    """ INTERPOLATION """
    interpolated_samples = np.zeros((num_interpolations + 1, 1, config["dimension"], config["dimension"]))

    for i in range(config["interpolation"]["num_interpolations"] + 1):
        curr_latent = latent_1 + difference * (i/config["interpolation"]["num_interpolations"])
        curr_latent = torch.from_numpy(curr_latent).cuda().unsqueeze(0)
        interpolated_samples[i] = G.synthesis(curr_latent).cpu().numpy()[0]

    to_audio(interpolated_samples, config["interpolation"]["output_path"], mel_config["sample_rate"], mel_config["mel_channels"] * 4, mel_config["hop_length"])