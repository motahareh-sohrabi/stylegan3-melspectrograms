import pickle
import yaml

from utils import to_audio, create_sample_latent_pkl

if __name__ == "__main__":
    """ SETUP """
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
        mel_config = config["mel_config"]
        emotions = config["emotions"]

    with open(config["network_path"], 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()

    """ SAMPLE GENERATION """
    samples = create_sample_latent_pkl(G, config["num_samples"], config["dimension"], config["output_path"], emotions.index("Awe"))
    to_audio(samples, config["output_path"], mel_config["sample_rate"], mel_config["mel_channels"] * 4, mel_config["hop_length"])
