import pandas as pd
import numpy as np
import os
import torch
import torchaudio
import librosa
import soundfile
import matplotlib.pyplot as plt
import noisereduce as nr
import imageio
import json
import shutil
import glob
import yaml

from tqdm import tqdm
from transforms import get_wav, preprocess_audio, wav_to_mel, mel_to_img
from utils import get_ratio, save_image, parse_file_id

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    """ SETUP """
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
        mel_config = config["mel_config"]

    df = pd.read_csv(config["csv_path"])
    df["File_ID"] = df["File_ID"].apply(parse_file_id)

    # We will add lengths and whether the sample is empty to the df
    lengths_dict = {}
    empty_dict = {}

    # Patch to mel filters to make it invertable with librosa
    melspec = torchaudio.transforms.MelSpectrogram(
        mel_config["sample_rate"],
        hop_length=mel_config["hop_length"],
        n_fft=mel_config["mel_channels"] * 4,
        n_mels=mel_config["mel_channels"]
    )

    melspec.mel_scale.fb = torch.tensor(
        librosa.filters.mel(mel_config["sample_rate"], n_mels=mel_config["mel_channels"],
                            n_fft=mel_config["mel_channels"] * 4, norm=1).T
    )

    """ DATA PROCESSING """
    for file_name in tqdm(glob.glob(f"{config['wav_path']}/*.wav")):
        # Load
        data, sr = get_wav(file_name, config["wav_path"])

        # Remove noise/trim audio
        data = preprocess_audio(data, sr)

        # Get length and filter out empty/very short samples
        length = data.shape[0]/16000
        lengths_dict[file_name] = length

        # If sample after trimming is less than 0.05s, is essentially silent => remove
        if length < 0.05:
            empty_dict[file_name] = True
            continue

        # Convert to mel
        data = wav_to_mel(data, melspec, mel_config["pad_length"], mel_config["pad_val"])

        # Filter out silent samples
        empty = get_ratio(data) > 0.98
        empty_dict[file_name] = empty

        # Remove samples where at least 98% of pixels are of low amplitude (sound mostly empty)
        if empty: continue

        # Convert to image scale
        data = mel_to_img(data)

        # Output
        save_image(data, config["output_path"], file_name)

    """ CSV/JSON PROCESSING """
    # Update CSV with length and whether the samples are empty
    df["Length"] = lengths_dict.values()
    df["Empty"] = empty_dict.values()

    df.to_csv(config["csv_output_path"], index=False)

    # Create dataset.json for conditional training
    labels = []
    for _, row in df.iterrows():
        # We don't have access to emotion values for Test data
        print(row["Empty"])
        if row["Split"] == "Test" or row["Empty"]: continue


        for i, emotion in enumerate(config["emotions"]):
            if row[emotion] == 1:
                labels.append([f"{row['File_ID']}.png", i])

    json_dict = {"labels": labels}
    json_path = os.path.join(config["output_path"], "dataset.json")
    with open(json_path, "w") as f:
        json.dump(json_dict, f)