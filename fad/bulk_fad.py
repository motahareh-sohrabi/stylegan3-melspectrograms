import os

from frechet_distance import calculate_frechet_distance
from utils import load_activation_statistics, get_activation_statistics

if __name__ == "__main__":
    base_path = ""
    emotions = ["Awe", "Excitement", "Amusement", "Awkwardness", "Fear", "Horror", "Distress", "Sadness", "Surprise"]

    for emotion in emotions:
        mu1, sigma1 = get_activation_statistics(os.path.join(base_path, emotion), 32)
        mu2, sigma2 = load_activation_statistics(f"{emotion}.pkl")

        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        print(f"{emotion}: {fid:.2f}")