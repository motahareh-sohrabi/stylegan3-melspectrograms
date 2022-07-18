# Generating Diverse Vocal Bursts with StyleGAN2 and MEL-Spectrograms
Repository for the paper "Generating Diverse Vocal Bursts with StyleGAN2 and MEL-Spectrogram". Forked from the stylegan3 repository (https://github.com/NVlabs/stylegan3).

## File structure
- `data_processing/`: Utilities for converting the audio dataset to mel-spectrograms that you can try StyleGAN2 on.
- `audio_sample_generation/`: Utilities for generating audio samples or performing interpolation using a trained model.
- `fad/`: Utilities for computing FAD along with pre-computed statistics of the data.
- `stylegan3/`: Original StyleGAN3 repository cloned here (we use the StyleGAN3 repository since it has additional convenience functions and still allows for training with the StyleGAN2-Ada setup).
- `audio_inversion_experiments.ipynb`: Mel-spectrogram round-tripping experiments showing the effect of various parameters on audio inversion quality.
- `train.sh`: Training script for StyleGAN