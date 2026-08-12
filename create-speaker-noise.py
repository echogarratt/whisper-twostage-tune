import os
import glob
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
from pathlib import Path

DONOR_PATH = "./path_to_base_wavs"
SAMPLE_PATH = "./path_to_target_wavs" # remember test/train split
OUTPUT_PATH = "./path_to_output_wavs"
SPEAKERS = ["F000", "M001", "F002"]




def extract_noise_profile(folder, n_fft=2048, hop_length=512, threshold_db=-35):
    spectra = []

    for f in Path(folder).glob('*.wav'):
        audio, _ = librosa.load(f, sr=None, mono=True)

        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        P = np.abs(D)**2

        frame_db = librosa.power_to_db(P.mean(axis=0), ref=np.max)
        speech = frame_db > threshold_db

        if speech.any():
            spectra.append(P[:, speech])

    return np.hstack(spectra).mean(axis=1)



def spectral_subtract(audio, noise_profile, amount=1.0, floor=0.02, n_fft=2048, hop_length=512):
    
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    P = np.abs(D)**2

    P_clean = np.maximum(P - amount*noise_profile[:, None], floor*noise_profile[:, None])
    
    return librosa.istft(np.sqrt(P_clean) * np.exp(1j*np.angle(D)), hop_length=hop_length, length=len(audio))






def spectral_add(audio, noise_profile, amount=1.0, n_fft=2048, hop_length=512):
    
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    P = np.abs(D)**2

    P_noisy = P + amount*noise_profile[:, None]
    
    return librosa.istft(np.sqrt(P_noisy) * np.exp(1j*np.angle(D)), hop_length=hop_length, length=len(audio))



sample_noise = extract_noise_profile(SAMPLE_PATH)

directory_path = Path(SAMPLE_PATH)
sample_files = {}

for file_path in directory_path.glob("*.wav"):
    audio, sr = librosa.load(file_path, sr=None)
    sample_files[file_path.name] = spectral_subtract(audio, sample_noise)

for speaker in SPEAKERS:
    noise = extract_noise_profile(DONOR_PATH + "/" + speaker + "/train")
    noise_added = {}
    
    for filename, sample in sample_files.items():
        noise_added[filename] = spectral_add(sample, noise)
        
    speaker_output_path = Path(OUTPUT_PATH) / speaker
    speaker_output_path.mkdir(parents=True, exist_ok=True)
    
    for filename, audio_data in noise_added.items():
        file_save_path = speaker_output_path / filename
        sf.write(file_save_path, audio_data, sr)
