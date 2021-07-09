#! /usr/bin/env python

""" Downsample and normalize voice samples from:
https://github.com/vocobox/human-voice-dataset .
"""

import os
import wave
import pickle
import time
import numpy as np

from scipy.signal import resample
from typing import Tuple


repo_path = os.path.join("..", "..", "human-voice-dataset")


def load_wav(name: str, verbose: bool = True) -> Tuple[np.ndarray, int]:
    data = None
    framerate = None
    with wave.open(name, "rb") as f:
        data = f.readframes(-1)
        framerate = f.getframerate()
        if verbose:
            print(
                f"Loaded {name}, {f.getnchannels()} channels, "
                f"{f.getsampwidth()} bytes/sample, {framerate / 1000:.1f}kHz."
            )

        numpy_type = {2: "int16", 4: "int32", 8: "int64"}[f.getsampwidth()]
        data = np.frombuffer(data, dtype=numpy_type)

    return data, framerate


def normalize_wav(v: np.ndarray) -> np.ndarray:
    median = np.median(v)
    span = np.quantile(np.abs(v), 0.95)
    return (v - median) / span


def generate_vowel_dataset() -> dict:
    vowels = {}

    path = os.path.join(
        repo_path, "data", "voices", "martin", "voyels", "exports", "mono"
    )
    file_names = {
        "a": "_-a-c3-2.wav",
        "e": "_-e-c3-2.wav",
        "i": "_-i-c3-2.wav",
        "o": "_-o-c3-2.wav",
        "u": "_-ou-c3-2.wav",
    }

    for vowel, name in file_names.items():
        original, sampling = load_wav(os.path.join(path, name))
        normalized = normalize_wav(original)
        subsampled = resample(normalized, len(normalized) * 8000 // sampling)

        vowels[vowel] = subsampled

    return vowels


def generate_pitch_dataset() -> dict:
    pitches = {}

    path = os.path.join(
        repo_path, "data", "voices", "martin", "notes", "exports", "mono"
    )
    file_names = {_: _.upper() + "3.wav" for _ in "cdefgab"}

    for pitch, name in file_names.items():
        original, sampling = load_wav(os.path.join(path, name))
        normalized = normalize_wav(original)
        subsampled = resample(normalized, len(normalized) * 8000 // sampling)

        pitches[pitch] = subsampled

    return pitches


if __name__ == "__main__":
    t0 = time.time()

    print("Generating and storing vowel dataset...")
    vowels = generate_vowel_dataset()
    with open("vowel_dataset.pkl", "wb") as f:
        pickle.dump(vowels, f)

    print("Generating and storing pitch dataset...")
    pitches = generate_pitch_dataset()
    with open("pitch_dataset.pkl", "wb") as f:
        pickle.dump(pitches, f)

    print(f"Took {time.time() - t0:.2f} seconds.")
