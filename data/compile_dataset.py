import numpy as np
import os
import sys
from PIL import Image
from random import shuffle

MAX_LENGTH = 2048
END_SONG_TOKEN = 130
IMG_SIZE = 224

def get_filename(fname):
    return fname.split("/")[-1].split(".")[0]

if __name__ == "__main__":
    bg_path = sys.argv[1]
    sequence_path = sys.argv[2]

    sequence_paths = [os.path.join(sequence_path, path) for path in os.listdir(sequence_path) if path.endswith("npy")]
    bg_paths = [os.path.join(bg_path, get_filename(path).split("-")[0].strip()+".jpg") for path in sequence_paths]

    bgs = np.zeros((len(sequence_paths), 3, IMG_SIZE, IMG_SIZE))
    for i, bg_path in enumerate(bg_paths):
        bg = Image.open(bg_path)
        bg = bg.resize((IMG_SIZE, IMG_SIZE))
        bg = np.swapaxes(bg, 0, 1)
        bg = np.swapaxes(bg, 0, 2)
        bgs[i] = bg


    sequences = np.zeros((len(sequence_paths), MAX_LENGTH))
    for i, sequence_path in enumerate(sequence_paths):
        sequence = np.load(sequence_path)
        sequence_length = min(MAX_LENGTH, sequence.shape[0])
        sequences[i,:sequence_length] = sequence[:sequence_length]
        sequences[i,sequence_length:] = END_SONG_TOKEN
        sequences[i,sequence_length-1] = END_SONG_TOKEN

    np.save("bgs.npy", bgs)
    np.save("sequences.npy", sequences)
