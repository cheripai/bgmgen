import numpy as np
import os
import pickle
import sys
from PIL import Image
from random import shuffle

IMG_SIZE = 224

def get_filename(fname):
    return fname.split("/")[-1].split(".")[0].lower()

if __name__ == "__main__":
    bg_path = sys.argv[1]
    pianoroll_path = sys.argv[2]

    pianoroll_paths = [os.path.join(pianoroll_path, path) for path in os.listdir(pianoroll_path) if path.endswith("npy")]
    bg_paths = [os.path.join(bg_path, get_filename(path).split("-")[0].strip()+".jpg") for path in pianoroll_paths]

    bgs = np.zeros((len(pianoroll_paths), 3, IMG_SIZE, IMG_SIZE))
    for i, bg_path in enumerate(bg_paths):
        bg = Image.open(bg_path)
        bg = bg.resize((IMG_SIZE, IMG_SIZE))
        bg = np.swapaxes(bg, 0, 1)
        bg = np.swapaxes(bg, 0, 2)
        bgs[i] = bg


    songs = []
    tokens = []
    for i, pianoroll_path in enumerate(pianoroll_paths):
        pianoroll = np.load(pianoroll_path)
        song = np.zeros(pianoroll.shape[0])
        for j, step in enumerate(pianoroll):
            notes = np.where(step == 1)[0]
            if len(notes) > 3:
                notes = sorted(np.random.choice(notes, 3, replace=False))
            token = str(list(notes))
            if token not in tokens:
                tokens.append(token)
            song[j] = tokens.index(token)
        songs.append(song)

    tokens.append("SOS")

    print("{} songs with {} unique tokens".format(len(songs), len(tokens)))
    np.save("bgs.npy", bgs)
    pickle.dump(songs, open("songs.pkl", "wb"))
    pickle.dump(tokens, open("tokens.pkl", "wb"))
