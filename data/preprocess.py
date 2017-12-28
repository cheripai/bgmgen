import numpy as np
import os
import pretty_midi as pm
import sys
from melody import Melody

MELODY_INDEX = 0


def replace_extension(name, ext):
    base = "".join(name.split(".")[:-1])
    if "." in ext:
        return base + ext
    else:
        return base + "." + ext


if __name__ == "__main__":
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for midi_file in os.listdir(source_dir):
        if midi_file.endswith("midi") or midi_file.endswith("mid"):
            midi_data = pm.PrettyMIDI(os.path.join(source_dir, midi_file), resolution=8)
            print(midi_data.resolution)
            melody = Melody(midi_data.instruments[MELODY_INDEX].get_piano_roll())
            np.save(os.path.join(target_dir, replace_extension(midi_file, "npy").lower().replace("'", "")), melody.sequence)
