import numpy as np
import pretty_midi as pm


def to_piano_roll(pm_container, fs=8):
    roll = np.copy(pm_container.get_piano_roll(fs=fs).T)

    # transform note velocities into 1s
    roll = (roll > 0).astype(float)

    # remove empty beginning
    for i, col in enumerate(roll):
        if col.sum() != 0:
            break
    roll = roll[i:]

    if roll.sum() == 0:
        raise Exception("Roll is empty")

    return roll

if __name__ == "__main__":
    pm_container = pm.PrettyMIDI("data/Pop_Music_Midi/Around The World - Verse.midi")
    del pm_container.instruments[1]
    roll = to_piano_roll(pm_container)
    np.save("test.npy", roll)
