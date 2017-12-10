import numpy as np
from mido import Message, MidiFile, MidiTrack


def write_pianoroll(pianoroll, path, min_subdivision=32):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    for i in range(pianoroll.shape[0]):
        for j in range(pianoroll.shape[1]):
            if pianoroll[i, j] == 0:
                track.append(Message("note_off", note=j, velocity=127, time=i*min_subdivision))
            elif pianoroll[i, j] == 1:
                track.append(Message("note_on", note=j, velocity=127, time=i*min_subdivision))
            else:
                raise Exception("Invalid value in pianoroll at {}, {}".format(i, j))

    mid.save(path)
