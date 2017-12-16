import numpy as np
from mido import Message, MidiFile, MidiTrack

REST_TOKEN = 0
SOS_TOKEN = 129
EOS_TOKEN = 130

def write_pianoroll(input_roll, path, min_subdivision=32):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    pianoroll = np.zeros((input_roll.shape[0]+1,input_roll.shape[1]))
    pianoroll[1:,:] = input_roll
    for i in range(pianoroll.shape[0]):
        for j in range(pianoroll.shape[1]):
            if j == 127:
                time = min_subdivision
            else:
                time = 0
            if pianoroll[i, j] == 0:
                track.append(Message("note_off", note=j, velocity=127, time=time))
            elif pianoroll[i, j] == 1 and pianoroll[i-1, j] == 0:
                track.append(Message("note_on", note=j, velocity=127, time=time))

    mid.save(path)

def write_melody(melody, path, min_subdivision=32):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    for i in range(len(melody)):
        note = int(melody[i])
        if note == SOS_TOKEN:
            continue
        elif note == EOS_TOKEN:
            break
        elif note == REST_TOKEN:
            prev_note = int(melody[i-1])-1
            if prev_note >= 0 and prev_note < 128:
                track.append(Message("note_off", note=prev_note, velocity=127, time=min_subdivision))
            else:
                track.append(Message("note_on", time=min_subdivision))
        else:
            prev_note = int(melody[i-1])-1
            if prev_note == note-1:
                track.append(Message("note_on", time=min_subdivision))
            else:
                track.append(Message("note_on", note=note-1, velocity=127, time=min_subdivision))

    mid.save(path)

if __name__ == "__main__":
    import bcolz
    melody = bcolz.open("data/melody/Super_Mario_Kart/DonutPlains.bcolz")
    write_melody(melody, "test.midi")
    # pianoroll = bcolz.open("data/pianoroll/Super_Mario_Kart/DonutPlains.bcolz")
    # write_pianoroll(pianoroll, "test.midi")
