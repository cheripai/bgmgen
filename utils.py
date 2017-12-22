import numpy as np
import pretty_midi as pm

REST_TOKEN = 0
SOS_TOKEN = 129
EOS_TOKEN = 130


def write_pianoroll(input_roll, path, min_subdivision=32):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    pianoroll = np.zeros((input_roll.shape[0] + 1, input_roll.shape[1]))
    pianoroll[1:, :] = input_roll
    for i in range(pianoroll.shape[0]):
        for j in range(pianoroll.shape[1]):
            if j == 127:
                time = min_subdivision
            else:
                time = 0
            if pianoroll[i, j] == 0:
                track.append(Message("note_off", note=j, velocity=127, time=time))
            elif pianoroll[i, j] == 1 and pianoroll[i - 1, j] == 0:
                track.append(Message("note_on", note=j, velocity=127, time=time))

    mid.save(path)


def write_melody(melody, path, resolution=50, instrument_name="Acoustic Grand Piano"):
    mid = pm.PrettyMIDI(resolution=resolution)
    piano = pm.Instrument(program=pm.instrument_name_to_program(instrument_name))

    prev_note = -1
    i, j = 0, 0
    while True:
        note = int(melody[i])
        if note == SOS_TOKEN:
            pass
        elif note == REST_TOKEN:
            pass
        elif note == EOS_TOKEN:
            break
        else:
            prev_note = note
            j = i
            while note == prev_note:
                j += 1
                note = int(melody[j])
            note = pm.Note(velocity=127, pitch=prev_note - 1, start=mid.tick_to_time(i), end=mid.tick_to_time(j))
            piano.notes.append(note)
            i = j - 1
        i += 1

    mid.instruments.append(piano)
    mid.write("test.mid")


if __name__ == "__main__":
    melody = np.load("data/Pop_Melodies/Around The World - Chorus.npy")
    write_melody(melody, "test.midi")
