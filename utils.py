import numpy as np
import pretty_midi

REST_TOKEN = 0
SOS_TOKEN = 129
EOS_TOKEN = 130


def song_to_pianoroll(song, tokens):
    pianoroll = np.zeros((len(song), 128))
    for i, index in enumerate(song):
        try:
            notes = eval(tokens[index])
        except:
            notes = []
        for note in notes:
            pianoroll[i, note] = 1
    return pianoroll.T

def write_pianoroll(piano_roll, path, fs=8, instrument_name="Acoustic Grand Piano"):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')
    piano_roll *= 127

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    pm.write(path)


def write_melody(melody, path, resolution=12, instrument_name="Acoustic Grand Piano"):
    mid = pretty_midi.PrettyMIDI(resolution=resolution)
    piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_note = -1
    i, j = 0, 0
    while True:
        if i >= melody.shape[0]:
            break
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
                if j >= melody.shape[0]:
                    break
                note = int(melody[j])
            note = pretty_midi.Note(velocity=127, pitch=prev_note - 1, start=mid.tick_to_time(i), end=mid.tick_to_time(j))
            piano.notes.append(note)
            i = j - 1
        i += 1

    mid.instruments.append(piano)
    mid.write(path)


if __name__ == "__main__":
    pianoroll = np.load("data/pianorolls/around the world - chorus.npy")
    write_pianoroll(pianoroll.T, "test.midi")
