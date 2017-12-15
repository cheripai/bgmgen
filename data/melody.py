import numpy as np


MIN_SUBDIVISION = 32
MAX_NOTES = 128

REST_TOKEN = 0
SOS_TOKEN = 129
EOS_TOKEN = 131


class Melody:
    
    def __init__(self, bpm, notes):
        self.bps = bpm / 60
        self.notes = notes
        self.total_duration = self._get_total_duration()
        self.total_beats = self._get_total_beats()
        self.total_subdivisions = self._get_subdivisions()
        self.pianoroll = self._to_pianoroll()
        self.sequence = self._to_sequence()


    def _get_total_duration(self):
        durations = np.zeros((len(self.notes)))
        for i in range(len(self.notes)):
            durations[i] = self.notes[i]["time"] + self.notes[i]["duration"]
        return np.max(durations)

    def _get_total_beats(self):
        return int(round(self.bps * self.total_duration))

    def _get_subdivisions(self):
        return int(round(self.total_beats * MIN_SUBDIVISION))

    def _to_index(self, time):
        depth_prop = time / self.total_duration
        return int(round(depth_prop * self.total_subdivisions))

    def _to_pianoroll(self):
        roll = np.zeros((self.total_subdivisions, MAX_NOTES))
        for note in self.notes:
            for i in range(self._to_index(note["time"]), self._to_index(note["time"] + note["duration"])):
                roll[i, note["midi"]] = 1
        return roll

    def _to_sequence(self):
        sequence = np.zeros(self.total_subdivisions+2)
        sequence[0] = SOS_TOKEN
        for i, slice_ in enumerate(self.pianoroll):
            note = np.where(slice_ == 1)[0]
            if len(note) > 0:
                sequence[i+1] = note[-1]
        sequence[-1] = EOS_TOKEN
        return sequence
