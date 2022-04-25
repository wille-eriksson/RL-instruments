import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from sklearn.preprocessing import normalize
from pretty_midi import PrettyMIDI, Instrument, Note
import librosa


class PianoRollManager():

    granularity_dict = {"whole": 1,
                        "half": 2,
                        "quarter": 4,
                        "eight": 8,
                        "sixteenth": 16,
                        "thirty-second": 32}

    def __init__(self, piano_roll, bpm, note_granularity, sr=8092, base_note_number=60):
        self.piano_roll = piano_roll
        self.bpm = bpm
        self.sr = sr
        self.base_note_number = base_note_number

        try:
            self.note_granularity_factor = self.granularity_dict[note_granularity]
        except:
            raise ValueError("Valid values for \"note_granularity\" are: %s." % (
                ", ".join(self.granularity_dict.keys())))

        self.midi = self._create_midi_from_piano_roll()
        self.audio = self._synthesize()
        self.stft = normalize(
            np.abs(librosa.stft(
                self.audio, n_fft=1024, win_length=1024, hop_length=1024)),
            axis=0)

    def get_stft(self):
        return self.stft

    def get_audio(self):
        return self.audio

    def save_audio(self, path):
        sf.write(path, self.audio, self.sr)

    def _synthesize(self):
        audio = self.midi.synthesize(self.sr)
        note_length = self.bpm/60/self.note_granularity_factor
        n_samples = int(self.sr * note_length)*self.piano_roll.shape[1]
        return audio[:n_samples]

    def _create_midi_from_piano_roll(self):

        midi_data = PrettyMIDI()
        instrument = Instrument("")

        for note in range(self.piano_roll.shape[0]):
            note_roll = self.piano_roll[note, :]
            note_number = self._get_note_number(note)
            notes = self._get_notes(note_number, note_roll)

            for note in notes:
                instrument.notes.append(note)

        midi_data.instruments.append(instrument)

        return midi_data

    def _get_notes(self, note_number, note_roll):
        note_length = self.bpm/60/self.note_granularity_factor
        notes = []

        for idx, pressed in enumerate(note_roll):

            if (pressed and idx == 0):
                start = idx * note_length
            elif (not pressed and idx == 0):
                pass
            elif pressed and not prev_pressed:
                start = idx * note_length
            elif not pressed and prev_pressed:
                end = idx * note_length
                new_note = Note(velocity=100, pitch=note_number,
                                start=start, end=end)
                notes.append(new_note)

            if pressed and idx == len(note_roll) - 1:
                end = len(note_roll) * note_length
                new_note = Note(velocity=100, pitch=note_number,
                                start=start, end=end)
                notes.append(new_note)

            prev_pressed = pressed

        return notes

    def _get_note_number(self, idx):
        return self.base_note_number + idx


def plot_piano_roll(piano_roll, title="Piano roll"):
    """
    Function for plotting a piano roll.
    """
    notes = ["C", "C#", "D", "C#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    plt.pcolormesh(piano_roll)
    plt.yticks(np.arange(12)+0.5, notes)
    plt.ylabel("Note")
    plt.xlabel("Bar")
    plt.title(title)
    plt.show()


def predict_piano_roll(model, env):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        score += reward

    return obs.reshape((env.n_notes, env.n_bars)), score
