from gym import Env
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from pretty_midi import PrettyMIDI, Instrument, Note
import librosa
from rl_instruments.models import WrappedModel


class PianoRollManager():

    def __init__(self, piano_roll: np.ndarray, bpm: int, note_value: float, sr: int = 8092, base_note_number: int = 60) -> None:
        self.piano_roll = piano_roll
        self.bpm = bpm
        self.sr = sr
        self.base_note_number = base_note_number
        self.note_value = note_value

        self.midi = self._create_midi_from_piano_roll()
        self.audio = self._synthesize()
        self.stft = normalize(
            np.abs(librosa.stft(
                self.audio, n_fft=1024, win_length=1024, hop_length=1024)),
            axis=0)

    def get_stft(self) -> np.ndarray:
        return self.stft

    def get_audio(self) -> np.ndarray:
        return self.audio

    def _synthesize(self) -> np.ndarray:
        audio = self.midi.synthesize(self.sr) * (self.piano_roll.max()/100)
        note_length = (60/self.bpm)*(4*self.note_value)
        n_samples = int(self.sr * note_length)*self.piano_roll.shape[1]
        return audio[:n_samples]

    def _create_midi_from_piano_roll(self) -> PrettyMIDI:

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

    def _get_notes(self, note_number: int, note_roll: np.ndarray) -> 'list[Note]':
        note_length = (60/self.bpm)*(4*self.note_value)
        notes = []

        for idx, velocity in enumerate(note_roll):

            if (velocity != 0):
                start = idx * note_length
                end = (idx+1) * note_length
                new_note = Note(velocity=velocity, pitch=note_number,
                                start=start, end=end)
                notes.append(new_note)

        return notes

    def _get_note_number(self, idx: int) -> int:
        return self.base_note_number + idx


def plot_piano_roll(piano_roll: np.ndarray, title: str = "Piano roll") -> None:
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


def predict_piano_roll(model: WrappedModel, env: Env) -> 'tuple[np.ndarray,list[float]]':
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)

    return obs.reshape((env.n_keys, env.n_notes)), rewards
