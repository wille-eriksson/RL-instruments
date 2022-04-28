import numpy as np
import librosa
from gym import Env
from gym.spaces import Discrete, MultiBinary
from sklearn.preprocessing import normalize
from rl_instruments.utils.piano import plot_piano_roll, PianoRollManager


class AudioTargetEnv(Env):
    """
    Environment defining a piano roll. Only one note may be pressed at the same time.

    :param 2D np.array piano_roll: The ground truth that an agent will try to learn.
    """

    def __init__(self, audio: np.ndarray, sr: int, bpm: int, note_value: float, n_notes: int, n_keys: int = 12):
        self.target_audio = audio
        self.sr = sr
        self.bpm = bpm
        self.note_value = note_value
        self.n_keys = n_keys
        self.n_notes = n_notes

        self.spn = int(self.sr*(4*self.note_value) *
                       (60/self.bpm))  # Samples per note

        self.target_stfts = [self._get_stft(
            self.target_audio[m*self.spn:(m+1)*self.spn]) for m in range(self.n_notes)]

        # The possible actions, i.e. which key to press for the current note.
        self.action_space = Discrete(self.n_keys)

        self.observation_space = MultiBinary(
            n=np.product([self.n_keys, self.n_notes]))

        self.reset()

    def step(self, action: int) -> 'tuple[np.ndarray,float,bool,dict]':

        # Update state of learnt roll.
        self.state[action, self.current_note] = 100

        midi_data = PianoRollManager(self.state[:, :self.current_note + 1],
                                     bpm=self.bpm,
                                     note_value=self.note_value,
                                     sr=self.sr)

        note_audio = midi_data.get_audio()[self.current_note *
                                           self.spn:(self.current_note+1)*self.spn]

        reward = self._get_reward(note_audio)

        self.current_note += 1

        # We are done if all bars have been written.
        done = self.current_note >= self.n_notes

        info = {}

        return self.state.flatten(), reward, done, info

    def render(self, mode="human") -> np.ndarray:
        plot_piano_roll(self.state)
        return self.state

    def reset(self) -> np.ndarray:
        self.current_note = 0
        self.state = np.zeros((self.n_keys, self.n_notes), dtype=np.int8)
        return self.state.flatten()

    def _get_reward(self, audio: np.ndarray) -> float:
        stft = self._get_stft(audio)
        reward_array = self.target_stfts[self.current_note] * stft
        return np.sum(np.sqrt(np.sum(reward_array, axis=0)))/reward_array.shape[1]

    def _get_stft(self, audio: np.ndarray) -> float:
        return normalize(
            np.abs(librosa.stft(
                audio, n_fft=1024, win_length=1024, hop_length=1024)),
            axis=0)
