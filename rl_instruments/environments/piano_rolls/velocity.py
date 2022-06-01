from typing import Tuple
import numpy as np
import librosa
from numpy.lib.stride_tricks import sliding_window_view
from gym import Env
from gym.spaces import MultiDiscrete, MultiBinary
from sklearn.preprocessing import normalize
from rl_instruments.utils.piano import plot_piano_roll, PianoRollManager


class VelocitytEnv(Env):
    """
    Environment defining a piano roll. Only one note may be pressed at the same time.

    :param 2D np.array piano_roll: The ground truth that an agent will try to learn.
    """

    MAX_AMPLITUDE: float = 1.0

    def __init__(self, audio: np.ndarray,
                 sample_rate: int,
                 bpm: int,
                 note_value: float,
                 n_notes: int,
                 n_keys: int = 12,
                 n_velocities: int = 2) -> None:
        self.target_audio = audio
        self.sample_rate = sample_rate
        self.bpm = bpm
        self.note_value = note_value
        self.n_keys = n_keys
        self.n_notes = n_notes
        self.n_velocities = n_velocities

        self.spn = int(self.sample_rate*(4*self.note_value) *
                       (60/self.bpm))  # Samples per note

        self.target_stfts = [self._get_stft(
            self.target_audio[m*self.spn:(m+1)*self.spn]) for m in range(self.n_notes)]

        self.target_envelopes = [self._get_envelope(
            self.target_audio[m*self.spn:(m+1)*self.spn]) for m in range(self.n_notes)]

        # The possible actions, i.e. which key to press for the current note.
        self.action_space = MultiDiscrete([self.n_keys, self.n_velocities])

        self.observation_space = MultiBinary(
            n=np.product([self.n_keys, self.n_notes]))

        self.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:

        # Update state of learnt roll.
        pressed_key, velocity_boost = action

        self.state[pressed_key, self.current_note] = 50 * \
            (1 + velocity_boost/(self.n_velocities-1))

        midi_data = PianoRollManager(self.state[:, :self.current_note + 1],
                                     bpm=self.bpm,
                                     note_value=self.note_value,
                                     sample_rate=self.sample_rate)

        note_audio = midi_data.get_audio()[self.current_note *
                                           self.spn:(self.current_note+1)*self.spn]

        reward = self._get_reward(note_audio)

        self.current_note += 1

        # We are done if all bars have been written.
        done = self.current_note >= self.n_notes

        return self.state.flatten(), reward, done, self.info

    def render(self, mode="human") -> np.ndarray:
        plot_piano_roll(self.state)
        return self.state

    def reset(self) -> np.ndarray:
        self.current_note = 0
        self.state = np.zeros((self.n_keys, self.n_notes), dtype=np.int8)
        self.info = {
            "frequency_reward": 0.0,
            "envelope_reward": 0.0
        }
        return self.state.flatten()

    def _get_reward(self, audio: np.ndarray) -> float:
        frequency_reward = self._get_frequency_reward(audio)
        envelope_reward = self._get_envelope_reward(audio)

        self.info["frequency_reward"] += frequency_reward
        self.info["envelope_reward"] += envelope_reward

        return (frequency_reward + envelope_reward)/2

    def _get_frequency_reward(self, audio: np.ndarray) -> float:
        stft = self._get_stft(audio)
        reward_array = self.target_stfts[self.current_note] * stft
        return np.sum(np.sqrt(np.sum(reward_array, axis=0)))/reward_array.shape[1]

    def _get_envelope_reward(self, audio: np.ndarray) -> float:
        envelope = self._get_envelope(audio)
        avg_dev = np.sum(np.abs((envelope - self.target_envelopes[self.current_note]) /
                                self.MAX_AMPLITUDE))/envelope.size
        return 1 - avg_dev

    def _get_stft(self, audio: np.ndarray) -> np.ndarray:
        return normalize(
            np.abs(librosa.stft(
                audio, n_fft=1024, win_length=1024, hop_length=1024)),
            axis=0)

    def _get_envelope(self, audio: np.ndarray) -> np.ndarray:
        return sliding_window_view(audio, 100)[::10, :].max(axis=1)
