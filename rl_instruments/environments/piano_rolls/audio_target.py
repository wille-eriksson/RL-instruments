import numpy as np
import matplotlib.pyplot as plt
import copy
import librosa
from gym import Env
from gym.spaces import Discrete, MultiBinary
from sklearn.preprocessing import normalize
from rl_instruments.utils import PianoRollManager


class AudioTargetEnv(Env):
    """
    Environment defining a piano roll. Only one note may be pressed at the same time.

    :param 2D np.array piano_roll: The ground truth that an agent will try to learn.
    """

    log_dir = "saved_models/audio_target/"

    granularity_dict = {"whole": 1,
                        "half": 2,
                        "quarter": 4,
                        "eight": 8,
                        "sixteenth": 16,
                        "thirty-second": 32}

    def __init__(self, audio, sr, bpm, note_granularity, n_bars, n_notes=12, base_note_number=60):
        self.sr = sr
        self.bpm = bpm
        self.note_granularity = note_granularity
        self.n_notes = n_notes
        self.n_bars = n_bars

        # Set ground truth to be normalized chromagram
        self.ground_truth = self._get_stft(audio)

        # The possible actions, i.e. which keys to press the current bar.
        self.action_space = Discrete(self.n_notes)

        self.observation_space = MultiBinary(
            n=np.product([self.n_notes, self.n_bars]))

        self.current_bar = 0
        self.prev_last_window = 0

        # Initialize all entries in piano roll to 0.
        self.state = np.zeros((self.n_notes, self.n_bars), dtype="int8")

    def _next_bar(self):
        """
        Increases the current bar count by one and sets all keys in bar to 0.
        """
        self.current_bar += 1

        if self.current_bar < self.n_bars:
            self.state[:, self.current_bar] = 0

    def step(self, action):

        # Update state of learnt roll.
        self.state[action, self.current_bar] = 1

        midi_data = PianoRollManager(self.state[:, :self.current_bar + 1],
                                     bpm=self.bpm,
                                     note_granularity=self.note_granularity,
                                     sr=self.sr)

        audio = midi_data.get_audio()

        stft = self._get_stft(audio)

        reward = self._get_reward(stft)

        self._next_bar()

        # We are done if all bars have been written.
        done = self.current_bar >= self.n_bars

        info = {}

        return self.state.flatten(), reward, done, info

    def render(self, mode="human"):
        plot_roll = copy.deepcopy(self.state)
        plot_roll[plot_roll == 2] = 0
        plt.plot_piano_roll(self.state)
        return self.state

    def reset(self):
        self.state = np.zeros((self.n_notes, self.n_bars), dtype="int8")
        self.current_bar = 0
        self.prev_last_window = 0
        return self.state.flatten()

    def _get_reward(self, stft):
        start = self.prev_last_window
        end = stft.shape[1]
        self.prev_last_window = end
        reward_array = self.ground_truth[:, start:end] * stft[:, start:end]
        return np.sum(np.sqrt(np.sum(reward_array, axis=0)))/reward_array.shape[1]

    def _get_stft(self, audio):
        return normalize(
            np.abs(librosa.stft(
                audio, n_fft=1024, win_length=1024, hop_length=1024)),
            axis=0)
