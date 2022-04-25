import numpy as np
import librosa
from typing import Union
from enum import Enum
from numpy.lib.stride_tricks import sliding_window_view
from gym import Env
from gym.spaces import Box, MultiDiscrete, Discrete
from sklearn.preprocessing import normalize
from rl_instruments.utils.ks import MelodyData, make_melody


# Frequency settings
DEFAULT_FREQUENCY: float = 110.0
MIN_FREQUENCY: float = 82.0  # Two octaves, E2 to E4
MAX_FREQUENCY: float = 330.0
N_FREQUENCIES: int = int(MAX_FREQUENCY-MIN_FREQUENCY) + 1

# Pluck position settings
DEFAULT_PLUCK_POSITION: float = 0.5
MIN_PLUCK_POSITION: float = 0.0
MAX_PLUCK_POSITION: float = 0.5
N_PLUCK_POSITIONS: int = 3

# Loss factor settings
DEFAULT_LOSS_FACTOR: float = 0.996
MIN_LOSS_FACTOR: float = 0.93
MAX_LOSS_FACTOR: float = 0.996
N_LOSS_FACTORS: int = 2

# Amplitude settings
DEFAULT_AMPLITUDE: float = 1.0
MIN_AMPLITUDE: float = 1.0
MAX_AMPLITUDE: float = 2.0
N_AMPLITUDES: int = 2


class ControlableParameter(Enum):
    FREQUENCY = 'Frequency', DEFAULT_FREQUENCY, MIN_FREQUENCY, MAX_FREQUENCY, N_FREQUENCIES
    PLUCK_POSITION = 'Pluck position', DEFAULT_PLUCK_POSITION, MIN_PLUCK_POSITION, MAX_PLUCK_POSITION, N_PLUCK_POSITIONS
    LOSS_FACTOR = 'Loss factor', DEFAULT_LOSS_FACTOR, MIN_LOSS_FACTOR, MAX_LOSS_FACTOR, N_LOSS_FACTORS
    AMPLITUDE = 'Amplitude', DEFAULT_AMPLITUDE, MIN_AMPLITUDE, MAX_AMPLITUDE, N_AMPLITUDES

    def __init__(self, name: str, default_value: float = None, min_value: float = None, max_value: float = None, n: int = None):
        self._name_ = name
        self._default_value_ = default_value
        self._min_value_ = min_value
        self._max_value_ = max_value
        self._n_ = n

    @property
    def name(self):
        return self._name_

    @property
    def default_value(self):
        return self._default_value_

    @property
    def min_value(self):
        return self._min_value_

    @property
    def max_value(self):
        return self._max_value_

    @property
    def n(self):
        return self._n_


class KSEnv(Env):

    def __init__(self, target_melody: MelodyData) -> None:
        self.target_audio = target_melody.audio
        self.n_samples = target_melody.audio.size
        self.sr = target_melody.sr
        self.n_notes = target_melody.n_notes
        self.bpm = target_melody.bpm
        self.note_value = target_melody.note_value

        self.spm = int(self.sr*(4*self.note_value) *
                       (60/self.bpm))  # Samples per note

        self.target_stfts = [self._get_stft(
            self.target_audio[m*self.spm:(m+1)*self.spm]) for m in range(self.n_notes)]

        self.target_envelopes = [self._get_envelope(
            self.target_audio[m*self.spm:(m+1)*self.spm]) for m in range(self.n_notes)]

        low = (np.ones((self.n_notes, 1)) @
               np.array([MIN_FREQUENCY, MIN_PLUCK_POSITION, MIN_LOSS_FACTOR, MIN_AMPLITUDE]).reshape(1, -1)).astype(np.float32)
        high = (np.ones((self.n_notes, 1)) @
                np.array([MAX_FREQUENCY, MAX_PLUCK_POSITION, MAX_LOSS_FACTOR, MAX_AMPLITUDE]).reshape(1, -1)).astype(np.float32)

        self.observation_space = Box(low=low, high=high)

        self.reset()

    def step(self, action: Union[int, 'list[int]']) -> 'tuple[np.ndarray,float,bool,dict]':

        freq, pluck_position, loss_factor, amplitude = self._get_parameters(
            action)

        self.state[self.current_note, :] = [
            freq, pluck_position, loss_factor, amplitude]

        audio = make_melody(
            self.state[:self.current_note+1, 0],
            self.state[:self.current_note+1, 1],
            self.state[:self.current_note+1, 2],
            self.state[:self.current_note+1, 3],
            self.bpm,
            self.sr,
            self.note_value)

        note_audio = audio[self.current_note *
                           self.spm:(self.current_note+1)*self.spm]

        reward = self._get_reward(note_audio)

        self.current_note += 1

        done = self.current_note >= self.n_notes

        info = {}

        return self.state, reward, done, info

    def render(self, mode: str = "human") -> None:
        print(self.state)

    def reset(self) -> np.ndarray:
        self.state = np.zeros(self.observation_space.shape)
        self.current_note = 0
        return self.state

    def _get_reward(self, audio: np.ndarray) -> float:
        frequency_reward = self._get_frequency_reward(audio)
        envelope_reward = self._get_envelope_reward(audio)
        return (frequency_reward + envelope_reward)/2

    def _get_frequency_reward(self, audio: np.ndarray) -> float:
        stft = self._get_stft(audio)
        reward_array = self.target_stfts[self.current_note] * stft
        return np.sum(np.sqrt(np.sum(reward_array, axis=0)))/reward_array.shape[1]

    def _get_envelope_reward(self, audio: np.ndarray) -> float:
        envelope = self._get_envelope(audio)
        mse = np.sum(((envelope - self.target_envelopes[self.current_note]) /
                     MAX_AMPLITUDE)**2)/envelope.size
        return 1 - mse

    def _get_envelope(self, audio: np.ndarray) -> np.ndarray:
        return sliding_window_view(audio, 100)[::10, :].max(axis=1)

    def _get_stft(self, audio: np.ndarray) -> np.ndarray:
        return normalize(
            np.abs(librosa.stft(
                audio, n_fft=1024, win_length=1024, hop_length=1024)),
            axis=0)


class KSSingleParamEnv(KSEnv):
    def __init__(self, target_melody: MelodyData, controlable_parameter: ControlableParameter) -> None:

        super().__init__(target_melody)

        self.controlable_parameter = controlable_parameter
        self._define_action_space(controlable_parameter)

    def _define_action_space(self, controlable_parameter: ControlableParameter) -> None:
        if controlable_parameter is ControlableParameter.FREQUENCY:
            self.action_space = Discrete(N_FREQUENCIES)

        elif controlable_parameter is ControlableParameter.PLUCK_POSITION:
            self.action_space = Discrete(N_PLUCK_POSITIONS)

        elif controlable_parameter is ControlableParameter.LOSS_FACTOR:
            self.action_space = Discrete(N_LOSS_FACTORS)

        elif controlable_parameter is ControlableParameter.AMPLITUDE:
            self.action_space = Discrete(N_AMPLITUDES)

    def _get_parameters(self, action: int) -> 'tuple[float, float, float, float]':

        freq, pluck_position, loss_factor, amplitude = DEFAULT_FREQUENCY, DEFAULT_PLUCK_POSITION, DEFAULT_LOSS_FACTOR, DEFAULT_AMPLITUDE

        controlable_parameter_value = self.controlable_parameter.min_value + action * \
            (self.controlable_parameter.max_value -
             self.controlable_parameter.min_value)/(self.controlable_parameter.n-1)

        if self.controlable_parameter is ControlableParameter.FREQUENCY:
            freq = controlable_parameter_value

        elif self.controlable_parameter is ControlableParameter.PLUCK_POSITION:
            pluck_position = controlable_parameter_value

        elif self.controlable_parameter is ControlableParameter.LOSS_FACTOR:
            loss_factor = controlable_parameter_value

        elif self.controlable_parameter is ControlableParameter.AMPLITUDE:
            amplitude = controlable_parameter_value

        return freq, pluck_position, loss_factor, amplitude


class KSMultiParamEnv(KSEnv):
    def __init__(self, target_melody: MelodyData, controlable_parameters: 'set[ControlableParameter]') -> None:

        if len(controlable_parameters) == 0:
            raise ValueError

        super().__init__(target_melody)

        self.controlable_parameters = controlable_parameters
        self._define_action_space(controlable_parameters)

    def _define_action_space(self, controlable_parameters: str) -> None:

        action_space_array = []

        if ControlableParameter.FREQUENCY in controlable_parameters:
            action_space_array.append(N_FREQUENCIES)

        if ControlableParameter.PLUCK_POSITION in controlable_parameters:
            action_space_array.append(N_PLUCK_POSITIONS)

        if ControlableParameter.LOSS_FACTOR in controlable_parameters:
            action_space_array.append(N_LOSS_FACTORS)

        if ControlableParameter.AMPLITUDE in controlable_parameters:
            action_space_array.append(N_AMPLITUDES)

        self.action_space = MultiDiscrete(action_space_array)

    def _get_parameters(self, action: 'list[int]') -> 'tuple[float, float, float, float]':

        freq, pluck_position, loss_factor, amplitude = DEFAULT_FREQUENCY, DEFAULT_PLUCK_POSITION, DEFAULT_LOSS_FACTOR, DEFAULT_AMPLITUDE

        action_idx = 0

        if ControlableParameter.FREQUENCY in self.controlable_parameters:
            freq = MIN_FREQUENCY + action[action_idx] * \
                (MAX_FREQUENCY - MIN_FREQUENCY)/(N_FREQUENCIES - 1)
            action_idx += 1

        if ControlableParameter.PLUCK_POSITION in self.controlable_parameters:
            pluck_position = MIN_PLUCK_POSITION + action[action_idx] * \
                (MAX_PLUCK_POSITION - MIN_PLUCK_POSITION)/(N_PLUCK_POSITIONS - 1)
            action_idx += 1

        if ControlableParameter.LOSS_FACTOR in self.controlable_parameters:
            loss_factor = MIN_LOSS_FACTOR + action[action_idx] * \
                (MAX_LOSS_FACTOR - MIN_LOSS_FACTOR)/(N_LOSS_FACTORS - 1)
            action_idx += 1

        if ControlableParameter.AMPLITUDE in self.controlable_parameters:
            amplitude = MIN_AMPLITUDE + action[action_idx] * \
                (MAX_AMPLITUDE - MIN_AMPLITUDE)/(N_AMPLITUDES - 1)
            action_idx += 1

        return freq, pluck_position, loss_factor, amplitude
