from abc import abstractmethod
from typing import Union
from enum import Enum
import numpy as np
import librosa
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
MIN_PLUCK_POSITION: float = 0.1
MAX_PLUCK_POSITION: float = 0.5
N_PLUCK_POSITIONS: int = 3

# Loss factor settings
DEFAULT_LOSS_FACTOR: float = 0.996
MIN_LOSS_FACTOR: float = 0.91
MAX_LOSS_FACTOR: float = 0.996
N_LOSS_FACTORS: int = 2

# Amplitude settings
DEFAULT_AMPLITUDE: float = 0.5
MIN_AMPLITUDE: float = 0.5
MAX_AMPLITUDE: float = 1.0
N_AMPLITUDES: int = 2


class ControlableParameter(Enum):
    FREQUENCY = 0, DEFAULT_FREQUENCY, MIN_FREQUENCY, MAX_FREQUENCY, N_FREQUENCIES
    PLUCK_POSITION = 1, DEFAULT_PLUCK_POSITION, MIN_PLUCK_POSITION, MAX_PLUCK_POSITION, N_PLUCK_POSITIONS
    LOSS_FACTOR = 2,  DEFAULT_LOSS_FACTOR, MIN_LOSS_FACTOR, MAX_LOSS_FACTOR, N_LOSS_FACTORS
    AMPLITUDE = 3, DEFAULT_AMPLITUDE, MIN_AMPLITUDE, MAX_AMPLITUDE, N_AMPLITUDES

    def __init__(self,
                 parameter_order: int,
                 default_value: float = None,
                 min_value: float = None,
                 max_value: float = None,
                 n: int = None):
        self._parameter_order_ = parameter_order
        self._default_value_ = default_value
        self._min_value_ = min_value
        self._max_value_ = max_value
        self._n_ = n

    @property
    def parameter_order(self):
        return self._parameter_order_

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

    def __lt__(self, other):
        return self.parameter_order < other.parameter_order

    def __gt__(self, other):
        return self.parameter_order > other.parameter_order


class KSEnv(Env):

    def __init__(self, target_melody: MelodyData) -> None:
        self.target_audio = target_melody.audio
        self.n_samples = target_melody.audio.size
        self.sr = target_melody.sr
        self.n_notes = target_melody.n_notes
        self.bpm = target_melody.bpm
        self.note_value = target_melody.note_value

        self.spn = int(self.sr*(4*self.note_value) *
                       (60/self.bpm))  # Samples per note

        self.target_stfts = [self._get_stft(
            self.target_audio[m*self.spn:(m+1)*self.spn]) for m in range(self.n_notes)]

        self.target_envelopes = [self._get_envelope(
            self.target_audio[m*self.spn:(m+1)*self.spn]) for m in range(self.n_notes)]

        low = (np.ones((self.n_notes, 1)) @
               np.array([MIN_FREQUENCY, MIN_PLUCK_POSITION,
                        MIN_LOSS_FACTOR, MIN_AMPLITUDE])
               .reshape(1, -1)).astype(np.float32)
        high = (np.ones((self.n_notes, 1)) @
                np.array([MAX_FREQUENCY, MAX_PLUCK_POSITION,
                         MAX_LOSS_FACTOR, MAX_AMPLITUDE])
                .reshape(1, -1)).astype(np.float32)

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
                           self.spn:(self.current_note+1)*self.spn]

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
        avg_dev = np.sum(np.abs((envelope - self.target_envelopes[self.current_note]) /
                                MAX_AMPLITUDE))/envelope.size
        return 1 - avg_dev

    @abstractmethod
    def _get_parameters(self, action):
        pass

    def _get_envelope(self, audio: np.ndarray) -> np.ndarray:
        return sliding_window_view(audio, 100)[::10, :].max(axis=1)

    def _get_stft(self, audio: np.ndarray) -> np.ndarray:
        return normalize(
            np.abs(librosa.stft(
                audio, n_fft=1024, win_length=1024, hop_length=1024)),
            axis=0)

    def _get_controlable_parameter_value(self,
                                         action: int,
                                         controlable_parameter: ControlableParameter):

        value = controlable_parameter.min_value + action * \
            (controlable_parameter.max_value -
             controlable_parameter.min_value)/(controlable_parameter.n-1)

        return value


class KSSingleParamEnv(KSEnv):
    def __init__(self,
                 target_melody: MelodyData,
                 controlable_parameter: ControlableParameter) -> None:

        super().__init__(target_melody)

        self.controlable_parameter = controlable_parameter
        self._define_action_space(controlable_parameter)

    def _define_action_space(self, controlable_parameter: ControlableParameter) -> None:

        self.action_space = Discrete(controlable_parameter.n)

    def _get_parameters(self, action: int) -> 'tuple[float, float, float, float]':

        controlable_parameter_value = self._get_controlable_parameter_value(
            action, self.controlable_parameter)

        return [controlable_parameter_value if cp is self.controlable_parameter
                else cp.default_value for cp in sorted(ControlableParameter)]


class KSMultiParamEnv(KSEnv):
    def __init__(self,
                 target_melody: MelodyData,
                 controlable_parameters: 'set[ControlableParameter]') -> None:

        if len(controlable_parameters) == 0:
            raise ValueError

        super().__init__(target_melody)

        self.controlable_parameters = controlable_parameters
        self._define_action_space(controlable_parameters)

    def _define_action_space(self, controlable_parameters: 'list[ControlableParameter]') -> None:

        action_space_array = [cp.n for cp in sorted(controlable_parameters)]

        self.action_space = MultiDiscrete(action_space_array)

    def _get_parameters(self, action: 'list[int]') -> 'tuple[float, float, float, float]':

        parameters = []

        action_idx = 0

        for cp in sorted(ControlableParameter):
            if cp in self.controlable_parameters:
                param = self._get_controlable_parameter_value(
                    action[action_idx], cp)
                action_idx += 1
            else:
                param = cp.default_value

            parameters.append(param)

        return parameters
