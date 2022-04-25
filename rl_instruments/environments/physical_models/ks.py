import numpy as np
import librosa
from typing import Union
from enum import Enum
from numpy.lib.stride_tricks import sliding_window_view
from gym import Env
from gym.spaces import Box, MultiDiscrete, Discrete
from sklearn.preprocessing import normalize
from rl_instruments.utils.ks import MelodyData, make_melody


# Two octaves, E2 to E4
MIN_FREQ = 82
MAX_FREQ = 330

# Maximum possible amplitude of a synthesized waveform
MAX_AMP = 2.0

# Default values for uncontrolable parameters


class DefaultParameterValue(Enum):
    FREQ = 110
    PLUCK_POSITION = 0.5
    LOSS_FACTOR = 0.996
    AMP = 1.0


class ControlableParameter(Enum):
    FREQUENCY = 'Frequency', DefaultParameterValue.FREQ
    PLUCK_POSITION = 'Pluck position', DefaultParameterValue.PLUCK_POSITION
    LOSS_FACTOR = 'Loss factor', DefaultParameterValue.LOSS_FACTOR
    AMPLITUDE = 'Amplitude', DefaultParameterValue.AMP

    def __init__(self, _: str, default: DefaultParameterValue = None):
        self._default_ = default.value

    @property
    def default(self):
        return self._default_


class KSEnv(Env):

    def __init__(self, target_melody: MelodyData) -> None:
        self.target_audio = target_melody.audio
        self.n_samples = target_melody.audio.size
        self.sr = target_melody.sr
        self.n_measures = target_melody.n_measures
        self.bpm = target_melody.bpm
        self.granularity_factor = target_melody.granularity_factor

        self.spm = int(self.sr*(4/self.granularity_factor) *
                       (60/self.bpm))  # Samples per measure

        self.target_stfts = [self._get_stft(
            self.target_audio[m*self.spm:(m+1)*self.spm]) for m in range(self.n_measures)]

        self.target_envelopes = [self._get_envelope(
            self.target_audio[m*self.spm:(m+1)*self.spm]) for m in range(self.n_measures)]

        self.freq_range = MAX_FREQ - MIN_FREQ

        low = (np.ones((self.n_measures, 1)) @
               np.array([MIN_FREQ, 0.0, 0.96, 1.0]).reshape(1, -1)).astype(np.float32)
        high = (np.ones((self.n_measures, 1)) @
                np.array([MAX_FREQ, 0.5, 0.996, MAX_AMP]).reshape(1, -1)).astype(np.float32)

        self.observation_space = Box(low=low, high=high)

        self.reset()

    def step(self, action: Union[int, 'list[int]']) -> 'tuple[np.ndarray,float,bool,dict]':

        freq, pluck_position, loss_factor, amplitude = self._get_parameters(
            action)

        self.state[self.current_measure, :] = [
            freq, pluck_position, loss_factor, amplitude]

        audio = make_melody(
            self.state[:self.current_measure+1, 0],
            self.state[:self.current_measure+1, 1],
            self.state[:self.current_measure+1, 2],
            self.state[:self.current_measure+1, 3],
            self.bpm,
            self.sr,
            self.granularity_factor)

        measure_audio = audio[self.current_measure *
                              self.spm:(self.current_measure+1)*self.spm]

        reward = self._get_reward(measure_audio)

        self.current_measure += 1

        done = self.current_measure >= self.n_measures

        info = {}

        return self.state, reward, done, info

    def render(self, mode: str = "human") -> None:
        print(self.state)

    def reset(self) -> np.ndarray:
        self.state = np.zeros(self.observation_space.shape)
        self.current_measure = 0
        return self.state

    def _get_reward(self, audio: np.ndarray) -> float:
        frequency_reward = self._get_frequency_reward(audio)
        envelope_reward = self._get_envelope_reward(audio)
        return (frequency_reward + envelope_reward)/2

    def _get_frequency_reward(self, audio: np.ndarray) -> float:
        stft = self._get_stft(audio)
        reward_array = self.target_stfts[self.current_measure] * stft
        return np.sum(np.sqrt(np.sum(reward_array, axis=0)))/reward_array.shape[1]

    def _get_envelope_reward(self, audio: np.ndarray) -> float:
        envelope = self._get_envelope(audio)
        mse = np.sum(((envelope - self.target_envelopes[self.current_measure]) /
                     MAX_AMP)**2)/envelope.size
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
            self.action_space = Discrete(self.freq_range)

        elif controlable_parameter is ControlableParameter.PLUCK_POSITION:
            self.action_space = Discrete(3)

        elif controlable_parameter is ControlableParameter.LOSS_FACTOR:
            self.action_space = Discrete(2)

        elif controlable_parameter is ControlableParameter.AMPLITUDE:
            self.action_space = Discrete(2)

    def _get_parameters(self, action: int) -> 'tuple[int, float, float, float]':

        freq, pluck_position, loss_factor, amplitude = DefaultParameterValue.FREQ.value, DefaultParameterValue.PLUCK_POSITION.value, DefaultParameterValue.LOSS_FACTOR.value, DefaultParameterValue.AMP.value

        if self.controlable_parameter is ControlableParameter.FREQUENCY:
            freq = MIN_FREQ + action

        elif self.controlable_parameter is ControlableParameter.PLUCK_POSITION:
            pluck_position = action*0.25

        elif self.controlable_parameter is ControlableParameter.LOSS_FACTOR:
            loss_factor = 0.996 - 0.036 * action

        elif self.controlable_parameter is ControlableParameter.AMPLITUDE:
            amplitude = (MAX_AMP - 1) * action + 1

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
            action_space_array.append(self.freq_range)

        if ControlableParameter.PLUCK_POSITION in controlable_parameters:
            action_space_array.append(3)

        if ControlableParameter.LOSS_FACTOR in controlable_parameters:
            action_space_array.append(2)

        if ControlableParameter.AMPLITUDE in controlable_parameters:
            action_space_array.append(2)

        self.action_space = MultiDiscrete(action_space_array)

    def _get_parameters(self, action: 'list[int]') -> 'tuple[int, float, float, float]':

        freq, pluck_position, loss_factor, amplitude = DefaultParameterValue.FREQ.value, DefaultParameterValue.PLUCK_POSITION.value, DefaultParameterValue.LOSS_FACTOR.value, DefaultParameterValue.AMP.value

        action_idx = 0

        if ControlableParameter.FREQUENCY in self.controlable_parameters:
            freq = MIN_FREQ + action[action_idx]
            action_idx += 1

        if ControlableParameter.PLUCK_POSITION in self.controlable_parameters:
            pluck_position = action[action_idx]*0.25
            action_idx += 1

        if ControlableParameter.LOSS_FACTOR in self.controlable_parameters:
            loss_factor = 0.996 - 0.036 * action[action_idx]
            action_idx += 1

        if ControlableParameter.AMPLITUDE in self.controlable_parameters:
            amplitude = (MAX_AMP - 1) * action[action_idx] + 1
            action_idx += 1

        return freq, pluck_position, loss_factor, amplitude
