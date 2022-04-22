import numpy as np
import random
from dataclasses import dataclass
from rl_instruments.models import WrappedModel
from rl_instruments.environments.physical_models import KSEnv


@dataclass
class GuitarMelody:
    audio: np.ndarray
    sr: int
    bpm: int
    n_measures: int
    granularity_factor: int

# Modified version of https://inst.eecs.berkeley.edu/~cs61a/fa12/labs/lab05/guitar.py


"""A Python implementation of the Karplus-Strong algorithm"""


def create_pluck(n: int, pluck_position: float = 0.5, amplitude: float = 1.0) -> 'list[float]':
    """Create a list of n random values between -0.5 and 0.5 representing
    initial string excitation (white noise). Apply comb filter to simulate pluck position.
    """
    # BEGIN SOLUTION
    random.seed(2)

    noise = [amplitude*(random.random()-0.5) for _ in range(n)]

    sample_pos = int(pluck_position*n)

    for i in range(sample_pos, n):
        noise[i] -= noise[i-sample_pos]

    return noise
    # END SOLUTION


def apply_ks(s: int, n: int, decay_factor: float = 0.996) -> np.ndarray:
    """Apply n Karplus-Strong updates to the list s and return the result,
    using the initial length of s as the frequency.

    >>> s = [0.2, 0.4, 0.5]
    >>> apply_ks(s, 4)
    [0.2, 0.4, 0.5, 0.29880000000000007, 0.4482, 0.39780240000000006, 0.37200600000000006]
    """
    # BEGIN SOLUTION
    for t in range(n):
        s.append(decay_factor * (s[t] + s[t+1])/2)
    return np.array(s)
    # END SOLUTION


def guitar_string(frequency: int,
                  pluck_position: float = 0.5,
                  decay_factor: float = 0.996,
                  amplitude: float = 1.0,
                  n_samples: int = 30000,
                  sample_rate: int = 44100) -> np.ndarray:
    """Return a list of num_samples samples synthesizing a guitar string."""

    delay = int(sample_rate / frequency)

    pluck = create_pluck(delay, pluck_position, amplitude)

    samples = apply_ks(pluck, n_samples, decay_factor)

    return samples[:n_samples]


def make_melody(freqs: 'list[int]',
                pluck_positions: 'list[float]',
                decay_factors: 'list[float]',
                amplitudes: 'list[float]',
                bpm: int,
                sr: int,
                granularity_factor: int) -> np.ndarray:

    t = (60/bpm)*(4/granularity_factor)
    n_samples = int(sr*t)

    return np.concatenate([guitar_string(freq, pluck_position, decay_factor, amp, n_samples, sr)
                           for freq, pluck_position, decay_factor, amp in zip(freqs, pluck_positions, decay_factors, amplitudes)])


def predict_melody(env: KSEnv, model: WrappedModel) -> 'tuple[np.ndarray,float]':
    obs = env.reset()
    done = False
    score = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        score += reward
    predicted_audio = make_melody(
        obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], env.bpm, env.sr, env.granularity_factor)
    return predicted_audio, score
