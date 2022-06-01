import random
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from gym import Env

# Modified version of https://inst.eecs.berkeley.edu/~cs61a/fa12/labs/lab05/guitar.py


def create_pluck(n: int, pluck_position: float = 0.5, amplitude: float = 1.0) -> List[float]:
    """Create a list of n random values between -0.5 and 0.5 representing
    initial string excitation (white noise). Apply comb filter to simulate pluck position.
    """
    # BEGIN SOLUTION
    state = random.getstate()

    random.seed(2)
    noise = [amplitude*(random.random()-0.5) for _ in range(n)]

    random.setstate(state)

    sample_pos = int(pluck_position*n)

    for i in range(sample_pos, n):
        noise[i] -= noise[i-sample_pos]

    return noise
    # END SOLUTION


def apply_ks(s: int, n: int, loss_factor: float = 0.996) -> np.ndarray:
    """Apply n Karplus-Strong updates to the list s and return the result,
    using the initial length of s as the frequency.

    >>> s = [0.2, 0.4, 0.5]
    >>> apply_ks(s, 4)
    [0.2, 0.4, 0.5, 0.29880000000000007, 0.4482, 0.39780240000000006, 0.37200600000000006]
    """
    # BEGIN SOLUTION
    for t in range(n):
        s.append(loss_factor * (s[t] + s[t+1])/2)
    return np.array(s)
    # END SOLUTION


def synthesize_string(frequency: int,
                      pluck_position: float = 0.5,
                      loss_factor: float = 0.996,
                      amplitude: float = 1.0,
                      n_samples: int = 30000,
                      sample_rate: int = 44100) -> np.ndarray:
    """Return a list of num_samples samples synthesizing a guitar string."""

    delay = int(sample_rate / frequency)

    pluck = create_pluck(delay, pluck_position, amplitude)

    samples = apply_ks(pluck, n_samples, loss_factor)

    return samples[:n_samples]


def make_melody(freqs: List[int],
                pluck_positions: List[float],
                loss_factors: List[float],
                amplitudes: List[float],
                bpm: int,
                sample_rate: int,
                note_value: float) -> np.ndarray:

    # Note length in seconds
    note_length = (60/bpm)*(4*note_value)

    # Note samples
    n_samples = int(sample_rate*note_length)

    return np.concatenate([synthesize_string(freq, pluck_position, loss_factor, amp, n_samples, sample_rate)
                           for freq, pluck_position, loss_factor, amp in zip(freqs, pluck_positions, loss_factors, amplitudes)])


def predict_melody(env: Env, model) -> Tuple[np.ndarray, List[float], np.ndarray]:
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    predicted_audio = make_melody(
        obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], env.bpm, env.sample_rate, env.note_value)
    return predicted_audio, rewards, obs


@dataclass
class MelodyData:
    audio: np.ndarray
    sample_rate: int
    bpm: int
    n_notes: int
    note_value: int
