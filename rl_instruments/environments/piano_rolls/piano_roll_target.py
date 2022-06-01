from typing import Tuple
import numpy as np
from gym.spaces import Discrete, MultiBinary
from gym import Env


class PianoRollTargetEnv(Env):
    """
    Environment defining a piano roll. Only one note per bar is allowed in this setting.

    :param 2D np.array piano_roll: The ground truth that an agent will try to learn.
    """

    def __init__(self, piano_roll: np.ndarray) -> None:
        self.target = piano_roll
        self.n_keys, self.n_notes = piano_roll.shape

        # The possible actions, i.e. press one key.
        self.action_space = Discrete(self.n_keys)

        self.observation_space = MultiBinary(
            n=np.product([self.n_keys, self.n_notes]))

        self.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:

        # Update state of written piano roll.
        self.state[action, self.current_note] = 1

        # Correctly pressed key gives a reward of one, incorrect penalty of -1.
        reward = 1 if self.target[action, self.current_note] == 1 else 0

        self.current_note += 1

        # We are done if all bars have been written.
        done = self.current_note >= self.n_notes

        info = {}

        return self.state.flatten(), reward, done, info

    def render(self, mode="human") -> None:
        pass

    def reset(self) -> np.ndarray:
        self.state = np.zeros((self.n_keys, self.n_notes), dtype=np.int8)
        self.current_note = 0
        return self.state.flatten()
