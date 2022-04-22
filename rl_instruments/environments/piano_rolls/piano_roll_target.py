import numpy as np
from gym.spaces import Discrete, MultiBinary
from gym import Env


class PianoRollTargetEnv(Env):
    """
    Environment defining a piano roll. Only one note per bar is allowed in this setting.

    :param 2D np.array piano_roll: The ground truth that an agent will try to learn.
    """

    log_dir = "saved_models/piano_roll_target/"

    def __init__(self, piano_roll):
        self.target = piano_roll
        self.n_notes, self.n_bars = piano_roll.shape

        # The possible actions, i.e. press one key.
        self.action_space = Discrete(self.n_notes)

        self.observation_space = MultiBinary(
            n=np.product([self.n_notes, self.n_bars]))

        self.current_bar = 0

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

        # Update state of written piano roll.
        self.state[action, self.current_bar] = 1

        # Correctly pressed key gives a reward of one, incorrect penalty of -1.
        if self.target[action, self.current_bar] == 1:
            reward = 1
        else:
            reward = -1

        self._next_bar()

        # We are done if all bars have been written.
        done = self.current_bar >= self.n_bars

        info = {}

        return self.state.flatten(), reward, done, info

    def render(self):
        pass

    def reset(self):
        self.state = np.zeros((self.n_notes, self.n_bars), dtype="int8")
        self.current_bar = 0
        return self.state.flatten()
