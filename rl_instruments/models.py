import os
from os.path import exists
from typing import Union
from gym import Env
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


class SaveOnBestTrainingRewardCallback(BaseCallback):
  # https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


class WrappedModel:
    def __init__(self, algorithm: Union[PPO, DQN], env: Env, log_dir: str, policy="MlpPolicy"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        env = Monitor(env, self.log_dir)
        self.algorithm = algorithm
        self.model = algorithm(policy, env, verbose=1)

        logger = configure(log_dir, ["csv"])
        self.model.set_logger(logger)

    def learn(self, total_timesteps: int, check_freq: int = 1000) -> None:
        callback = SaveOnBestTrainingRewardCallback(
            check_freq=check_freq, log_dir=self.log_dir)

        self.model.learn(total_timesteps=total_timesteps,
                         callback=callback)
        self.load_best_model()

    def load_best_model(self) -> None:
        path = self.log_dir + "best_model.zip"
        if exists(path):
            del self.model
            self.model = self.algorithm.load(path)
            print("Best model loaded.")
        else:
            print("No model file exists.")

    def predict(self, obs, deterministic: bool = True):
        return self.model.predict(obs, deterministic=deterministic)


class WrappedPPO(WrappedModel):
    def __init__(self, env: Env, log_dir: str, policy="MlpPolicy") -> None:
        super().__init__(PPO, env, log_dir, policy)


class WrappedDQN:
    def __init__(self, env: Env, log_dir: str, policy="MlpPolicy") -> None:
        super().__init__(DQN, env, log_dir, policy)
