import os
import pathlib
import csv
from random import randint
import numpy as np
from rl_instruments.environments.piano_rolls.piano_roll_target import PianoRollTargetEnv
from rl_instruments.models import WrappedDQN, WrappedModel
from rl_instruments.utils.piano import predict_piano_roll


def generate_random_piano_roll(n_notes: int, n_keys: int) -> np.ndarray:

    piano_roll = np.zeros((n_keys, n_notes), dtype=np.int8)

    for note in range(n_notes):
        pressed_key = randint(0, n_keys-1)
        piano_roll[pressed_key, note] = 1

    return piano_roll


def save_piano_roll(log_dir: str, filename: str, piano_roll: np.ndarray) -> None:
    complete_filename = f"{log_dir}/{filename}.npy"
    np.save(complete_filename, piano_roll)


def save_experiment_parameters(log_dir: str,
                               n_runs: int,
                               total_timesteps: int,
                               n_keys: int,
                               n_notes: int,
                               ) -> None:

    header = ['Number of runs', 'Total timesteps',
              "Number of keys", "Number of notes"]
    data = [n_runs, total_timesteps, n_keys, n_notes]

    filename = log_dir + "/experiment_parameters.csv"

    with open(filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)


def run_piano_roll_target_experiment(algorithm: WrappedModel,
                                     base_log_path: str,
                                     n_runs: int,
                                     total_timesteps: int,
                                     n_keys: int = 12,
                                     n_notes: int = 4) -> None:

    # Create log directory if it does not exist and save expariment parameters

    log_dir = f"{base_log_path}/"
    os.makedirs(log_dir, exist_ok=True)

    save_experiment_parameters(base_log_path,
                               n_runs,
                               total_timesteps,
                               n_keys,
                               n_notes)

    for run in range(n_runs):

        # Create path to log directory

        log_dir = f"{base_log_path}/{algorithm.__name__}/runs/{run}/"
        os.makedirs(log_dir, exist_ok=True)

        # Generate a random piano roll
        target_piano_roll = generate_random_piano_roll(n_notes, n_keys)
        save_piano_roll(log_dir, "target_piano_roll", target_piano_roll)

        # Create environment
        env = PianoRollTargetEnv(target_piano_roll)

        # Create and train model
        model = algorithm(env, log_dir)
        model.learn(total_timesteps)

        # Make and save predicitons
        predicted_piano_roll, rewards = predict_piano_roll(model, env)

        save_piano_roll(log_dir, "predicted_piano_roll", predicted_piano_roll)
        np.save(log_dir + "rewards.npy", np.array(rewards))


if __name__ == '__main__':

    # Audio generation parameters

    N_KEYS = 12
    N_NOTES_ARRAY = [4, 8]

    # Training parameters

    N_RUNS: int = 50
    TOTAL_TIMESTEPS: int = 30000

    for N_NOTES in N_NOTES_ARRAY:
        EXPERIMENT_NAME = f"piano_roll_target-{N_NOTES}-notes"
        # Define path for logging experiment

        BASE_LOG_PATH: str = f"{pathlib.Path(__file__).parent.resolve()}/logs/{EXPERIMENT_NAME}"

        for ALGORITHM in [WrappedDQN]:
            run_piano_roll_target_experiment(ALGORITHM,
                                             BASE_LOG_PATH,
                                             N_RUNS,
                                             TOTAL_TIMESTEPS,
                                             N_KEYS,
                                             N_NOTES)
