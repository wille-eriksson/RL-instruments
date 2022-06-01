from fractions import Fraction
import os
import pathlib
import csv
from random import randint
import numpy as np
from rl_instruments.environments.piano_rolls.audio_target import AudioTargetEnv
from rl_instruments.models import WrappedDQN, WrappedModel, WrappedPPO
from rl_instruments.utils.piano import PianoRollManager, predict_piano_roll


def generate_random_piano_roll(n_notes: int, n_keys: int) -> np.ndarray:

    piano_roll = np.zeros((n_keys, n_notes), dtype=np.int8)

    for note in range(n_notes):
        pressed_key = randint(0, n_keys-1)
        piano_roll[pressed_key, note] = 100

    return piano_roll


def save_piano_roll(log_dir: str, filename: str, piano_roll: np.ndarray) -> None:
    complete_filename = f"{log_dir}/{filename}.npy"
    np.save(complete_filename, piano_roll)


def save_experiment_parameters(log_dir: str,
                               n_runs: int,
                               total_timesteps: int,
                               sample_rate: int,
                               bpm: int,
                               n_keys: int,
                               n_notes: int,
                               note_value: int,
                               base_note_number: int) -> None:

    header = ['Number of runs', 'Total timesteps', 'Sample rate', 'BPM',
              "Number of keys", "Number of notes", 'Note value', 'Base note number']
    data = [n_runs, total_timesteps, sample_rate, bpm, n_keys,
            n_notes,  str(Fraction(note_value)), base_note_number]

    filename = log_dir + "/experiment_parameters.csv"

    with open(filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)


def run_audio_target_experiment(algorithm: WrappedModel,
                                base_log_path: str,
                                n_runs: int,
                                total_timesteps: int,
                                sample_rate: int = 8000,
                                bpm: int = 120,
                                n_keys: int = 12,
                                n_notes: int = 4,
                                note_value: int = 1/8,
                                base_note_number: int = 60) -> None:

    # Create log directory if it does not exist and save expariment parameters

    log_dir = f"{base_log_path}/"
    os.makedirs(log_dir, exist_ok=True)

    save_experiment_parameters(base_log_path,
                               n_runs,
                               total_timesteps,
                               sample_rate,
                               bpm,
                               n_keys,
                               n_notes,
                               note_value,
                               base_note_number)

    for run in range(1, 1):

        # Create path to log directory

        log_dir = f"{base_log_path}/{algorithm.__name__}/runs/{run}/"
        os.makedirs(log_dir, exist_ok=True)

        # Generate a random piano roll
        target_piano_roll = generate_random_piano_roll(n_notes, n_keys)
        save_piano_roll(log_dir, "target_piano_roll", target_piano_roll)

        # Synthesize audio
        target_audio = PianoRollManager(
            target_piano_roll, bpm, note_value, sample_rate, base_note_number).get_audio()

        # Create environment
        env = AudioTargetEnv(target_audio, sample_rate, bpm,
                             note_value, n_notes, n_keys)

        # Create and train model
        model = algorithm(env, log_dir)
        model.learn(total_timesteps)

        # Make and save predicitons
        predicted_piano_roll, rewards = predict_piano_roll(model, env)

        save_piano_roll(log_dir, "predicted_piano_roll", predicted_piano_roll)
        np.save(log_dir + "rewards.npy", np.array(rewards))


if __name__ == '__main__':

    # Audio generation parameters

    SR = 8000
    BPM = 120
    N_KEYS = 12
    N_NOTES_ARRAY = [4, 8]
    NOTE_VALUE = 1/8
    BASE_NOTE_NUMBER = 60

    # Training parameters

    N_RUNS: int = 50
    TOTAL_TIMESTEPS: int = 50000

    # Define path for logging experiment

    for N_NOTES in N_NOTES_ARRAY:
        EXPERIMENT_NAME = f"audio_target-{N_NOTES}-notes"

        for ALGORITHM in [WrappedPPO, WrappedDQN]:
            # Define path for logging experiment
            BASE_LOG_PATH: str = f"{pathlib.Path(__file__).parent.resolve()}/logs/{EXPERIMENT_NAME}"

            run_audio_target_experiment(ALGORITHM,
                                        BASE_LOG_PATH,
                                        N_RUNS,
                                        TOTAL_TIMESTEPS,
                                        SR,
                                        BPM,
                                        N_KEYS,
                                        N_NOTES,
                                        NOTE_VALUE,
                                        BASE_NOTE_NUMBER)
