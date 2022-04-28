from fractions import Fraction
import os
import numpy as np
import pathlib
import csv
from random import randint
from rl_instruments.environments.piano_rolls.audio_target import AudioTargetEnv
from rl_instruments.models import WrappedPPO
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
                               sr: int,
                               bpm: int,
                               n_keys: int,
                               n_notes: int,
                               note_value: int,
                               base_note_number: int) -> None:

    header = ['Number of runs', 'Total timesteps', 'Sample rate', 'BPM',
              "Number of keys", "Number of notes", 'Note value', 'Base note number']
    data = [n_runs, total_timesteps, sr, bpm, n_keys,
            n_notes,  str(Fraction(note_value)), base_note_number]

    filename = log_dir + "/experiment_parameters.csv"

    with open(filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)


def run_audio_target_experiment(base_log_path: str,
                                n_runs: int,
                                total_timesteps: int,
                                sr: int = 8000,
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
                               sr,
                               bpm,
                               n_keys,
                               n_notes,
                               note_value,
                               base_note_number)

    for run in range(n_runs):

        # Create path to log directory

        log_dir = f"{base_log_path}/{run}/"
        os.makedirs(log_dir, exist_ok=True)

        # Generate a random piano roll
        target_piano_roll = generate_random_piano_roll(n_notes, n_keys)
        save_piano_roll(log_dir, "target_piano_roll", target_piano_roll)

        # Synthesize audio
        target_audio = PianoRollManager(
            target_piano_roll, bpm, note_value, sr, base_note_number).get_audio()

        # Create environment
        env = AudioTargetEnv(target_audio, sr, bpm,
                             note_value, n_notes, n_keys)

        # Create and train model
        model = WrappedPPO(env, log_dir)
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
    N_NOTES = 4
    NOTE_VALUE = 1/8
    BASE_NOTE_NUMBER = 60

    EXPERIMENT_NAME = "audio_target"

    # Training parameters

    N_RUNS: int = 1
    TOTAL_TIMESTEPS: int = 1000

    # Define path for logging experiment

    BASE_LOG_PATH: str = f"{pathlib.Path(__file__).parent.resolve()}/logs/{EXPERIMENT_NAME}"

    run_audio_target_experiment(BASE_LOG_PATH,
                                N_RUNS,
                                TOTAL_TIMESTEPS,
                                SR,
                                BPM,
                                N_KEYS,
                                N_NOTES,
                                NOTE_VALUE,
                                BASE_NOTE_NUMBER)
