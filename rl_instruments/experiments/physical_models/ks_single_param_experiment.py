from fractions import Fraction
from typing import Tuple, List
import os
import pathlib
import csv
from random import randint
import numpy as np
from scipy.io.wavfile import write
from rl_instruments.environments.physical_models import KSSingleParamEnv, ControlableParameter
from rl_instruments.models import WrappedPPO
from rl_instruments.utils.ks import make_melody, MelodyData, predict_melody


def create_target_parameters(controlable_parameter: ControlableParameter,
                             controlable_array: List[str]) -> Tuple[List[float],
                                                                    List[float],
                                                                    List[float],
                                                                    List[float]]:

    n_notes = len(controlable_array)
    target_params = [controlable_array if controlable_parameter is param else [
        param.default_value]*n_notes for param in sorted(ControlableParameter)]

    return target_params


def create_target_melody(frequencies: List[float],
                         pluck_positions: List[float],
                         loss_factors: List[float],
                         amplitudes: List[float],
                         sr: int,
                         bpm: int,
                         note_value: int) -> MelodyData:

    target_audio = make_melody(frequencies, pluck_positions, loss_factors,
                               amplitudes, bpm, sr, note_value)

    return MelodyData(
        target_audio, sr, bpm, len(frequencies), note_value)


def generate_random_parameters(n_notes: int, controlable_parameter: ControlableParameter) -> List[int]:

    min_value = controlable_parameter.min_value
    max_value = controlable_parameter.max_value
    n = controlable_parameter.n

    return [min_value + randint(0, n-1) * (max_value - min_value) / (n-1) for _ in range(n_notes)]


def save_experiment_parameters(log_dir: str,
                               n_runs: int,
                               total_timesteps: int,
                               controlable_parameter: ControlableParameter,
                               sr: int,
                               bpm: int,
                               n_notes: int,
                               note_value: int) -> None:

    header = ['Number of runs', 'Total timesteps', 'Controlable parameter',
              'Sample rate', 'BPM', "Number of notes", 'Note value']
    data = [n_runs, total_timesteps,  controlable_parameter.name,
            sr, bpm,  n_notes,  str(Fraction(note_value))]

    filename = log_dir + "/experiment_parameters.csv"

    with open(filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)


def save_target_parameters(log_dir: str,
                           frequencies: List[float],
                           pluck_positions: List[float],
                           loss_factors: List[float],
                           amplitudes: List[float]) -> None:

    header = ['Frequency', 'Pluck position', 'Loss factor', 'Amplitude']

    filename = log_dir + "/target_parameters.csv"

    with open(filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for f, pp, lf, a in zip(frequencies, pluck_positions, loss_factors, amplitudes):
            writer.writerow([f, pp, lf, a])


def save_prediction(log_dir: str, predicted_audio: np.ndarray, sr: int, predicted_parameters: np.ndarray, rewards: List[float]) -> None:
    header = ['Frequency', 'Pluck position',
              'Loss factor', 'Amplitude', 'Reward']

    parameter_filename = log_dir + "/predicted_parameters.csv"

    with open(parameter_filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for f, pp, lf, a, r in zip(predicted_parameters[:, 0],
                                   predicted_parameters[:, 1],
                                   predicted_parameters[:, 2],
                                   predicted_parameters[:, 3],
                                   rewards):
            writer.writerow([f, pp, lf, a, r])

    audio_filename = log_dir + "/predicted_audio.wav"
    write(audio_filename, sr, predicted_audio)


def run_single_param_ks_experiment(base_log_path: str,
                                   n_runs: int,
                                   total_timesteps: int,
                                   controlable_parameter: ControlableParameter,
                                   sr: int,
                                   bpm: int,
                                   n_notes: int,
                                   note_value: int) -> None:

    # Create log directory if it does not exist and save expariment parameters

    log_dir = f"{base_log_path}/"
    os.makedirs(log_dir, exist_ok=True)

    save_experiment_parameters(base_log_path,
                               n_runs,
                               total_timesteps,
                               controlable_parameter,
                               sr,
                               bpm,
                               n_notes,
                               note_value)

    for run in range(n_runs):
        # Create path to log directory

        log_dir = f"{base_log_path}/{run}/"
        os.makedirs(log_dir, exist_ok=True)

        # Generate array of controlable parameters
        controlable_array = generate_random_parameters(
            n_notes, controlable_parameter)

        # Extract and save parameters for melody
        frequencies, pluck_positions, loss_factors, amplitudes = create_target_parameters(
            controlable_parameter, controlable_array)

        save_target_parameters(log_dir,
                               frequencies,
                               pluck_positions,
                               loss_factors,
                               amplitudes)

        # Create target melody
        target_melody = create_target_melody(
            frequencies, pluck_positions, loss_factors, amplitudes, sr, bpm, note_value)

        # Create environment and train model
        env = KSSingleParamEnv(target_melody, CONTROLABLE_PARAMETER)

        model = WrappedPPO(env, log_dir)
        model.learn(total_timesteps)

        # Make and save predictions
        predicted_audio, rewards, predicted_parameters = predict_melody(
            env, model)
        save_prediction(log_dir, predicted_audio, sr,
                        predicted_parameters, rewards)


if __name__ == '__main__':

    # Audio generation parameters

    SR = 8000
    BPM = 120
    N_NOTES = 4
    NOTE_VALUE = 1/8

    EXPERIMENT_NAME = "single_param"

    # Training parameters

    N_RUNS: int = 1
    TOTAL_TIMESTEPS: int = 10000
    CONTROLABLE_PARAMETER: ControlableParameter = ControlableParameter.LOSS_FACTOR

    # Define path for logging experiment

    BASE_LOG_PATH: str = f"{pathlib.Path(__file__).parent.resolve()}/logs/{EXPERIMENT_NAME}"

    # Run experiment
    run_single_param_ks_experiment(BASE_LOG_PATH,
                                   N_RUNS,
                                   TOTAL_TIMESTEPS,
                                   CONTROLABLE_PARAMETER,
                                   SR,
                                   BPM,
                                   N_NOTES,
                                   NOTE_VALUE)
