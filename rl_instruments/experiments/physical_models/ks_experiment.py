import os
import numpy as np
import pathlib
import pickle
from random import randint
from scipy.io.wavfile import write
from rl_instruments.environments.physical_models import KSSingleParamEnv, ControlableParameter
from rl_instruments.models import WrappedPPO
from rl_instruments.utils.ks import make_melody, MelodyData, predict_melody


def create_target_parameters(controlable_parameter: ControlableParameter, controlable_array: 'list[str]') -> 'tuple[list[float], list[float], list[float], list[float]]':

    n_notes = len(controlable_array)
    target_params = {}

    for param in ControlableParameter:
        target_params[param] = controlable_array if controlable_parameter is param else [
            param.default_value]*n_notes

    return target_params[ControlableParameter.FREQUENCY], target_params[ControlableParameter.PLUCK_POSITION], target_params[ControlableParameter.LOSS_FACTOR], target_params[ControlableParameter.AMPLITUDE]


def create_target_melody(frequencies: 'list[float]',
                         pluck_positions: 'list[float]',
                         loss_factors: 'list[float]',
                         amplitudes: 'list[float]',
                         sr: int,
                         bpm: int,
                         note_value: int) -> MelodyData:

    target_audio = make_melody(frequencies, pluck_positions, loss_factors,
                               amplitudes, bpm, sr, note_value)

    return MelodyData(
        target_audio, sr, bpm, len(frequencies), note_value)


def generate_random_parameters(n_notes: int, controlable_parameter: ControlableParameter) -> 'list[int]':

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

    params = {'N_RUNS': n_runs, 'TOTAL_TIMESTEPS': total_timesteps, 'CONTROLABLE_PARAMETER': controlable_parameter.value,
              'sr': sr, 'bpm': bpm, "n_notes": n_notes, 'note_value': note_value}

    filename = log_dir + "experiment_parameters.pickle"

    with open(filename, 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_target_parameters(log_dir: str,
                           frequencies: 'list[float]',
                           pluck_positions: 'list[float]',
                           loss_factors: 'list[float]',
                           amplitudes: 'list[float]') -> None:

    params = {'frequencies': frequencies, 'pluck_positions': pluck_positions, 'loss_factors': loss_factors,
              'amplitudes': amplitudes}

    filename = log_dir + "target_parameters.pickle"

    with open(filename, 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_prediction(log_dir: str, predicted_audio: np.ndarray, sr: int, predicted_parameters: np.ndarray, score: float) -> None:
    params = {'frequencies': predicted_parameters[:, 0], 'pluck_positions': predicted_parameters[:, 1],
              'loss_factors': predicted_parameters[:, 2], 'amplitudes': predicted_parameters[:, 3], "score": score}

    parameter_filename = log_dir + "predicted_parameters.pickle"

    with open(parameter_filename, 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    audio_filename = log_dir + "predicted_audio.wav"
    write(audio_filename, sr, predicted_audio)


def run_experiment(base_log_path: str,
                   n_runs: int,
                   total_timesteps: int,
                   controlable_parameter: ControlableParameter,
                   sr: int = 8000,
                   bpm: int = 120,
                   n_notes: int = 4,
                   note_value: int = 1/8) -> None:

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
        predicted_audio, score, predicted_parameters = predict_melody(
            env, model)
        save_prediction(log_dir, predicted_audio, sr,
                        predicted_parameters, score)


if __name__ == '__main__':

    # Audio generation parameters

    SR = 8000
    BPM = 120
    N_NOTES = 4
    NOTE_VALUE = 1/8

    EXPERIMENT_NAME = "logs"

    # Training parameters

    N_RUNS: int = 3
    TOTAL_TIMESTEPS: int = 1000
    CONTROLABLE_PARAMETER: ControlableParameter = ControlableParameter.FREQUENCY

    BASE_LOG_PATH: str = f"{pathlib.Path(__file__).parent.resolve()}/{EXPERIMENT_NAME}"

    run_experiment(BASE_LOG_PATH,
                   N_RUNS,
                   TOTAL_TIMESTEPS,
                   CONTROLABLE_PARAMETER,
                   SR,
                   BPM,
                   N_NOTES,
                   NOTE_VALUE)
