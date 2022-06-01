from fractions import Fraction
from typing import List, Tuple, Set
import os
import pathlib
import csv
import numpy as np
from scipy.io.wavfile import write
from rl_instruments.environments.physical_models import ControlableParameter
from rl_instruments.environments.physical_models.ks import KSMultiParamEnv
from rl_instruments.models import WrappedPPO
from rl_instruments.utils.ks import make_melody, MelodyData, predict_melody


def create_target_parameters() -> \
        Tuple[List[float], List[float], List[float], List[float]]:

    frequencies = [82, 123, 165, 208, 208, 165, 123, 82]
    pluck_positions = [0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]
    loss_factors = [0.996, 0.996, 0.996, 0.996, 0.91, 0.91, 0.91, 0.91]
    amplitudes = [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]

    return (frequencies, pluck_positions, loss_factors, amplitudes)


def create_target_melody(frequencies: List[float],
                         pluck_positions: List[float],
                         loss_factors: List[float],
                         amplitudes: List[float],
                         sample_rate: int,
                         bpm: int,
                         note_value: int) -> MelodyData:

    target_audio = make_melody(frequencies, pluck_positions, loss_factors,
                               amplitudes, bpm, sample_rate, note_value)

    return MelodyData(
        target_audio, sample_rate, bpm, len(frequencies), note_value)


def save_experiment_parameters(log_dir: str,
                               total_timesteps: int,
                               controlable_parameters: Set[ControlableParameter],
                               sample_rate: int,
                               bpm: int,
                               n_notes: int,
                               note_value: int) -> None:

    header = ['Total timesteps', 'Controlable parameter',
              'Sample rate', 'BPM', "Number of notes", 'Note value']
    data = [total_timesteps,  [cp.name for cp in controlable_parameters],
            sample_rate, bpm,  n_notes,  str(Fraction(note_value))]

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


def save_prediction(log_dir: str,
                    predicted_audio: np.ndarray,
                    sample_rate: int,
                    predicted_parameters: np.ndarray,
                    rewards: List[float]) -> None:
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
    write(audio_filename, sample_rate, predicted_audio)


def run_multi_param_ks_experiment(base_log_path: str,
                                  total_timesteps: int,
                                  controlable_parameters: Set[ControlableParameter],
                                  sample_rate: int,
                                  bpm: int,
                                  n_notes: int,
                                  note_value: int) -> None:

    # Create log directory if it does not exist and save expariment parameters

    log_dir = f"{base_log_path}/"
    os.makedirs(log_dir, exist_ok=True)

    save_experiment_parameters(base_log_path,
                               total_timesteps,
                               controlable_parameters,
                               sample_rate,
                               bpm,
                               n_notes,
                               note_value)

    # Extract and save parameters for melody
    frequencies, pluck_positions, loss_factors, amplitudes = create_target_parameters()

    os.makedirs(log_dir + "target", exist_ok=True)
    save_target_parameters(log_dir + "target",
                           frequencies,
                           pluck_positions,
                           loss_factors,
                           amplitudes)

    # Create target melody
    target_melody = create_target_melody(
        frequencies, pluck_positions, loss_factors, amplitudes, sample_rate, bpm, note_value)

    write(log_dir + "target/target_audio.wav",
          sample_rate, target_melody.audio)

    env = KSMultiParamEnv(target_melody, controlable_parameters)

    model = WrappedPPO(env, log_dir, info_keywords=(
        "frequency_reward", "envelope_reward"), seed=1, load_best=False)

    predicted_audio, rewards, predicted_parameters = predict_melody(
        env, model)

    timestep_dir = log_dir + str(0)

    os.makedirs(timestep_dir, exist_ok=True)

    save_prediction(timestep_dir, predicted_audio, sample_rate,
                    predicted_parameters, rewards)

    for i in range(1, 16):
        # if os.path.exists(log_dir + "best_model.zip"):
        #     os.remove(log_dir + "best_model.zip")

        # Create environment and train model

        model.learn(10000)

        # Make and save predictions
        predicted_audio, rewards, predicted_parameters = predict_melody(
            env, model)

        timestep_dir = log_dir + str(i*10000)

        os.makedirs(timestep_dir, exist_ok=True)

        save_prediction(timestep_dir, predicted_audio, sample_rate,
                        predicted_parameters, rewards)


if __name__ == '__main__':

    # Audio generation parameters

    SR = 8000
    BPM = 120
    N_NOTES = 8
    NOTE_VALUE = 1/8

    # Training parameters

    TOTAL_TIMESTEPS: int = 150000
    CONTROLABLE_PARAMETERS: Set[ControlableParameter] = {
        ControlableParameter.FREQUENCY,
        ControlableParameter.PLUCK_POSITION,
        ControlableParameter.LOSS_FACTOR,
        ControlableParameter.AMPLITUDE}

    EXPERIMENT_NAME = "training_illustration"

    BASE_LOG_PATH: str = f"{pathlib.Path(__file__).parent.resolve()}/logs/{EXPERIMENT_NAME}"

    # Run experiment
    run_multi_param_ks_experiment(BASE_LOG_PATH,
                                  TOTAL_TIMESTEPS,
                                  CONTROLABLE_PARAMETERS,
                                  SR,
                                  BPM,
                                  N_NOTES,
                                  NOTE_VALUE)
