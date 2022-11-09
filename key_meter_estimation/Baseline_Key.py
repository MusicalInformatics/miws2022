#!/usr/bin/env python
# -*- coding: utf-8 -*-

# single script challenge submission template
from typing import Union, Tuple

import numpy as np
import partitura as pt

from scipy.stats import mode

from partitura.utils.misc import PathLike

from hiddenmarkov import HMM, ConstantTransitionModel, ObservationModel

from key_profiles import build_key_profile_matrix, KEYS

from key_estimation_challenge import compare_keys, load_submission

import warnings

warnings.filterwarnings("ignore")

## DEFINE HMM


class KeyProfileObservationModel(ObservationModel):
    """
    Observation model that takes a pitch class distribution
    and returns pitch class profiles.

    Parameters
    ----------
    key_profile_matrix : np.ndarray or {'kk','cbms','kp'}
        The key profile matrix. If a string is given, it needs to be
        in {'kk','cbms','kp'}. Otherwise, a (24, 12) array is expected.
    """

    def __init__(
        self,
        key_profile_matrix: Union[str, np.ndarray] = "kp",
    ) -> None:
        super().__init__()
        if isinstance(key_profile_matrix, str):
            self.key_profile_matrix = build_key_profile_matrix(
                profile=key_profile_matrix,
            )
        elif isinstance(key_profile_matrix, np.ndarray):
            assert key_profile_matrix.shape == (24, 12)
            self.key_profile_matrix = key_profile_matrix

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """
        Give the likelihood of the observed pitch class distribution
        given the keys.

        Parameters
        ----------
        observation : np.ndarray
            A 12-dimensional vector representing the pitch class distribution

        Returns
        -------
        likelihood: np.ndarray
            A 24-dimensional vector representing the likelihood of the
            observed pitch class distribution given the keys.
            If `self.use_log_probabilities` is True, this is
            the log-likelihood, otherwise returns is the actual probabilities.
        """
        if not self.use_log_probabilities:
            p_obs_given_key = np.array(
                [
                    np.prod((kp ** observation) * (1 - kp) ** (1 - observation))
                    for kp in self.key_profile_matrix
                ]
            )
            likelihood = p_obs_given_key
        elif self.use_log_probabilities:

            log_p_obs_given_key = np.array(
                [
                    np.sum(
                        (
                            observation * np.log(kp + 1e-10)
                            + np.log1p(-(kp + 1e-10)) * (1 - observation)
                        )
                    )
                    for kp in self.key_profile_matrix
                ]
            )
            likelihood = log_p_obs_given_key

        return likelihood


def compute_transition_probabilities(inertia_param: float = 0.8) -> np.ndarray:
    """
    Matrix of transition probabilities

    Parameters
    ----------
    intertia_param : float
        Parameter between 0 and 1 indicating how likely it is that
        we will stay on the same key

    Returns
    -------
    A : np.ndarray
        Matrix of transition probabilities.

    Notes
    -----
    * This is a very naive assumption, but so you should definitely explore
    other transition probabilities
    """
    modulation_prob = (1 - inertia_param) / 23.0
    A = modulation_prob * (np.ones(24) - np.eye(24)) + inertia_param * np.eye(24)

    return A


def key_identification(
    filename: PathLike,
    key_profiles: Union[str, np.ndarray] = "kp",
    inertia_param: float = 0.8,
    piano_roll_resolution: int = 16,
    win_size: float = 2,
) -> Tuple[str, float]:
    """
    Temperley's Probabilistic Key Identification

    Parameters
    ----------
    fn : filename
        MIDI file
    key_profiles: {"kp", "kk", "cbms"}
        Key profiles to use in the KeyProfileObservationModel
        (see definition in `key_profiles.py`)
    intertia_param: float
        Parameter between 0 and 1 indicating how likely it is that
        we will stay on the same key
    piano_roll_resolution: int
        Resolution of the piano roll (i.e., how many cells per second)
    win_size: float
        Window size in seconds

    Returns
    -------
    key : str
        The estimated key of the piece
    log_lik:
        The log-likelihood of the estimated key
    """
    # build observation model
    observation_model = KeyProfileObservationModel(
        key_profile_matrix=key_profiles,
    )

    # Compute transition model
    transition_probabilities = compute_transition_probabilities(
        inertia_param=inertia_param
    )
    transition_model = ConstantTransitionModel(transition_probabilities)

    hmm = HMM(
        observation_model=observation_model,
        transition_model=transition_model,
    )
    # Load score
    perf = pt.load_performance_midi(filename)

    # Compute piano roll
    piano_roll = pt.utils.compute_pianoroll(
        perf,
        time_div=piano_roll_resolution,
    ).toarray()

    # Number of windows in the piano roll
    n_windows = int(np.ceil(piano_roll.shape[1] / (piano_roll_resolution * win_size)))

    # window size in cells
    window_size = win_size * piano_roll_resolution

    # Constuct observations (these are non-overlapping windows,
    # but you can test other possibilities)
    observations = np.zeros((n_windows, 12))
    for win in range(n_windows):
        idx = slice(win * window_size, (win + 1) * window_size)
        segment = piano_roll[:, idx].sum(1)
        dist = np.zeros(12)
        pitch_idxs = np.where(segment != 0)[0]
        for pix in pitch_idxs:
            dist[pix % 12] += segment[pix]
        # Normalize pitch class distribution
        if dist.sum() > 0:
            # avoid NaN for empty segments
            dist /= dist.sum()

        observations[win] = dist

    # Compute the sequence
    path, _ = hmm.find_best_sequence(observations)

    key_idx = int(mode(path).mode[0])

    key = KEYS[key_idx]

    return key


if __name__ == "__main__":

    import argparse
    import os
    import glob

    # DO NOT CHANGE THIS!
    parser = argparse.ArgumentParser(description="Key Estimation")
    parser.add_argument(
        "--datadir",
        "-i",
        help="path to the input files",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--challenge",
        "-c",
        help="Export results for challenge",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--outfile",
        "-o",
        help="Output file",
        type=str,
        default="key_estimation.txt",
    )

    parser.add_argument(
        "--ground-truth",
        "-t",
        help="File with the ground truth labels",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # Adapt this part as needed!
    midi_files = glob.glob(os.path.join(args.datadir, "*.mid"))
    midi_files.sort()

    ground_truth = {}
    if args.ground_truth:
        ground_truth = load_submission(args.ground_truth)

    results = []
    evaluation = []
    for i, mfn in enumerate(midi_files):

        piece = os.path.basename(mfn)
        predicted_key = key_identification(
            filename=mfn,
            key_profiles="kp",
            inertia_param=0.8,
            piano_roll_resolution=16,
            win_size=2,
        )

        results.append((piece, predicted_key))

        if piece in ground_truth:
            expected_key = ground_truth[piece]
            tonal_distance = compare_keys(predicted_key, expected_key)
            acc = expected_key == predicted_key
            print(
                f"{i+1}/{len(midi_files)} {piece}: "
                f"\tPredicted:{predicted_key} "
                f"\tExpected:{expected_key} "
                f"\tTonal Distance:{tonal_distance}"
            )
            evaluation.append((tonal_distance, acc))

    mean_eval = np.mean(evaluation, 0)
    if len(evaluation) > 0:
        print("\n\nAverage Performance over dataset")
        print(f"\tTonal Distance{mean_eval[0]: .2f}")
        print(f"\tAccuracy: {mean_eval[1]: .2f}")

    if args.challenge:
        # Export predictions for the challenge
        np.savetxt(
            args.outfile,
            np.array(results),
            fmt="%s",
            delimiter=",",
            comments="//",
            header="filename,key",
        )
