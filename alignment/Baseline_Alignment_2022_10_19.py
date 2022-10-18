#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
single script challenge submission template
"""

import partitura as pt
import os

# Uncomment this line if the kernel keeps crashing
# See https://stackoverflow.com/a/53014308
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from typing import Union, List

import numpy as np
from fastdtw import fastdtw
from scipy.spatial import distance as sp_dist

from challenge_utils import (
    load_dataset,
    compare_alignments,
    export_to_challenge,
)


#### FEATURES: PIANO ROLLS


def compute_pianoroll_score(
    score_note_array: np.ndarray,
    time_div: Union[str, int] = "auto",
) -> (np.ndarray, np.ndarray):
    """
    Compute Piano Roll for the Score

    Parameters
    ----------
    score_note_array : structured array
        Structured array with note information for the score
    time_div : int or "auto"
       Resolution of the piano roll (how many "cells" per beat).

    Returns
    -------
    pianoroll : np.ndarray
       A 2D pianoroll where rows represent time steps (1 / time_div)
       and columns represent musical pitch
    idx : np.ndarray
       An array of indices of the notes in the spart

    Note: In this example, both `compute_pianoroll_score` and
    `compute_pianoroll_performance` are almost identical functions,
    but you can/should play with different settings/options/features
    """
    piano_roll, idx = pt.utils.music.compute_pianoroll(
        note_info=score_note_array,
        return_idxs=True,
        piano_range=True,  # Since we are using only piano music,
        time_div=time_div,
    )
    return piano_roll.todense().T, idx


def compute_pianoroll_performance(
    performance_note_array: np.ndarray, time_div: Union[str, int] = "auto"
) -> (np.ndarray, np.ndarray):
    """
    Compute Piano Roll for the Score

    Parameters
    ----------
    performance_note_array : structured array
       Structured array with performance information
    time_div : int or "auto"
       Resolution of the piano roll (how many "cells" per second).

    Returns
    -------
    pianoroll : np.ndarray
       A 2D pianoroll where rows represent time steps (1 / time_div)
       and columns represent musical pitch
    idx : np.ndarray
       An array of indices of the notes in the ppart

    Note: In this example, both `compute_pianoroll_score` and
    `compute_pianoroll_performance` are almost identical functions,
    but you can/should play with different settings/options/features
    """
    piano_roll, idx = pt.utils.music.compute_pianoroll(
        note_info=performance_note_array,
        return_idxs=True,
        piano_range=True,  # Since we are using only piano music,
        time_div=time_div,
    )

    # Discard MIDI velocity
    piano_roll = piano_roll.todense().T
    piano_roll[piano_roll > 0] = 1
    return piano_roll, idx


#### DYNAMIC TIME WARPING


def fast_dynamic_time_warping(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
) -> (np.ndarray, float):
    """
     Fast Dynamic Time Warping

    This is an approximate solution to dynamic time warping.

    Parameters
    ----------
    X : np.ndarray
    Y: np.ndarray
    metric : str
       The name of the metric to use

    Returns
    -------
    warping_path: np.ndarray
        The warping path for the best alignment
    dtwd : float
        The dynamic time warping distance of the alignment.
    """

    # Get distance measure from scipy dist

    if metric == "euclidean":
        # Use faster implementation
        dist = 2
    else:
        dist = getattr(sp_dist, metric)
    dtwd, warping_path = fastdtw(X, Y, dist=dist)

    # Make path a numpy array
    warping_path = np.array(warping_path)
    return warping_path, dtwd


#### NOTEWISE ALIGNMENT


def greedy_note_alignment(
    warping_path: np.ndarray,
    idx1: np.ndarray,
    note_array1: np.ndarray,
    idx2: np.ndarray,
    note_array2: np.ndarray,
) -> List[dict]:
    """
    Greedily find and store possible note alignments

    Parameters
    ----------
    warping_path : numpy ndarray
        alignment sequence idx in stacked columns
    idx1: numpy ndarray
        pitch, start, and end coordinates of all notes in note_array1
    note_array1: numpy structured array
        note_array of sequence 1 (the score)
    idx2: numpy ndarray
        pitch, start, and end coordinates of all notes in note_array2
    note_array2: numpy structured array
        note_array of sequence 2 (the performance)

    Returns
    ----------
    note_alignment : list
        list of note alignment dictionaries

    """
    note_alignment = []
    used_notes1 = list()
    used_notes2 = list()

    # loop over all notes in sequence 1
    for note1, coord1 in zip(note_array1, idx1):
        note1_id = note1["id"]
        pitch1, s1, e1 = coord1

        # find the coordinates of the note in the warping_path

        idx_in_warping_path = np.all(
            [warping_path[:, 0] >= s1, warping_path[:, 0] <= e1], axis=0
        )
        # print(idx_in_warping_path, idx_in_warping_path.shape)
        range_in_sequence2 = warping_path[idx_in_warping_path, 1]
        max2 = np.max(range_in_sequence2)
        min2 = np.min(range_in_sequence2)

        # loop over all notes in sequence 2 and pick the notes with same pitch
        # and position
        for note2, coord2 in zip(note_array2, idx2):
            note2_id = note2["id"]
            pitch2, s2, e2 = coord2
            if note2_id not in used_notes2:
                if pitch2 == pitch1 and s2 <= max2 and e2 >= min2:

                    note_alignment.append(
                        {
                            "label": "match",
                            "score_id": note1_id,
                            "performance_id": str(note2_id),
                        }
                    )
                    used_notes2.append(str(note2_id))
                    used_notes1.append(note1_id)

        # check if a note has been found for the sequence 1 note,
        # otherwise add it as deletion
        if note1_id not in used_notes1:
            note_alignment.append({"label": "deletion", "score_id": note1_id})
            used_notes1.append(note1_id)

    # check again for all notes in sequence 2, if not used,
    # add them as insertions
    for note2 in note_array2:
        note2_id = note2["id"]
        if note2_id not in used_notes2:
            note_alignment.append(
                {
                    "label": "insertion",
                    "performance_id": str(note2_id),
                }
            )
            used_notes2.append(str(note2_id))

    return note_alignment


if __name__ == "__main__":

    import argparse

    # DO NOT CHANGE THIS!
    parser = argparse.ArgumentParser(
        description="Score-to-performance Alignment",
    )
    parser.add_argument(
        "--datadir",
        "-i",
        help="path to the input files",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--outdir",
        "-o",
        help="Output text file directory",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--challenge",
        "-c",
        help="Export results for challenge",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if args.datadir is None:
        raise ValueError("No data directory given")

    # 1. Load the data
    dataset = load_dataset(args.datadir)

    alignments = []
    evaluation = []
    piece_names = []
    for i, (piece_name, pdata) in enumerate(dataset.items()):

        piece_names.append(piece_name)

        performance_note_array, score_note_array, gt_alignment = pdata

        # 2. Compute the features (Adapt this part as needed!)
        score_features, score_idx = compute_pianoroll_score(
            score_note_array=score_note_array,
            time_div="auto",
        )

        performance_features, performance_idx = compute_pianoroll_performance(
            performance_note_array=performance_note_array,
            time_div="auto",
        )

        # 3. Compute the alignment (Adapt this part as needed!)
        warping_path, _ = fast_dynamic_time_warping(
            X=score_features,
            Y=performance_features,
            metric="euclidean",
        )

        predicted_alignment = greedy_note_alignment(
            warping_path=warping_path,
            idx1=score_idx,
            note_array1=score_note_array,
            idx2=performance_idx,
            note_array2=performance_note_array,
        )

        # Compute evaluation (Do not change this)
        alignments.append(predicted_alignment)

        piece_eval = compare_alignments(
            prediction=predicted_alignment,
            ground_truth=gt_alignment,
        )

        print(
            f"{i+1}/{len(dataset)} {piece_name}: "
            f"F-score:{piece_eval[2]:.2f} "
            f"Precision:{piece_eval[0]:.2f} "
            f"Recall:{piece_eval[1]:.2f}"
        )

        evaluation.append(piece_eval)

    # compute mean evaluation
    mean_eval = np.mean(evaluation, 0)

    print(
        "\n\nAverage Performance over the dataset\n"
        f"F-score:{mean_eval[2]:.2f}\t"
        f"Precision:{mean_eval[0]:.2f}\t",
        f"Recall:{mean_eval[1]:.2f}",
    )
    if args.challenge:
        # Do not modify this!
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        outfile = os.path.join(args.outdir, f"{script_name}_challenge.npz")

        export_to_challenge(
            alignments=alignments,
            piece_names=piece_names,
            out=outfile,
        )
