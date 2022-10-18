#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for the alignment challenge
"""
import glob
import os

import numpy as np

from typing import List, Dict, Tuple

from partitura.utils.misc import PathLike

from partitura.io.importparangonada import (
    _load_csv,
    load_parangonada_alignment,
)


def load_challenge_piece(
    dirname: PathLike,
) -> np.ndarray:
    """
    Load Parangonada Project alignment files

    Parameters
    ----------
    dirname : PathLike
        Directory with the CSV files in Parangonada

    Returns
    -------
    piece_name : str
        The name of the piece
    perf_note_array : np.ndarray
        Structured array with performance information
    score_note_array
        A note array containing note information in the score.
        score object in a future release!
    alignment : List of dict
        The main alignment (If there is no align.csv in `dirname`, it will
        return an empty list)
    """

    piece_name = os.path.basename(dirname)
    # Get filenames
    perf_note_array_fn = os.path.join(dirname, "ppart.csv")
    score_note_array_fn = os.path.join(dirname, "part.csv")
    alignment_fn = os.path.join(dirname, "align.csv")

    # Load performance note array
    perf_note_array = _load_csv(perf_note_array_fn)

    # Load score note array
    score_note_array = _load_csv(score_note_array_fn)

    # The alignment is initialized as an empty list
    alignment = []

    if os.path.exists(alignment_fn):
        alignment = load_parangonada_alignment(alignment_fn)

    return (
        piece_name,
        perf_note_array,
        score_note_array,
        alignment,
    )


def load_dataset(
    data_dir: PathLike,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[dict]]]:
    """
    Load a dataset of alignments

    Parameters
    ----------
    data_dir : PathLike
        Path to the directory with the data

    Returns
    -------
    dataset: dict
        A dictionary with score and performance note arrays for each piece.
        If there is a ground truth alignments they will also be loaded
        (otherwise the dataset includes an empty list).
    """
    # Get all piece directories
    piece_dirs = [
        pd for pd in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(pd)
    ]

    dataset = dict()
    for pd in piece_dirs:

        piece_name, perf_na, score_na, alignment = load_challenge_piece(pd)

        dataset[piece_name] = (perf_na, score_na, alignment)

    return dataset


def compare_alignments(
    prediction: List[dict],
    ground_truth: List[dict],
    types: List[str] = ["match", "deletion", "insertion"],
) -> (float, float, float):
    """
    Parameters
    ----------
    prediction: List of dicts
        List of dictionaries containing the predicted alignments
    ground_truth:
        List of dictionaries containing the ground truth alignments
    types: List of strings
        List of alignment types to consider for evaluation
        (e.g ['match', 'deletion', 'insertion']

    Returns
    -------
    precision: float
       The precision
    recall: float
        The recall
    f_score: float
       The F score
    """

    pred_filtered = list(filter(lambda x: x["label"] in types, prediction))
    gt_filtered = list(filter(lambda x: x["label"] in types, ground_truth))

    filtered_correct = [pred for pred in pred_filtered if pred in gt_filtered]

    n_pred_filtered = len(pred_filtered)
    n_gt_filtered = len(gt_filtered)
    n_correct = len(filtered_correct)

    if n_pred_filtered > 0 or n_gt_filtered > 0:
        precision = n_correct / n_pred_filtered if n_pred_filtered > 0 else 0.0
        recall = n_correct / n_gt_filtered if n_gt_filtered > 0 else 0
        f_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
    else:
        # no prediction and no ground truth for a
        # given type -> correct alignment
        precision, recall, f_score = 1.0, 1.0, 1.0

    return precision, recall, f_score


def export_to_challenge(
    alignments: List[List[dict]],
    piece_names: List[str],
    out: PathLike,
) -> None:
    """
    Export alignement result to be uploaded to the challenge server

    Parameters
    ----------
    alignments: List of lists of dictionaries
        List of alignments for each piece
    piece_names : List of strings
        Name of each piece
    out: PathLike
       The output file
    """
    dataset = dict(
        [
            (pname, algn)
            for pname, algn in zip(
                piece_names,
                alignments,
            )
        ]
    )

    np.savez_compressed(
        file=out,
        **dataset,
    )


if __name__ == "__main__":

    pass
