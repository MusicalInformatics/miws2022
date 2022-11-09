#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluates a submission on the challenge server,
computing the average tonal distance
"""
import argparse
import re
import numpy as np

key_pat = re.compile("([A-G])([xb\#]*)(m*)")

MAJOR_KEYS = [
    "F",
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
    "G#",
    "D#",
    "A#",
    "E#",
]

MINOR_KEYS_u = [k + "m" for k in MAJOR_KEYS]
MINOR_KEYS = MINOR_KEYS_u[4:] + MINOR_KEYS_u[:4]


def enharmonic_spelling(key: str) -> str:
    """
    Use enharmonic spelling to rename
    the labels to the list of expected keys.
    as specified in `MAJOR_KEYS` and `MINOR_KEYS`

    Parameter
    ---------
    key : str
        A string representing the key.

    Returns
    -------
    key : str
        The enharmonic spelling of the key appearing
        in the labels.
    """
    if key == "None":
        print(f"Invalid key: {key}")
        return "C"
    steps = ["C", "D", "E", "F", "G", "A", "B"]

    step, alter, mode = key_pat.search(key).groups()

    if step + alter == "B#":
        return "C"
    elif step + alter == "E#":
        return "F"
    elif step + alter == "Cb":
        return "B"
    elif step + alter == "Fb":
        return "E"

    if alter == "b":
        kstep = steps.index(step) - 1
        return steps[kstep] + "#" + mode
    else:
        return key


def compare_keys(prediction_key: str, ground_truth_key: str) -> int:
    """
    Tonal Distance between two keys.

    This method computes the tonal distance (in terms of closeness in
    the circle of fifths).

    Parameters
    ----------
    prediction_key: str
        Predicted key.

    ground_truth_key: str
        Ground truth key.

    Returns
    -------
    score: int
        Tonal distance.
    """

    pred_key = enharmonic_spelling(prediction_key)
    gt_key = enharmonic_spelling(ground_truth_key)
    if pred_key in MAJOR_KEYS and gt_key in MAJOR_KEYS:
        pidx = MAJOR_KEYS.index(pred_key)
        gidx = MAJOR_KEYS.index(gt_key)
        return min((gidx - pidx) % 12, (pidx - gidx) % 12)
    elif pred_key in MINOR_KEYS and gt_key in MINOR_KEYS:
        pidx = MINOR_KEYS.index(pred_key)
        gidx = MINOR_KEYS.index(gt_key)
        return min((gidx - pidx) % 12, (pidx - gidx) % 12)
    elif pred_key in MAJOR_KEYS and gt_key in MINOR_KEYS:
        pidx = MAJOR_KEYS.index(pred_key)
        gidx = MINOR_KEYS.index(gt_key)
        return min((gidx - pidx) % 12, (pidx - gidx) % 12) + 1
    elif pred_key in MINOR_KEYS and gt_key in MAJOR_KEYS:
        pidx = MINOR_KEYS.index(pred_key)
        gidx = MAJOR_KEYS.index(gt_key)
        return min((gidx - pidx) % 12, (pidx - gidx) % 12) + 1
    else:
        raise Exception(
            "input keys need to be in the following format:",
            MAJOR_KEYS,
            MINOR_KEYS,
        )


def load_submission(fn: str) -> dict:
    """
    Load a submission
    """

    gt = np.loadtxt(
        fn,
        dtype=str,
        delimiter=",",
        comments="//",
    )

    ground_truth = dict([(g[0], g[1]) for g in gt])

    return ground_truth


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    tonal_distance = []

    submission = load_submission(args.submission)

    target = load_submission(args.target)

    for piece, key in target.items():

        if piece in submission:
            td = compare_keys(submission[piece], key)
        else:
            # If the piece is not found, assume that
            # the maximal tonal distance.
            td = 7

        tonal_distance.append(td)
    mean_score = np.mean(tonal_distance)
    print(mean_score)
