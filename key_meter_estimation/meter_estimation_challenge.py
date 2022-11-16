#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluates a submission on the challenge server
"""
import argparse
import numpy as np


def compare_meter(pred_meter, target_meter):

    result = 0.0

    if pred_meter == target_meter:
        result = 1.0

    elif pred_meter / 2 == target_meter or pred_meter * 2 == target_meter:
        result = 0.5

    elif pred_meter / 3 == target_meter or pred_meter * 3 == target_meter:
        result = 0.5

    return result


def compute_tempo_error(pred_tempo, target_tempo):
    tempo_error = abs(pred_tempo - target_tempo)
    half_tempo_error = abs(0.5 * pred_tempo - target_tempo)
    double_tempo_error = abs(2 * pred_tempo - target_tempo)

    return min(tempo_error, half_tempo_error, double_tempo_error)


def compare_meter_and_tempo(
    pred_meter,
    target_meter,
    pred_tempo,
    target_tempo,
):

    tempo_error = compute_tempo_error(pred_tempo, target_tempo)

    meter_accuracy = compare_meter(pred_meter, target_meter)

    return meter_accuracy, tempo_error


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

    if gt.shape[1] > 3:
        ground_truth = dict([(g[0], (int(g[2]), float(g[4]))) for g in gt])
    else:
        ground_truth = dict([(g[0], (int(g[1]), float(g[2]))) for g in gt])

    return ground_truth


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    metrics = []

    submission = load_submission(args.submission)

    target = load_submission(args.target)

    for piece, (meter, tempo) in target.items():

        if piece in submission:
            pred_meter, pred_tempo = submission[piece]
            meter_accuracy, tempo_error = compare_meter_and_tempo(
                pred_meter, meter, pred_tempo, tempo
            )
        else:
            # If the piece is not found, assume that
            # the maximal tempo error and accuracy of zero
            meter_accuracy = 0
            tempo_error = tempo

        metrics.append((meter_accuracy, tempo_error))
    mean_score = np.mean(metrics, 0)
    print(mean_score[0])
