#!/usr/bin/env python
# -*- coding: utf-8 -*-

# single script challenge submission template
from typing import Tuple

import numpy as np
import partitura as pt

from partitura.musicanalysis.meter import MultipleAgents

from partitura.utils.misc import PathLike

from meter_estimation_challenge import load_submission, compare_meter_and_tempo

import warnings

warnings.filterwarnings("ignore")

CHORD_SPREAD_TIME = 0.05  # for onset aggregation


def estimate_time(filename: PathLike) -> Tuple[int, float]:
    """
    Estimate tempo, meter (currently only time signature numerator)

    Parameters
    ----------
    note_array : structured array

    Returns
    -------
    meter_numerator: int
        The numerator of the time signature
    tempo: float
        The tempo in beats per minute
    """
    performance = pt.load_performance_midi(filename)
    note_array = performance.note_array()
    onsets_raw = note_array["onset_sec"]

    # aggregate notes in clusters
    aggregated_notes = [(0, 0)]
    for note_on in onsets_raw:
        prev_note_on = aggregated_notes[-1][0]
        prev_note_salience = aggregated_notes[-1][1]
        if abs(note_on - prev_note_on) < CHORD_SPREAD_TIME:
            aggregated_notes[-1] = (note_on, prev_note_salience + 1)
        else:
            aggregated_notes.append((note_on, 1))

    onsets, saliences = list(zip(*aggregated_notes))
    ma = MultipleAgents()
    ma.run(onsets, saliences)

    meter_numerator = ma.getNum()
    tempo = ma.getTempo()

    return meter_numerator, tempo


if __name__ == "__main__":

    import argparse
    import os
    import glob

    # DO NOT CHANGE THIS!
    parser = argparse.ArgumentParser(description="Meter Estimation")
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
        default="meter_estimation.txt",
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
        predicted_meter, predicted_tempo = estimate_time(
            filename=mfn,
        )

        results.append((piece, predicted_meter, predicted_tempo))

        if piece in ground_truth:
            expected_meter, expected_tempo = ground_truth[piece]
            meter_accuracy, tempo_error = compare_meter_and_tempo(
                predicted_meter,
                expected_meter,
                predicted_tempo,
                expected_tempo,
            )
            print(
                f"{i+1}/{len(midi_files)} {piece}: "
                f"\tPredicted:{predicted_meter} {predicted_tempo:.2f}"
                f"\tExpected:{expected_meter} {expected_tempo:.2f}"
                f"\tTempo error:{tempo_error}"
            )
            evaluation.append((meter_accuracy, tempo_error))

    mean_eval = np.mean(evaluation, 0)
    if len(evaluation) > 0:
        print("\n\nAverage Performance over dataset")
        print(f"\tMeter accuracy{mean_eval[0]: .2f}")
        print(f"\tTempo error: {mean_eval[1]: .2f}")

    if args.challenge:
        # Export predictions for the challenge
        np.savetxt(
            args.outfile,
            np.array(results),
            fmt="%s",
            delimiter=",",
            comments="//",
            header="filename,ts_num,tempo(bpm)",
        )
