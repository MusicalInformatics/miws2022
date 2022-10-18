#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper methods for plotting and visualizing alignments
"""
import numpy as np

from matplotlib import lines, pyplot as plt
from sklearn.datasets import make_blobs

# Define random state for reproducibility
RNG = np.random.RandomState(1984)


def generate_example_sequences(
    lenX: int = 100,
    centers: int = 3,
    n_features: int = 5,
    maxreps: int = 4,
    minreps: int = 1,
    noise_scale: float = 0.01,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Generates example pairs of related sequences. Sequence X are samples of
    an K-dimensional space around a specified number of centroids.
    Sequence Y is a non-constant "time-streched" version of X with some
    noise added.

    Parameters
    ----------
    lenX : int
        Number of elements in the X sequence
    centers: int
        Number of different centers ("classes") that the elements
        of the sequences represent
    n_features: int
        Dimensionality of the features ($K$) in the notation of the
        Notebook
    noise_scale: float
        Scale of the noise

    Returns
    -------
    X : np.ndarray
        Sequence X (a matrix where each row represents
        an element of the sequence)
    Y: np.ndarray
        Sequence Y
    ground_truth_path: np.ndarray
        Alignment between X and Y where the first column represents the indices
        in X and the second column represents the corresponding index in Y.
    """

    X, _ = make_blobs(n_samples=lenX, centers=centers, n_features=n_features)
    # Time stretching X! each element in sequence X is
    # repeated a random number of times
    # and then we add some noise to spice things up :)
    n_reps = RNG.randint(minreps, maxreps, len(X))
    y_idxs = [rp * [i] for i, rp in enumerate(n_reps)]
    y_idxs = np.array([el for reps in y_idxs for el in reps], dtype=int)
    # Add a bias, so that Y has a different "scaling" than X
    Y = X[y_idxs]
    # add some noise
    Y += noise_scale * RNG.randn(*Y.shape)
    ground_truth_path = np.column_stack((y_idxs, np.arange(len(Y))))
    return X, Y, ground_truth_path


def plot_alignment(
    X: np.ndarray,
    Y: np.ndarray,
    alignment_path: np.ndarray,
) -> None:
    """
    Visualize alignment between two sequences.

    Parameters
    ----------
    X : np.ndarray
        Reference sequence (a matrix where each row represents an element of
        the sequence)
    Y : np.ndarray
        The sequence we want to align to X.
    alignment_path : np.ndarray
        A 2D array where each row corresponds to the indices in array X and its
        corresponding element in X.
    """
    vmax = max(max(abs(X.max()), abs(X.min())), max(abs(Y.max()), abs(Y.min())))
    fig, axes = plt.subplots(2, sharex=True)
    axes[0].imshow(
        X.T,
        cmap="gray",
        origin="lower",
        aspect="equal",
        interpolation=None,
        vmax=vmax,
        vmin=-vmax,
    )
    axes[0].set_ylabel(r"$\mathbf{X}$")
    axes[1].imshow(
        Y.T,
        cmap="gray",
        origin="lower",
        aspect="equal",
        interpolation=None,
        vmax=vmax,
        vmin=-vmax,
    )
    axes[1].set_ylabel(r"$\mathbf{Y}$")
    axes[0].set_xlim((-1, max(len(X), len(Y)) + 1))
    axes[1].set_xlim((-1, max(len(X), len(Y)) + 1))
    axes[0].set_ylim((-1, X.shape[1] + 1))
    axes[1].set_ylim((-1, Y.shape[1] + 1))

    axis_off = True

    if axis_off:
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["bottom"].set_visible(False)
        axes[0].spines["left"].set_visible(False)
        axes[0].get_xaxis().set_ticks([])
        axes[0].get_yaxis().set_ticks([])
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["bottom"].set_visible(False)
        axes[1].spines["left"].set_visible(False)
        axes[1].get_xaxis().set_ticks([])
        axes[1].get_yaxis().set_ticks([])

    for ref_idx, perf_idx in alignment_path:
        # Add line from one subplot to the other
        xyA = [ref_idx, 0]
        axes[0].plot(*xyA)
        xyB = [perf_idx, Y.shape[1] - 0.75]
        axes[1].plot(*xyB)
        transFigure = fig.transFigure.inverted()
        coord1 = transFigure.transform(axes[0].transData.transform(xyA))
        coord2 = transFigure.transform(axes[1].transData.transform(xyB))
        line = lines.Line2D(
            (coord1[0], coord2[0]),  # xdata
            (coord1[1], coord2[1]),  # ydata
            transform=fig.transFigure,
            color="red",
            linewidth=0.5,
        )
        fig.lines.append(line)

    plt.show()
