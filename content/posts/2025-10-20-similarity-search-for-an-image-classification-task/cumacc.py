# Copyright (C) 2025 Igor Sivchek
# Licensed under the MIT License.
# See license text at [https://opensource.org/license/mit].

"""
Cumulative accuracy function.
This metric may be valuable for multiclass classification problems.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

__all__ = ["cumulative_accuracy"]


# It is like the cumulative distribution function in probability theory.
# x-axis - the share of classes or the number of classes.
# y-axis - accuracy.
# At a particular share of classes, the accuracy is not greater than the
# respective value.
# In an ideal case, it is a horizontal line with a value of 1.0.
def cumulative_accuracy(
    *,
    accuracies: np.ndarray,
    num_intervals: int = 10
) -> tuple[np.ndarray]:
    if num_intervals < 1:
        raise ValueError(
            "The number of intervals must be not less than 1, but given: {0}"
                .format(num_intervals)
        )
    num_pts = num_intervals + 1
    sorted_accs = np.sort(accuracies)
    shares = np.linspace(start=0.0, stop=1.0, num=num_pts)
    pts_per_interv = len(sorted_accs) / num_intervals
    cum_accs = np.zeros(num_pts, dtype=np.float32)
    for idx in range(1, num_pts):
        cum_num_pts = int(idx * pts_per_interv)
        acc_idx = (cum_num_pts - 1 if cum_num_pts > 0 else 0)
        cum_accs[idx] = sorted_accs[acc_idx]
    return cum_accs, shares


if __name__ == '__main__':
    labels = np.array([2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    # accuracies = np.array([0.25, 0.5, 1.0, 0.75, 0.8, 0.4, 0.6, 0.65, 0.9, 0.85, 0.95, 0.7, 0.55, 0.2, 0.8, 0.6, 0.85, 0.9, 0.95, 0.7, 0.45, 0.8, 0.95, 0.75, 0.1], dtype=np.float32)
    accuracies = np.array([0.2, 0.5, 0.75, 0.8, 0.4, 0.6, 0.65, 0.9, 0.85, 0.95, 0.7, 0.55, 0.2, 0.8, 0.6, 0.85, 0.9, 0.95, 0.7, 0.45, 0.8, 0.95, 0.75, 0.1], dtype=np.float32)

    cum_accs, shares = cumulative_accuracy(accuracies=accuracies)

    shares_cs = np.linspace(shares[0], shares[-1], 100)
    cum_accs_cs = scipy.interpolate.CubicSpline(shares, cum_accs)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), constrained_layout=True)
    fig.suptitle("Cumulative accuracy function")
    ax.plot(shares, cum_accs)
    ax.plot(shares_cs, cum_accs_cs(shares_cs))
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("share")
    ax.set_ylabel("accuracy", rotation="horizontal")
    ax.yaxis.set_label_coords(0.0, 1.05)
    ax.grid()
    fig.show()
