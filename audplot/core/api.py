import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import audmetric


def confusion_matrix(
        truth: pd.Series,
        prediction: pd.Series,
        *,
        labels: typing.Sequence[typing.Any] = None,
        percentage: bool = True,
) -> None:  # pragma: no cover
    r"""Confusion matrix between ground truth vs. predicted labels.

    Args:
        truth: truth values
        prediction: predicted values
        percentage: if ``True`` present the confusion matrix
            with percentage values instead of absolute numbers

    """
    sns.set()  # get prettier plots

    labels = audmetric.core.utils.infer_labels(labels)

    cm = audmetric.confusion_matrix(
        truth.values,
        prediction.values,
        labels=labels,
        normalize=percentage,
    )
    if percentage:
        for idx, row in enumerate(cm):
            if np.sum(row) == 0:
                cm[idx] = np.NaN
            else:
                cm[idx] /= np.sum(row)
        fmt = '.0%'
    else:
        fmt = 'd'
    sns.heatmap(
        cm,
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        fmt=fmt,
        cmap="Blues",
    )
    plt.yticks(rotation=0)
    plt.xlabel('prediction')
    plt.ylabel('truth')
    plt.tight_layout()


def distribution(
        truth: pd.Series,
        prediction: pd.Series,
) -> None:  # pragma: no cover
    r"""Distribution of truth and predicted labels.

    Args:
        truth: truth values
        prediction: predicted values

    """
    sns.distplot(truth, axlabel='')
    sns.distplot(prediction, axlabel='')
    plt.legend(['truth', 'prediction'])


def scatter(
        truth: pd.Series,
        prediction: pd.Series,
) -> None:
    minimum = min(truth + prediction)
    maximum = max(truth + prediction)
    plt.scatter(truth.values, prediction.values)
    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        color='r',
    )
    plt.xlim(minimum, maximum)
    plt.ylim(minimum, maximum)
    plt.xlabel('truth')
    plt.ylabel('prediction')


def series(
        truth: pd.Series,
        prediction: pd.Series,
) -> None:
    minimum = min(truth + prediction)
    maximum = max(truth + prediction)
    plt.plot(truth.values)
    plt.plot(prediction.values)
    plt.ylim(minimum, maximum)
    plt.legend(['truth', 'prediction'])
