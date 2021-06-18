import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import audmetric


def confusion_matrix(
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
        *,
        labels: typing.Sequence = None,
        percentage: bool = True,
):
    r"""Confusion matrix between ground truth vs. predicted labels.

    The confusion matrx is calculated by :mod:`audmetric.confusion_matrix`.

    Args:
        truth: truth values
        prediction: predicted values
        labels: labels to be included in confusion matrix
        percentage: if ``True`` present the confusion matrix
            with percentage values instead of absolute numbers

    Example:
        .. plot::
            :context: reset
            :include-source: false

            import matplotlib.pyplot as plt
            from audplot import confusion_matrix

        .. plot::
            :context: close-figs

            >>> truth = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
            >>> prediction = ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'C']
            >>> confusion_matrix(truth, prediction, percentage=False)

        .. plot::
            :context: close-figs

            >>> confusion_matrix(truth, prediction, labels=['A', 'B', 'C', 'D'])

    """  # noqa: 501
    sns.set()  # get prettier plots

    labels = audmetric.core.utils.infer_labels(truth, prediction, labels)

    cm = audmetric.confusion_matrix(
        truth,
        prediction,
        labels=labels,
        normalize=percentage,
    )
    if percentage:
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
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
):
    r"""Distribution of truth and predicted values.

    Args:
        truth: truth values
        prediction: predicted values

    Example:
        .. plot::
            :context: reset
            :include-source: false

            import pandas as pd
            from audplot import distribution

        .. plot::
            :context: close-figs

            >>> truth = pd.Series([0, 1, 1, 2])
            >>> prediction = pd.Series([0, 1, 2, 2])
            >>> distribution(truth, prediction)

    """
    sns.distplot(truth, axlabel='')
    sns.distplot(prediction, axlabel='')
    plt.legend(['truth', 'prediction'])


def scatter(
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
):
    r"""Scatter plot of truth and predicted values.

    Args:
        truth: truth values
        prediction: predicted values

    Example:
        .. plot::
            :context: reset
            :include-source: false

            from audplot import scatter

        .. plot::
            :context: close-figs

            >>> truth = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> prediction = [0.1, 0.8, 2.3, 2.4, 3.9, 5, 6.2, 7.1, 7.8, 9, 9]
            >>> scatter(truth, prediction)

    """
    minimum = min(truth + prediction)
    maximum = max(truth + prediction)
    plt.scatter(truth, prediction)
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
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
):
    r"""Time series plot of truth and predicted values.

    Args:
        truth: truth values
        prediction: predicted values

    Example:
        .. plot::
            :context: reset
            :include-source: false

            from audplot import series

        .. plot::
            :context: close-figs

            >>> truth = [-1, 0, 1, 0, -1, 0, 1]
            >>> prediction = [0, 1, 0, -1, 0, 1, 0]
            >>> series(truth, prediction)

    """
    minimum = min(truth + prediction)
    maximum = max(truth + prediction)
    plt.plot(truth)
    plt.plot(prediction)
    plt.ylim(minimum, maximum)
    plt.legend(['truth', 'prediction'])
