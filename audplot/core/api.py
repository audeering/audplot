import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import audmetric


def cepstrum(
        cc_matrix: np.ndarray,
        duration: float,
        *,
        channel: int = 0,
        num_ticks: int = 10,
        ax: plt.Axes = None,
        cmap: str = 'magma',
):
    r"""Cepstrum.

    Args:
        cc_matrix: matrix with magnitude values
        duration: duration in seconds
        channel: channel index
        num_ticks: number of ticks on x axis
        ax: axes in which to draw the plot
        cmap: color map

    Example:
        .. plot::
            :context: reset
            :include-source: false

            from audplot import cepstrum

        .. plot::
            :context: close-figs

            >>> import librosa
            >>> import matplotlib.pyplot as plt
            >>> x, sr = librosa.load(librosa.ex('trumpet'))
            >>> dur = len(x) / sr
            >>> y = librosa.feature.mfcc(x, sr)
            >>> cepstrum(y, dur)
            >>> plt.tight_layout()

    """

    ax = ax or plt.gca()
    cc_matrix = cc_matrix[channel] if cc_matrix.ndim == 3 else cc_matrix

    n_cc, n_cepstra = cc_matrix.shape
    ax.set_yticks(np.arange(n_cc) + 0.5)
    ax.set_yticklabels(np.arange(n_cc))
    ax.set_ylabel("Cepstral Coefficients")

    ts = np.linspace(0, cc_matrix.shape[1], num_ticks)
    ts_sec = ["{:4.2f}".format(i) for i in np.linspace(0, duration, num_ticks)]
    ax.set_xticks(ts)
    ax.set_xticklabels(ts_sec)
    ax.set_xlabel("Time (sec)")

    ax.margins(x=0)
    ax.imshow(
        cc_matrix,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        interpolation='none',
    )


def confusion_matrix(
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
        *,
        labels: typing.Sequence = None,
        percentage: bool = True,
        ax: plt.Axes = None,
):
    r"""Confusion matrix between ground truth vs. predicted labels.

    The confusion matrix is calculated by :mod:`audmetric.confusion_matrix`.

    Args:
        truth: truth values
        prediction: predicted values
        labels: labels to be included in confusion matrix
        percentage: if ``True`` present the confusion matrix
            with percentage values instead of absolute numbers
        ax: axes in which to draw the plot

    Example:
        .. plot::
            :context: reset
            :include-source: false

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
    ax = ax or plt.gca()
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
        ax=ax,
    )
    ax.tick_params(axis='y', rotation=0)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Truth')


def distribution(
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
        *,
        ax: plt.Axes = None,
):
    r"""Distribution of truth and predicted values.

    Args:
        truth: truth values
        prediction: predicted values
        ax: axes in which to draw the plot

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
    ax = ax or plt.gca()
    sns.distplot(truth, axlabel='', ax=ax)
    sns.distplot(prediction, axlabel='', ax=ax)
    ax.legend(['Truth', 'Prediction'])


def scatter(
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
        *,
        ax: plt.Axes = None,
):
    r"""Scatter plot of truth and predicted values.

    Args:
        truth: truth values
        prediction: predicted values
        ax: axes in which to draw the plot

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
    ax = ax or plt.gca()
    minimum = min(truth + prediction)
    maximum = max(truth + prediction)
    ax.scatter(truth, prediction)
    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        color='r',
    )
    ax.set_xlim(minimum, maximum)
    ax.set_ylim(minimum, maximum)
    ax.set_xlabel('Truth')
    ax.set_ylabel('Prediction')


def series(
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
        *,
        ax: plt.Axes = None,
):
    r"""Time series plot of truth and predicted values.

    Args:
        truth: truth values
        prediction: predicted values
        ax: axes in which to draw the plot

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
    ax = ax or plt.gca()
    minimum = min(truth + prediction)
    maximum = max(truth + prediction)
    ax.plot(truth)
    ax.plot(prediction)
    ax.set_ylim(minimum, maximum)
    ax.legend(['Truth', 'Prediction'])


def signal(
        x: np.ndarray,
        duration: float,
        *,
        channel: int = 0,
        num_ticks: int = 10,
        ax: plt.Axes = None,
):
    r"""Time signal.

    Args:
        x: array with signal values
        duration: duration in seconds
        channel: channel index
        num_ticks: number of ticks on x and y axis
        ax: axes to plot on

    Example:
        .. plot::
            :context: reset
            :include-source: false

            from audplot import signal

        .. plot::
            :context: close-figs

            >>> import librosa
            >>> x, sr = librosa.load(librosa.ex('trumpet'))
            >>> dur = len(x) / sr
            >>> signal(x, dur)

    """
    ax = ax or plt.gca()
    x = x[channel] if x.ndim == 2 else x

    ts = np.linspace(0, x.size, num_ticks)
    ts_sec = ["{:4.2f}".format(i) for i in np.linspace(0, duration, num_ticks)]
    ax.set_xticks(ts)
    ax.set_xticklabels(ts_sec)
    ax.set_xlabel("Time (sec)")

    ax.margins(x=0)
    ax.plot(x)


def spectrum(
        magnitude: np.ndarray,
        duration: float,
        centers: np.ndarray,
        *,
        channel: int = 0,
        num_ticks: int = 10,
        ax: plt.Axes = None,
        cmap: str = 'magma',
):
    r"""Plot spectrum.

    Args:
        magnitude: matrix with magnitude values
        duration: duration in seconds
        centers: array with center frequencies
        channel: channel index
        num_ticks: number of ticks on x and y axis
        ax: axes to plot on
        cmap: color map

    Example:
        .. plot::
            :context: reset
            :include-source: false

            from audplot import spectrum
            import numpy as np

        .. plot::
            :context: close-figs

            >>> import librosa
            >>> import matplotlib.pyplot as plt
            >>> x, sr = librosa.load(librosa.ex('trumpet'))
            >>> dur = len(x) / sr
            >>> y = librosa.feature.melspectrogram(x, sr, n_mels=40, fmax=4000)
            >>> y_db = librosa.power_to_db(y, ref=np.max)
            >>> centers = librosa.mel_frequencies(n_mels=40, fmax=4000)
            >>> spectrum(y_db, dur, centers)
            >>> plt.tight_layout()

    """
    ax = ax or plt.gca()
    magnitude = magnitude[channel] if magnitude.ndim == 3 else magnitude

    centers = np.round(centers, 2)
    idx = np.round(np.linspace(0, len(centers) - 1, num_ticks)).astype(int)
    ax.set_yticks(idx)
    ax.set_yticklabels(centers[idx])
    ax.set_ylabel("Frequency (Hz)")

    ts = np.linspace(0, magnitude.shape[1], num_ticks)
    ts_sec = ["{:4.2f}".format(i) for i in np.linspace(0, duration, num_ticks)]
    ax.set_xticks(ts)
    ax.set_xticklabels(ts_sec)
    ax.set_xlabel("Time (sec)")

    ax.margins(x=0)
    ax.imshow(
        magnitude,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        interpolation='none',
    )
