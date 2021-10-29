import math
import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# The scipy implementation is faster,
# but scipy is not an official dependency of audplot
try:
    from scipy.special import ndtri as inverse_normal_distribution
except ModuleNotFoundError:  # pragma: nocover
    from audmath import inverse_normal_distribution

import audmetric


def cepstrum(
        cc_matrix: np.ndarray,
        hop_duration: float,
        *,
        channel: int = 0,
        ax: plt.Axes = None,
        cmap: str = 'magma',
) -> matplotlib.image.AxesImage:
    r"""Cepstrum.

    Args:
        cc_matrix: cepstral coefficients matrix with magnitude values
        hop_duration: hop duration in seconds
        channel: channel index
        ax: axes in which to draw the plot
        cmap: color map

    Returns:
        Image object

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
            >>> y = librosa.feature.mfcc(x, sr)
            >>> hop_dur = 512 / sr  # default hop length is 512
            >>> image = cepstrum(y, hop_dur)
            >>> _ = plt.colorbar(image)
            >>> plt.tight_layout()

    """

    ax = ax or plt.gca()
    cc_matrix = cc_matrix[channel] if cc_matrix.ndim == 3 else cc_matrix

    n_cc, n_cepstra = cc_matrix.shape
    extent = [0, (n_cepstra - 1) * hop_duration, -0.5, n_cc - 0.5]
    ax.set_ylabel('Cepstral Coefficients')
    ax.set_xlabel('Time / s')

    ax.margins(x=0)
    image = ax.imshow(
        cc_matrix,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        interpolation='none',
        extent=extent,
    )

    # Adjust yticks to be located at real cepstral coefficient steps
    locs = ax.get_yticks()
    yticks_spacing = int(np.round(n_cc / len(locs)))
    locs = list(range(0, n_cc - 1, yticks_spacing))
    ax.set_yticks(locs)

    return image


def confusion_matrix(
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
        *,
        labels: typing.Sequence = None,
        percentage: bool = False,
        show_both: bool = False,
        ax: plt.Axes = None,
):
    r"""Confusion matrix between ground truth and prediction.

    The confusion matrix is calculated by :mod:`audmetric.confusion_matrix`.

    Args:
        truth: truth values
        prediction: predicted values
        labels: labels to be included in confusion matrix
        percentage: if ``True`` present the confusion matrix
            with percentage values instead of absolute numbers
        show_both: if ``True`` and percentage is ``True``
            it shows absolute numbers in brackets 
            below percentage values.
            If ``True`` and percentage is ``False``
            it shows the percentage in brackets
            below absolute numbers
        ax: axes in which to draw the plot

    Example:
        .. plot::
            :context: reset
            :include-source: false

            from audplot import confusion_matrix

        .. plot::
            :context: close-figs

            >>> truth = ['A', 'B', 'B', 'B', 'C', 'C', 'C'] * 1000
            >>> prediction = ['A', 'B', 'C', 'C', 'A', 'A', 'C'] * 1000
            >>> confusion_matrix(truth, prediction)

        .. plot::
            :context: close-figs

            >>> confusion_matrix(truth, prediction, percentage=True)

        .. plot::
            :context: close-figs

            >>> confusion_matrix(truth, prediction, show_both=True)

        .. plot::
            :context: close-figs

            >>> confusion_matrix(truth, prediction, percentage=True, show_both=True)

        .. plot::
            :context: close-figs

            >>> confusion_matrix(truth, prediction, labels=['A', 'B', 'C', 'D'])

    """  # noqa: 501
    ax = ax or plt.gca()
    if labels is None:
        labels = audmetric.utils.infer_labels(truth, prediction)

    cm = audmetric.confusion_matrix(
        truth,
        prediction,
        labels=labels,
        normalize=percentage,
    )
    cm = pd.DataFrame(cm, index=labels)

    # Set format of first row labels in confusion matrix
    if percentage:
        annot = cm.applymap(lambda x: f'{100 * x:.0f}%')
    else:
        annot = cm.applymap(lambda x: human_format(x))

    # Add a second row of annotations if requested
    if show_both:
        cm2 = audmetric.confusion_matrix(
            truth,
            prediction,
            labels=labels,
            normalize=not percentage,
        )
        cm2 = pd.DataFrame(cm2, index=labels)
        if percentage:
            annot2 = cm2.applymap(lambda x: human_format(x))
        else:
            annot2 = cm2.applymap(lambda x: f'{100 * x:.0f}%')

        # Combine strings from two dataframes
        # by vectorizing the underlying function.
        # See: https://stackoverflow.com/a/42277839

        def combine_string(x, y):
            return f'{x}\n({y})'

        combine_string = np.vectorize(combine_string)
        annot = pd.DataFrame(combine_string(annot, annot2), index=labels)

    sns.heatmap(
        cm,
        annot=annot,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        fmt='',
        cmap='Blues',
        ax=ax,
    )
    ax.tick_params(axis='y', rotation=0)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Truth')


def detection_error_tradeoff(
        x: typing.Union[typing.Sequence, pd.Series],
        y: typing.Union[typing.Sequence, pd.Series],
        *,
        error_rates: bool = False,
        xlim: typing.Sequence = [0.001, 0.5],
        ylim: typing.Sequence = [0.001, 0.5],
        label: str = None,
        ax: plt.Axes = None,
) -> typing.Callable:
    r"""Detection error tradeoff curve.

    A `detection error tradeoff (DET)`_ curve
    is a graphical plot of error rates for binary classification systems,
    plotting the false non-match rate (FNMR)
    against the false match rate (FMR).

    You can provide truth and prediction values
    as input or you can directly provide FMR and FNMR,
    which can be calculated using
    :func:`audmetric.detection_error_tradeoff`.

    The axes of the plot are scaled non-linearly
    by their `standard normal deviates`_.
    This means you have to scale every value
    by this transformation
    when you would like to change ticks positions
    or axis limits afterwards.
    The scaling is performed by :func:`scipy.special.ndtri`
    if :mod:`scipy` is installed,
    otherwise :func:`audmath.inverse_standard_distribution` is used,
    which is slower for large input arrays.


    .. _detection error tradeoff (DET): https://en.wikipedia.org/wiki/Detection_error_tradeoff
    .. _standard normal deviates: https://en.wikipedia.org/wiki/Standard_normal_deviate

    Args:
        x: truth values or false match rate (FMR)
        y: predicted values or false non-match rate (FNMR)
        error_rates: if ``False``
            it expects truth values as ``x``,
            and prediction values as ``y``.
            If ``True`` it expects FMR as ``x``,
            and FNMR as ``y``
        xlim: x-axis limits with :math:`x \in ]0, 1[`
        ylim: y-axis limits with :math:`y \in ]0, 1[`
        label: label to be shown in the legend.
            The legend will not be shown automatically
        ax: axes in which to draw the plot

    Returns:
        function to transform input values to standard normal derivate scale

    Example:
        .. plot::
            :context: reset
            :include-source: false

            import matplotlib.pyplot as plt
            import numpy as np
            from audplot import detection_error_tradeoff

            np.random.seed(0)

        .. plot::
            :context: close-figs

            >>> truth = np.array([1] * 1000 + [0] * 1000)
            >>> # Random prediction
            >>> pred1 = np.random.random_sample(2000)
            >>> # Better than random prediction
            >>> pred2 = np.zeros(2000,)
            >>> pred2[:1000] = np.random.normal(loc=0.6, scale=0.1, size=1000)
            >>> pred2[1000:] = np.random.normal(loc=0.4, scale=0.1, size=1000)
            >>> pred2 = np.clip(pred2, 0, 1)
            >>> transform = detection_error_tradeoff(
            ...     truth,
            ...     pred1,
            ...     xlim=[0.01, 0.99],  # use large limits for random
            ...     ylim=[0.01, 0.99],
            ...     label='pred1',
            ... )
            >>> # Add pred2 to plot using transformed FMR and FNMR values
            >>> import audmetric
            >>> fmr, fnmr, _ = audmetric.detection_error_tradeoff(truth, pred2)
            >>> _ = plt.plot(transform(fmr), transform(fnmr), label='pred2')
            >>> _ = plt.legend()
            >>> plt.tight_layout()

    """  # noqa: E501
    if not error_rates:
        x, y, _ = audmetric.detection_error_tradeoff(x, y)

    # Transform values to the normal derivate scale
    transform = inverse_normal_distribution

    g = sns.lineplot(
        x=transform(x),
        y=transform(y),
        label=label,
    )
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.xlabel('False Match Rate')
    plt.ylabel('False Non-Match Rate')
    plt.grid(alpha=0.4)

    ticks = [0.001, 0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]
    tick_locations = transform(ticks)
    tick_labels = [
        f'{t:.0%}' if (100 * t).is_integer() else f'{t:.1%}'
        for t in ticks
    ]
    g.set(xticks=tick_locations, xticklabels=tick_labels)
    g.set(yticks=tick_locations, yticklabels=tick_labels)
    plt.xlim(transform(xlim[0]), transform(xlim[1]))
    plt.ylim(transform(ylim[0]), transform(ylim[1]))

    sns.despine()

    return transform


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


def human_format(
        number: typing.Union[int, float],
) -> str:
    r"""Display large or small numbers in a human readable way.

    It replaces large or small numbers
    by no more than 3 significant digits
    and no more than 1 fractional digit.
    Instead it adds a string indicating the base,
    e.g. 12345 becomes 12.3k.

    The naming is according to:

    .. table::
        :widths: 10 10 15 12

        = =============== =========== =====
        n :math:`10^{-9}` nano
        u :math:`10^{-6}` micro
        m :math:`10^{-3}` milli
        k :math:`10^{3}`  thousand
        M :math:`10^{6}`  Million     Mega
        B :math:`10^{9}`  Billion     Giga
        T :math:`10^{12}` Trillion    Tera
        P :math:`10^{15}` Quadrillion Peta
        E :math:`10^{18}` Quintillion Exa
        Z :math:`10^{21}` Sextillion  Zetta
        Y :math:`10^{24}` Septillion  Yotta
        = =============== =========== =====

    Args:
        number: input number

    Returns:
        formatted number string

    Raises:
        ValueError: if ``number`` :math:`\ge 1000^9`
            or ``number`` :math:`\le 1000^{-4}`

    Example:
        >>> human_format(12345)
        '12.3k'
        >>> human_format(1234567)
        '1.2M'
        >>> human_format(123456789000)
        '123B'
        >>> human_format(0.000123)
        '123u'
        >>> human_format(0)
        '0'
        >>> human_format(-1000)
        '-1k'

    """
    sign = ''
    if number == 0:
        return '0'
    if number < 0:
        sign = '-'
        number = -1 * number
    units = [
        'n',  # 10^-9  nano
        'u',  # 10^-6  micro
        'm',  # 10^-3  milli
        '',   # 0
        'k',  # 10^3   thousand
        'M',  # 10^6   Million      Mega
        'B',  # 10^9   Billion      Giga
        'T',  # 10^12  Trillion     Tera
        'P',  # 10^15  Quadrillion  Peta
        'E',  # 10^18  Quintillion  Exa
        'Z',  # 10^21  Sextillion   Zetta
        'Y',  # 10^24  Septillion   Yotta
    ]
    k = 1000.0
    magnitude = int(math.floor(math.log(number, k)))
    number = f'{number / k**magnitude:.1f}'
    if magnitude >= 9:
        raise ValueError('Only magnitudes < 1000 ** 9 are supported.')
    if magnitude <= -4:
        raise ValueError('Only magnitudes > 1000 ** -4 are supported.')
    # Make sure we show only up to 3 significant digits
    if len(number) > 4:
        number = number[:-2]
    if number.endswith('.0'):
        number = number[:-2]
    return f'{sign}{number}{units[magnitude + 3]}'


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
    minimum = min([min(truth), min(prediction)])
    maximum = max([max(truth), max(prediction)])
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
    minimum = min([min(truth), min(prediction)])
    maximum = max([max(truth), max(prediction)])
    ax.plot(truth)
    ax.plot(prediction)
    ax.set_ylim(minimum, maximum)
    ax.legend(['Truth', 'Prediction'])


def signal(
        x: np.ndarray,
        sampling_rate: float,
        *,
        channel: int = 0,
        ax: plt.Axes = None,
):
    r"""Time signal.

    Args:
        x: array with signal values
        sampling_rate: sampling rate in Hz
        channel: channel index
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
            >>> signal(x, sr)

    """
    ax = ax or plt.gca()
    x = x[channel] if x.ndim == 2 else x

    time = np.arange(len(x)) / sampling_rate
    ax.set_xlabel('Time / s')

    ax.margins(x=0)
    ax.plot(time, x)


def spectrum(
        magnitude: np.ndarray,
        hop_duration: float,
        centers: np.ndarray,
        *,
        channel: int = 0,
        ax: plt.Axes = None,
        cmap: str = 'magma',
) -> matplotlib.image.AxesImage:
    r"""Plot spectrum.

    Args:
        magnitude: matrix with magnitude values
        hop_duration: hop duration in seconds
        centers: array with center frequencies
        channel: channel index
        ax: axes to plot on
        cmap: color map

    Returns:
        Image object

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
            >>> y = librosa.feature.melspectrogram(x, sr, n_mels=40, fmax=4000)
            >>> y_db = librosa.power_to_db(y, ref=np.max)
            >>> hop_dur = 512 / sr  # default hop length is 512
            >>> centers = librosa.mel_frequencies(n_mels=40, fmax=4000)
            >>> image = spectrum(y_db, hop_dur, centers)
            >>> _ = plt.colorbar(image, format='%+2.0f dB')
            >>> plt.tight_layout()

    """
    ax = ax or plt.gca()
    magnitude = magnitude[channel] if magnitude.ndim == 3 else magnitude

    frequencies, times = magnitude.shape
    extent = [0, times * hop_duration, -0.5, frequencies - 0.5]
    ax.set_ylabel('Frequency / Hz')
    ax.set_xlabel('Time / s')

    ax.margins(x=0)
    image = ax.imshow(
        magnitude,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        interpolation='none',
        extent=extent,
    )

    # Add center frequencies as yticks labels
    locs = ax.get_yticks()
    yticks_spacing = int(locs[1] - locs[0])
    # It might return negative yticks,
    # that are not visible
    negative_locs = locs[locs < 0]
    labels = ['' for _ in negative_locs]
    labels += [f'{cf:.1f}' for cf in centers[::yticks_spacing]]
    ax.set_yticklabels(labels)

    return image
