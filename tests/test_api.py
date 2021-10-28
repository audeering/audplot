import numpy as np
import pytest

import audplot


@pytest.mark.parametrize(
    'number, expected_string',
    [
        (0, '0'),
        (1, '1'),
        (10, '10'),
        (100, '100'),
        (1000, '1k'),
        (10000, '10k'),
        (100000, '100k'),
        (1000000, '1M'),
        (0.1, '100m'),
        (0.01, '10m'),
        (0.001, '1m'),
        (0.0015, '1.5m'),
        (0.0001, '100u'),
        (-1, '-1'),
        (-0.001, '-1m'),
        (-0.0015, '-1.5m'),
        pytest.param(
            1000 ** 9,
            '',
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            1000 ** -4,
            '',
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_human_format(number, expected_string):
    string = audplot.human_format(number)
    assert string == expected_string


def test_waveform():
    # Fail for non mono signals
    with pytest.raises(RuntimeError):
        x = np.ones((2, 100))
        audplot.waveform(x)
