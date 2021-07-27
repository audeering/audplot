import pytest

import audplot


def test_human_format():
    with pytest.raises(ValueError):
        audplot.human_format(1000 ** 9)
