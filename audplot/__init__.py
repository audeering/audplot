from audplot.core.api import (
    cepstrum,
    confusion_matrix,
    detection_error_tradeoff,
    distribution,
    human_format,
    scatter,
    series,
    signal,
    spectrum,
    waveform,
)


# Disencourage from audfoo import *
__all__ = []


# Dynamically get the version of the installed module
try:
    import importlib.metadata
    __version__ = importlib.metadata.version(__name__)
except Exception:  # pragma: no cover
    importlib = None  # pragma: no cover
finally:
    del importlib
