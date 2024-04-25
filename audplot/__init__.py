from audplot.core.api import cepstrum
from audplot.core.api import confusion_matrix
from audplot.core.api import detection_error_tradeoff
from audplot.core.api import distribution
from audplot.core.api import human_format
from audplot.core.api import scatter
from audplot.core.api import series
from audplot.core.api import signal
from audplot.core.api import spectrum
from audplot.core.api import waveform


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
