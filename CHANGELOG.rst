Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 1.3.0 (2021-11-03)
--------------------------

* Added: ``audplot.waveform()``
* Added: ``label_aliases`` argument to ``audplot.confusion_matrix()``
* Changed: show grid lines and remove top ad right axes as default
* Changed: Use ``seaborn.histplot()`` in ``audplot.distribution()``
* Fixed: xticks position for time axes


Version 1.2.0 (2021-07-30)
--------------------------

* Added: ``audplot.detection_error_tradeoff()``


Version 1.1.0 (2021-07-27)
--------------------------

* Added: ``audplot.human_format()``
* Added: ``show_both`` argument to ``audplot.confusion_matrix()``
  which allows showing percentage and absolute numbers
  in the same figure
* Changed: switch from ``True`` to ``False`` as default
  for ``percentage`` argument of ``audplot.confusion_matrix()``


Version 1.0.3 (2021-07-22)
--------------------------

* Fixed: install missing ``libsndfile1`` when publishing docs
* Fixed: calculate minimum and maximum in ``scatter()`` and ``series()``


Version 1.0.2 (2021-07-13)
--------------------------

* Added: ``cepstrum()``, ``signal()``, ``spectrum()``


Version 1.0.1 (2021-07-05)
--------------------------

* Fixed: URLs to documentation and source code inside Python package


Version 1.0.0 (2021-06-28)
--------------------------

* Added: initial release


.. _Keep a Changelog:
    https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning:
    https://semver.org/spec/v2.0.0.html
