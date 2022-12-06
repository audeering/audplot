Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 1.4.5 (2022-12-06)
--------------------------

* Fixed: x-axis scaling for ``audplot.waveform()``.
  x-axis values now always correspond
  to the actual number of samples
  of the signal


Version 1.4.4 (2022-12-02)
--------------------------

* Added: support for Python 3.10
* Changed: increase speed of ``audplot.waveform()``
  by factor >100 for long signals


Version 1.4.3 (2022-10-14)
--------------------------

* Fixed: require ``matplotlib!=3.6.1``
  to avoid failing of ``audplot.distribution()``


Version 1.4.2 (2022-01-07)
--------------------------

* Fixed: make ``audplot.scatter(..., fit=True)`` plots reproducible
  by using a fixed seed for bootstrapping


Version 1.4.1 (2022-01-04)
--------------------------

* Added: Python 3.9 support
* Fixed: contributing text
* Removed: Python 3.6 support


Version 1.4.0 (2021-12-10)
--------------------------

* Added: ``fit`` and ``order`` arguments to ``audplot.scatter()``
* Changed: ``audplot.scatter()`` shows just a scatter plot with default
  arguments


Version 1.3.2 (2021-11-17)
--------------------------

* Changed: show frequency instead of counts on the y-axis
  for ``audplot.distribution()``
* Fixed: ``audplot.distribution()`` bins now independently
  for each distribution,
  which ensures the same distribution
  is always plotted the same way


Version 1.3.1 (2021-11-03)
--------------------------

* Fixed: building public documentation in CI pipeline


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
