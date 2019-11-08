========
pybreaks
========

.. image:: https://travis-ci.com/wpreimes/pybreaks.svg?token=EMNRtJMV9D8ioyMpm5R1&branch=master
    :target: https://travis-ci.com/wpreimes/pybreaks

Detection and correction of structural breaks in climate time series.


Description
===========

This package currently includes methods to test for inhomogeneities in satellite soil moisture measurements.
In also contains implemenmtations for correcting detected breaks, currently there are three methods:
  - Linear Model Pair matching
  - Higher Order Moment adjustment (see also https://doi.org/10.1175/JCLI3855.1)
  - Quantile Category Matching (see also https://doi.org/10.1175/2010JAMC2376.1)

The main modules in the package are
    - ``TsRelBreakTest`` : Implements relative statistical tests between two subperiods to detect a break.
    - ``RegressPairFit``, ``HigherOrderMoments``, ``QuantileCatMatch`` : Classes that implement the correction methods
    - ``TsRelBreakAdjust`` : Combine the detection and correction methods to correct a break between two subperiods
    - ``TsRelMultiBreak`` : Iterates over multiples breaks in a time series to correct them.
Documentation
==============

Detailled documentation will follow shortly...

Note
====

This project has been set up using PyScaffold 2.5.9. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
