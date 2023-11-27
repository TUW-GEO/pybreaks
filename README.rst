========
pybreaks
========

.. image:: https://github.com/TUW-GEO/pybreaks/workflows/Automated%20Tests/badge.svg?branch=master
   :target: https://github.com/TUW-GEO/pybreaks/actions

Detection and correction of structural breaks in (climate) time series.

Description
===========

This package currently includes methods to test for inhomogeneities in satellite soil moisture measurements.
In also contains implementations for correcting detected breaks, currently there are three methods:

- Linear Model Pair matching
- Higher Order Moment adjustment (see also https://doi.org/10.1175/JCLI3855.1)
- Quantile Category Matching (see also https://doi.org/10.1175/2010JAMC2376.1)

The main modules in the package are:

- ``TsRelBreakTest`` : Implements relative statistical tests between two subperiods to detect a break between them.
- ``RegressPairFit``, ``HigherOrderMoments``, ``QuantileCatMatch`` : Classes that implement the correction methods.
- ``TsRelBreakAdjust`` : Combines the detection and correction methods to correct a break between two subperiods.
- ``TsRelMultiBreak`` : Iterates over multiples breaks in a time series to correct them.
  
Documentation
==============

Methods in this package are described in

   "W. Preimesberger, T. Scanlon, C. -H. Su, A. Gruber and W. Dorigo, "Homogenization of Structural Breaks in the Global ESA CCI Soil Moisture Multisatellite Climate Data Record," in IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 4, pp. 2845-2862, April 2021, doi: 10.1109/TGRS.2020.3012896."

Note
====

This project has been set up using PyScaffold 2.5.9. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
