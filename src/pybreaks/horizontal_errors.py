# -*- coding: utf-8 -*-
"""
Module to calculate error metrics for a CAN and a REF series grouped by a date
(break time) to compare the performance of different parts of the full series.
"""

import pandas as pd
from datetime import datetime
import numpy as np
from scipy import stats
from pytesmo import metrics


def compare(v0, v1, how='AbsDiff'):
    '''
    Takes the horizontal comparison metrics for a candidate set and compares it to the metric
    for a reference data set via the selected function:

    Parameters
    ----------
    v0 : float
        Value 0
    v1 : float
        Value 1
    how : str
        How to compare the 2 numbers

    Returns
    -------
    error : float
        The error from comparing the 2 inputs via the comparison method
    '''

    if np.isnan(v0) or np.isnan(v1):
        return np.nan

    if how == 'Ratio':
        try:
            rat = float(v0) / float(v1)
        except ZeroDivisionError:
            rat = np.nan
        return rat
    elif how == 'Diff':
        return float(v0) - float(v1)
    elif how == 'AbsDiff':
        return np.abs(float(v0) - float(v1))
    else:
        raise Exception('Unknown definition to combine the metrics')


class HorizontalVal(object):
    ''' Create group stats, calc horizontal metrics and horizontal errors'''
    def __init__(self, candidate, reference, breaktime):
        '''
        Initialze the object for horizontal cci_validation.

        Parameters
        ----------
        candidate : pandas.Series
            The candidate series (no bias!)
        reference : pandas.Series
            The reference series (no bias!)
        breaktime : datetime
            Time to split the candidate and reference into 2 sets
        '''
        self.candidate_name = candidate.name
        self.reference_name = reference.name

        self.df = pd.DataFrame(data={self.candidate_name: candidate.copy(True),
                                     self.reference_name: reference.copy(True)})

        self.breaktime = breaktime

        self.set0 = self.df.loc[:breaktime]
        self.set1 = self.df.loc[breaktime:]
        self.setfull = self.df.copy(True)

        self.basic_measures = np.array([('mean', 'Diff'), ('median', 'Diff'),
                                        ('min', 'Diff'), ('max', 'Diff'),
                                        ('var', 'Ratio'), ('iqr', 'Ratio')])

        self.pytesmo_measures = ['bias', 'mad', 'rmsd', 'nrmsd']


    def run(self, comparison_method='Diff'):
        '''
        Calculate the measures and compare them over the break time

        Parameters
        ----------
        comparison_method

        Returns
        -------

        '''
        self.group_stats = self._group_stats()
        self.vertical_metrics = self._vertical_metrics()

        self.horizontal_errors = self._horizontal_metrics(how=comparison_method)

        return self.horizontal_errors


    def _get_group_stats(self, can=True, ref=True):
        if can and ref:
            return self.group_stats
        elif can:
            return self.group_stats[self.group_stats.index.str.contains(self.candidate_name)]
        elif ref:
            return self.group_stats[self.group_stats.index.str.contains(self.reference_name)]
        else:
            return None



    def _group_stats(self):
        '''
        Calculate the statistical values needed for the horizontal cci_validation.

        Returns
        -------
        group_stats : pandas.DataFrame
            Frame that contains the group stats
        '''
        group_stats = pd.DataFrame()
        basic_measures = self.basic_measures[:,0]

        for group_no, subset_data in enumerate([self.set0, self.set1, self.setfull]):
            if group_no in [0,1]:
                group = 'group%i' % group_no
            else:
                group = 'FRAME'
            for name in [self.candidate_name, self.reference_name]:
                # Basic measures
                if 'min' in basic_measures:
                    min = np.nan if subset_data[name].empty else np.min(subset_data[name].values)
                    group_stats.at['min_%s' % name, '%s' % group] = min
                if 'max' in basic_measures:
                    max = np.nan if subset_data[name].empty else np.max(subset_data[name].values)
                    group_stats.at['max_%s' % name, '%s' % group] = max
                if 'var' in basic_measures:
                    var = np.nan if subset_data[name].empty else np.var(subset_data[name].values)
                    group_stats.at['var_%s' % name, '%s' % group] = var
                if 'mean' in basic_measures:
                    mean = np.nan if subset_data[name].empty else np.mean(subset_data[name].values)
                    group_stats.at['mean_%s' % name, '%s' % group] = mean
                if 'median' in basic_measures:
                    median = np.nan if subset_data[name].empty else np.median(subset_data[name].values)
                    group_stats.at['median_%s' % name, '%s' % group] = median
                if 'iqr' in basic_measures:
                    iqr = np.nan if subset_data[name].empty else stats.iqr(subset_data[name].values, nan_policy='omit')
                    group_stats.at['iqr_%s' % name, '%s' % group] = iqr

        return group_stats


    def _vertical_metrics(self):
        '''
        Calculate metrics for the 2 groups and the full set. Using pytesmo and
        stats for the groups

        Returns
        -------
        vertical_metrics : pd.DataFrame
            The combined basic and pytesmo metrics
        '''

        vmp = self._vertical_metrics_pytesmo()
        vmg = self._vertical_metrics_from_groupstats()

        return pd.concat([vmp, vmg], axis=0)


    def _vertical_metrics_from_groupstats(self):
        '''
        Compare group stats (before after break) to detect impact of the break
        on metrics.

        Returns
        -------
        vertical_basic_metrics: pd.DataFrame
            Metric comparison at break time for the 2 groups
        '''
        vertical_metrics = pd.DataFrame()

        # from basic stats
        for var, meth in self.basic_measures:
            for group in ['group0', 'group1', 'FRAME']:
                candidate_metric = self.group_stats.loc['%s_%s' % (var, self.candidate_name), group]
                reference_metric = self.group_stats.loc['%s_%s' % (var, self.reference_name), group]

                vmetric = compare(candidate_metric, reference_metric, meth)

                vertical_metrics.at['%s_%s' % (var, meth), group] = vmetric

        return vertical_metrics



    def _vertical_metrics_pytesmo(self):
        '''
        Calculate vertical metrics between candidate and reference using pytesmo.

        Returns
        -------
        vertical_metrics: pandas.DataFrame

        '''
        vertical_metrics = pd.DataFrame()

        pytesmo_measures = self.pytesmo_measures

        for group_no, subset_data in enumerate([self.set0, self.set1, self.setfull]):
            if group_no in [0,1]:
                group = 'group%i' % group_no
            else:
                group = 'FRAME'
            if 'bias' in pytesmo_measures:
                if any([subset_data[col].empty for col in [self.candidate_name, self.reference_name]]):
                    bias = np.nan
                else:
                    bias =metrics.bias(subset_data[self.reference_name].values,
                                       subset_data[self.candidate_name].values)
                vertical_metrics.at['bias', '%s' % group] = bias

            if 'mad' in pytesmo_measures:
                if any([subset_data[col].empty for col in [self.candidate_name, self.reference_name]]):
                    mad = np.nan
                else:
                    mad =metrics.mad(subset_data[self.reference_name].values,
                                     subset_data[self.candidate_name].values)
                vertical_metrics.at['mad', '%s' % group] = mad


            if 'rmsd' in pytesmo_measures:
                if any([subset_data[col].empty for col in [self.candidate_name, self.reference_name]]):
                    rmsd = np.nan
                else:
                    rmsd =metrics.rmsd(subset_data[self.reference_name].values,
                                       subset_data[self.candidate_name].values)
                vertical_metrics.at['rmsd', '%s' % group] = rmsd

            if 'nrmsd' in pytesmo_measures:
                if any([subset_data[col].empty for col in [self.candidate_name, self.reference_name]]):
                    nrmsd = np.nan
                else:
                    nrmsd =metrics.nrmsd(subset_data[self.reference_name].values,
                                         subset_data[self.candidate_name].values)
                vertical_metrics.at['nrmsd', '%s' % group] = nrmsd

        return vertical_metrics

    def _horizontal_metrics(self, how='Diff'):
        '''
        Compare metrics for the 2 groups via the passed comparison metric.

        Parameters
        ----------
        how : str
            Comparison metric for the metrics of the 2 groups.

        Returns
        -------
        horizontal_metrics : pandas.DataFrame
            The comparison metrics of the 2 groups
        '''

        horizontal_metrics = pd.Series()

        for var, meth in self.basic_measures:
            s0 = self.vertical_metrics.loc['%s_%s' % (var, meth), 'group0']
            s1 = self.vertical_metrics.loc['%s_%s' % (var, meth), 'group1']
            hmetric = compare(s0, s1, how)
            horizontal_metrics.at['%s_%s_%s' % (how, var, meth)] = hmetric

        for var in self.pytesmo_measures:
            s0 = self.vertical_metrics.loc[var, 'group0']
            s1 = self.vertical_metrics.loc[var, 'group1']
            hmetric = compare(s0, s1, how)
            horizontal_metrics.at['%s_%s' % (how, var)] = hmetric

        return horizontal_metrics


def usecase():
    pass


if __name__ == '__main__':
    can = pd.Series(index=pd.DatetimeIndex(start='2000-01-01', end='2000-12-31', freq='D'),
                         data = np.random.rand(366), name='thecan')

    ref = pd.Series(index=pd.DatetimeIndex(start='2000-01-01', end='2000-12-31', freq='D'),
                         data = np.random.rand(366), name='theref')

    ds = HorizontalVal(can, ref, datetime(2000,1,7))

    errors =  ds.run('Diff')
