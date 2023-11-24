# -*- coding: utf-8 -*-

from pybreaks.base import TsRelBreakBase
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from collections import OrderedDict
from scipy import stats
import collections
import warnings

warnings.simplefilter('always', UserWarning)

"""
Module implements the quantile matching adjustment (Wang 2010, Vincent 2012) 
for breaks in soil moisture observation time series.
"""

# TODO:
#   (-) add a function to both classes to plot the adjustments only, as done for the
#       other 2 correction classes.
#---------
# NOTES:
#   -

class N_Quantile_Exception(ValueError):
    def __init__(self, *args, **kwargs):
        super(N_Quantile_Exception, self).__init__(*args, **kwargs)


def _init_rank(indata, name='can', drop_dupes=False):
    """
    Sort/rank the passed dataframe  by the 'name'-column, drop duplicate
    'name'-values if chosen.

    Parameters
    -------
    indata : pd.DataFrame
        Series of SM observations over time
    name : str, optional (default: 'can')
        Name of the column that the CDF is based on
    drop_dupes : bool, optional (default: False)
        Drop candidate duplicates (after freq calc), so that there are no duplicate
        CFs in the return data_cf.

    Returns
    -------
    data : pd.DataFrame
        The input ds with additional columns for count, freq, ranks
        Contains duplicates.
    data_cf : pd.DataFrame
        data sorted by CF (index), if chosen without duplicates
    """
    data = indata.copy(True)

    data = data.dropna()

    # data has candidate ranks and dates
    data['%s_ranks' % name] = data[name].rank()

    data = data.sort_values(by=['%s_ranks' % name])

    # count the occurrence of each rank (each unique candidate)
    data_count = data['%s_ranks' % name].value_counts(normalize=False).sort_index()
    # count the occurrence of each rank (each unique candidate) and normalize to all observations
    data_norm_count = data['%s_ranks' % name].value_counts(normalize=True).sort_index()
    # sum up the number of normalized occurrences for each rank (each observation) and sort them.
    data_norm_cumfreq = data_norm_count.cumsum().sort_index()

    if drop_dupes:
        data_no_cand_dupes = data.loc[data['%s_ranks' % name].drop_duplicates().index, :]
        data_no_cand_dupes = data_no_cand_dupes.set_index('%s_ranks' % name)

        data_no_cand_dupes['F_%s' % name] = data_count
        data_no_cand_dupes['norm_F_%s' % name] = data_norm_count
        data_no_cand_dupes['norm_CF_%s' % name] = data_norm_cumfreq
        data_no_cand_dupes['rank_%s' % name] = data_no_cand_dupes.index

        data_cf = data_no_cand_dupes.set_index('norm_CF_%s' % name).sort_index()
    else:
        data = data.set_index('%s_ranks' % name)
        data['F_%s' % name] = data_count
        data['norm_F_%s' % name] = data_norm_count
        data['norm_CF_%s' % name] = data_norm_cumfreq
        data['rank_%s' % name] = data.index
        data_cf = data.set_index('norm_CF_%s' % name).sort_index()

    return data, data_cf


class QuantileCatMatch(TsRelBreakBase):
    """
    Relative matching of quantile categories between 2 parts of a candidate series
    relative to changes in a reference series.
    """

    def __init__(self, candidate, reference, breaktime,
                 bias_corr_method='cdf_match', adjust_group=0, categories=4,
                 first_last='formula', fit='mean'):
        '''
        Class that performs adjustment of candidate data before/after break
        time to match to the reference data

        Parameters
        ----------
        candidate : pd.Series
            Candidate data that should be adjusted
        reference : pd.Series
            Reference data that is used to fit candidate data to
        breaktime : datetime.datetime
            Time of the detected break
        bias_corr_method : str, optional
            Bias correction that is performed to fit reference to candidate
            before adjusting. Default is cdf_match
        adjust_group : int, optional
            Identifies the group (0=before break time, 1=after break time) that
            is being adjusted. By default the part before the break time is
            adjusted.
        categories : int, optional
            Number of percentiles that are fitted (equal distribution)
            (0 and 100 are always considered) must be >=1.
            Default is 4 quartile categories.
        first_last : str, optional (default: 'formula')
            'formula', 'equal' or 'None'
            Select 'formula' to calc the boundary values after the formula or
            'equal' to use the same value as for the quantile category they are in.
            If 'None' is passed the edge values are not calculated.
        fit : str, optional (default: 'mean')
            Select mean or median to fit the QC means or medians.
        '''

        candidate = candidate.copy(True)
        reference = reference.copy(True)

        candidate.name = 'can'
        reference.name = 'ref'

        TsRelBreakBase.__init__(self, candidate, reference, breaktime,
                                bias_corr_method, dropna=False)


        if first_last == 'None':
            first_last = None
        self.first_last = first_last
        self.fit = fit

        self._n_quantiles = categories
        self._init_percentiles()

        self.adjust_group, self.other_group = self._check_group_no(adjust_group)

        self.df_adjust = self.df_original.copy(True)

        self._init_group_data()

        mq_cand_0, mq_cand_1 = self.split_mq(self.data_cf0_can, self.data_cf1_can)

        self.D0, self.D1 = self.calc_qcm_models(mq_cand_0, mq_cand_1)

        m0, m1 = self.D0['adj'].dropna().to_dict(), self.D1['adj'].dropna().to_dict()
        self.model0_params = OrderedDict(sorted(m0.items()))
        self.model1_params = OrderedDict(sorted(m1.items()))

        self.adjusted_col_name = None
        self.adjust_obj = None

    def _init_percentiles(self):
        self._percentile_edges = np.linspace(0, 100, self.n_quantiles + 1)
        percentile_middles = []
        for p in np.linspace(0, 100, self.n_quantiles * 2 + 1)[1:-1]:
            if p not in self.percentile_edges:
                percentile_middles.append(p)
        self._percentile_middles = np.array(percentile_middles)

    def _init_group_data(self):
        data0 = self.get_group_data(0, self.df_original,
            [self.candidate_col_name, self.reference_col_name])
        data1 = self.get_group_data(1, self.df_original,
            [self.candidate_col_name, self.reference_col_name])
        data0['D'] = data0['can'] - data0['ref']
        data1['D'] = data1['can'] - data1['ref']
        self.data0 = data0
        self.data1 = data1
        # based on CAN:
        _, self.data_cf0_can = _init_rank(self.data0, 'can', drop_dupes=False)
        _, self.data_cf1_can = _init_rank(self.data1, 'can', drop_dupes=False)
        # based on REF, not used, do this when needed (plotting):
        self.data_cf0_ref = None
        self.data_cf1_ref = None

    @property
    def n_quantiles(self):
        return self._n_quantiles

    @property
    def percentile_edges(self):
        return self._percentile_edges

    @property
    def percentile_middles(self):
        return self._percentile_middles


    def plot_cdf_compare(self, ax=None, raw_plot=False):
        """
        Parameters
        ----------
        axs : matplotlib.Axes.axs, optional (default: None)
            Axes put the plot into
        raw_plot : bool, optional (default: False)
            Create a plot without title and interface, labels and legend.
        """
        if not ax:
            fig, ax = plt.subplots()

        self.D0['can'].plot(ax=ax, label='CAN Before Break')
        self.D1['can'].plot(ax=ax, label='CAN After Break')

        if self.adjust_obj is not None:
            df = self.adjust_obj.df_cf # type: pd.DataFrame
            df['input'].plot(ax=ax, label='Values to Adjust')
            if 'adj' in df.columns:
                df_adjusted = df['input'] + df['adj']
                df_adjusted.plot(ax=ax, label='After Adjustment')

        if not raw_plot:
            plt.suptitle('Candidate CDF compare')
            plt.legend()


    def plot_emp_dist_can_ref(self, groups=np.array([0, 1]), axs=None, raw_plot=False):
        """
        Plot the empirical distributions for CAN and REF, BEF and AFT break.

        Parameters
        -------
        groups : iterable, optional (default: [0,1])
            List of groups. eg. [0,1] = [HSP1, HSP2], [1]=HSP1 etc.
        axs : plt.Axes, optional (default: None)
            Use these axes to plot the groups
            Must have the same size as the selected list of groups
        raw_plot : bool, optional (default:False)
            Plot without labels, title and legend
        """

        if self.data_cf0_ref is None:
            _, self.data_cf0_ref = _init_rank(self.data0, 'ref', drop_dupes=False)
        if self.data_cf1_ref is None:
            _, self.data_cf1_ref = _init_rank(self.data1, 'ref', drop_dupes=False)

        plot_coll_fig = plt.figure(figsize=(9.3, 4.8),
                                   facecolor='w', edgecolor='k')

        for group in groups:
            if axs is not None:
                if isinstance(axs, collections.Iterable):
                    ax = axs[group]
                else:
                    raise ValueError('axes must be iterable and of same size as the passed groups')
            else:
                ax = plot_coll_fig.add_subplot(1, 2, group + 1)

            hsp = 'HSPa' if group == self.adjust_group else 'HSPr'

            if group == 0:
                can = self.data_cf0_can.copy(True)  # type: pd.DataFrame
                ref = self.data_cf0_ref.copy(True)  # type: pd.DataFrame
            else:
                can = self.data_cf1_can.copy(True)  # type: pd.DataFrame
                ref = self.data_cf1_ref.copy(True)  # type: pd.DataFrame

            # before break
            ax.plot(can['can'], can.index.values, color='red',
                    label='%s %s' % ('CAN', hsp))

            # after break
            ax.plot(ref['ref'], ref.index.values, color='black',
                    label='%s %s (not used)' % ('REF', hsp))

            # h = max(max(can['can']), max(ref['ref']))
            # b = min(min(can['can']), min(ref['ref']))
            for p in self.percentile_middles / 100:
                ax.axhline(p, color='grey', linestyle='--', linewidth=0)

            for p in self.percentile_edges / 100:
                ax.axhline(p, color='grey', linestyle=':', linewidth=2)

            if not raw_plot:
                ax.set_title('%s : Empirical Distributions' % hsp)
                ax.set_ylabel('Normalized CF')
                ax.set_xlabel('SM')

                ax.legend()

        plt.tight_layout()

        return plot_coll_fig

    def plot_pdf_compare(self, names=['can', 'ref', 'D'], cumulative=False):
        """
        Plots a collection of the PDFs for before and after the break time for
        the selected variables.

        Parameters
        -------
        names : list, optional (default: ['can', 'ref', 'D'])
            list of parameters (can, ref or D) which are plotted
        cumulative : bool, optional (default: False)
            True to create sums of values.

        Returns
        -------
        figure_collection :
            The pdf compare collection
        """

        rows, cols = 1, 2

        plot_coll_fig = plt.figure(figsize=(12.5, 5),
                                   facecolor='w', edgecolor='k')

        # plot pdf of can and ref
        for group in [0, 1]:

            ax = plot_coll_fig.add_subplot(rows, cols, group + 1)

            if group == 0:
                data = self.data0
            else:
                data = self.data1

            for i, name in enumerate(names):
                if name == 'can':
                    c = 'r'
                    dist_fit = None
                elif name == 'ref':
                    c = 'k'
                    dist_fit = None
                elif name == 'D':
                    c = 'b'
                    dist_fit = stats.norm
                else:
                    c = 'k'
                    dist_fit = None

                # plot pdfs
                style = '{}--'.format(c) if group == self.adjust_group else '{}-'.format(c)
                ax = self.plot_pdf(data=data, ax=ax, style=style,
                                   dist_fit=dist_fit, name=name, cumulative=cumulative)

            period = 'HSPa' if group == self.adjust_group else 'HSPr'
            ax.set_title('PDFs : %s' % period)

            ax.legend()

        return plot_coll_fig

    def plot_model(self, period, ax, vlines=True,
                   scatter_style=('.', 3, 'orange'),
                   bar_style=('/', 0.8, 'blue'),
                   vline_style=('grey', ':', '--'), label=None,
                   show_qc=False):
        """
        Plot the mode for the chosen part.

        Parameters
        ----------
        period : str
            'before' or 'after' or 'diff' to plot the according period
        ax : plt.ax
            Use this axes object
        vlines : bool, optional (default: True)
            Plot the vertical .. lines between the categories
        scatter_style : tuple, optional (default:('.', 3, 'orange'))
            0 : marker type, 1 : marker size, 2 : marker color
        bar_style : tuple, optional (default:('/', 0.8, 'blue'))
            0 : bar hatch style, 1 : bar alpha, 2 : bar color
        vline_style : tuple, optional (default:('/', 0.8, 'blue'))
            0 : vlines color, 1 : main vlines linestyle, 2 : middle vlines linestyle
        label : str, optional (default: None)
            A label for the bars
        show_qc : bool, optional (default: False)
            Also plot an axis with the category number.
        """
        if vlines:
            if vline_style[2]:
                for m in self.percentile_middles / 100:
                    ax.axvline(m, color=vline_style[0], linestyle=vline_style[2],
                               linewidth=1, zorder=0)
            if vline_style[1]:
                for e in self.percentile_edges / 100:
                    ax.axvline(e, color=vline_style[0], linestyle=vline_style[1],
                               linewidth=2, zorder=0)

        cat_data0, cat_data1 = self.D0['adj'].dropna(), self.D1['adj'].dropna()
        if period == 'before':
            cat_data = cat_data0
        elif period == 'after':
            cat_data = cat_data1
        elif period == 'diff':
            if self.adjust_group == 1:
                cat_data = cat_data0 - cat_data1
            else:
                cat_data = cat_data1 - cat_data0
        else:
            raise ValueError("period must be one of ´before´, ´after´ or ´diff´")

        ax.bar(cat_data.iloc[1:-1].index, cat_data.iloc[1:-1].values,
               color=bar_style[2], width=1. / float(self.n_quantiles), alpha=bar_style[1],
               label=label if label else None, hatch=bar_style[0], edgecolor='black')
        if scatter_style[0]:
            ax.scatter(cat_data.index, cat_data.values, color=scatter_style[2],
                       label=None, zorder=2, marker=scatter_style[0], s=scatter_style[1])

        if show_qc:
            new_tick_locations = self.percentile_middles / 100.
            ax2 = ax.twiny()
            ax2.grid(False)
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(new_tick_locations)
            ax2.set_xticklabels([r'$QC_%i$' % i for i in range(1, cat_data.index.size - 1)])
            # ax2.set_xlabel('QC')

    def plot_models(self, image_path=None, axs=None, names=None, supress_title=False):
        """
        Create 2 plots for the categories for data before and after the break

        Parameters
        -------
        image_path : str, optional (default: None)
            Directory where the image is stored to
        axs : list , optional (default: None)
            List of axes object
        names : list, optional (default: None)
            Names of vars to plot, if axs are passed, lengths must correspond
        supress_title : bool
            Do not create plot title
        """

        if names is None:
            if self.adjust_obj is None:
                names = ['before', 'after']
            else:
                names = ['before', 'after', 'diff']

        if axs is not None:
            if not (isinstance(axs, list) and (len(axs) == len(names))):
                raise Exception('Wrong number of axes passed')
            else:
                if image_path:
                    raise Exception('Cannot store the plot if axs are passed')
                else:
                    plot_coll_fig = None
        else:
            plot_coll_fig = plt.figure(figsize=(6 * len(names), 6),
                                       facecolor='w', edgecolor='k')

        for i, p in enumerate(names):
            if plot_coll_fig is not None:
                ax = plot_coll_fig.add_subplot(1, len(names), i + 1)
            else:
                ax = axs[i]

            if p == 'before':  # categories before bt
                self.plot_model('before', ax, True)
                period = 'HSPa' if self.adjust_group == 0 else 'HSPr'
                if not supress_title:
                    ax.set_title('%s: Quantile Categories' % period)
                ax.set_xlabel('Cumulative Frequency')
                ax.set_ylabel(r'Category Mean Difference (CAN-REF)')
                ax.legend()

            elif p == 'after':  # categories after bt
                self.plot_model('after', ax, True)
                period = 'HSPa' if self.adjust_group == 1 else 'HSPr'
                if not supress_title:
                    ax.set_title('%s: Quantile Categories' % period)
                ax.set_xlabel('Cumulative Frequency')
                ax.set_ylabel(r'Category Mean Difference (CAN-REF)')
                # ax.legend()

            elif p == 'diff' and len(names) == 3:
                # difference of category differences means
                self.plot_model('diff', ax, True)
                if self.adjust_obj is not None:
                    ax = self.plot_adjustments(ax)
                if not supress_title:
                    ax.set_title('Category Differences')
                ax.set_xlabel('Cumulative Frequency')
                ax.set_ylabel(r'$\Delta$(Category Mean Difference)')
                ax.legend()

        plt.tight_layout()
        return plot_coll_fig

    def plot_adjustments(self, ax=None):
        return self.adjust_obj.plot_adjustments(ax=ax)


    @staticmethod
    def plot_pdf(data, style, ax, name='can', dist_fit=stats.norm, cumulative=False):
        """
        Plot the pdfs of the passed data

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame that contains the column with NAME
        style : str
            Color and list style
        ax : plt.axes
            Axes to add the plot
        name : str
            Name of the column in the dataframe
        dist_fit : stats.dist
            Distribution to fit to the data
        cumulative : bool
            True to create sums of values.

        Returns
        -------
        ax
        """

        if not ax:
            fig, ax = plt.subplots()

        color = style[0]
        line = style[1:]

        pdf_data = data[name].dropna()
        ax.hist(pdf_data, color=color, density=True, alpha=0.3, cumulative=cumulative,
                bins=25, label='%s Empirical PDF' % name if name else '')

        if dist_fit is not None:
            ml_fit = dist_fit.fit(pdf_data)
            x = np.linspace(min(pdf_data), max(pdf_data), 100)
            y = dist_fit.pdf(x, *ml_fit)
            label = '%s Fit %s %s' % (name if name else '', dist_fit.name,
                                      str([np.round(p, 3) for p in ml_fit]))
            ax.plot(x, y, color=color, linestyle=line, label=label)

        ax.set_ylabel('%s Frequency' % ('Cumulative' if cumulative else ''))
        ax.set_xlabel('SM')

        return ax

    def calc_perc(self, group, calc_for, frame=None):
        """
        Transferts the percentils of the object to the candidate and reference
        data in the passed frame

        Parameters
        ----------
        group : int
            Group for which the percentiles shall be calculated
        calc_for : str
            Select whether the percentiles shall be calculated for the category
            middles ("mid"), or the category edges ("edge"), or the whole
            category ("cat") via the mean of all percentiles within the cat.
        frame : pd.DataFrame
            Frame from which the data is used for the percentile calculation

        Returns
        -------
        can_percentiles : np.array
            Candidate values at the percentiles
        ref_percentiles : np.array
            Reference values at the percentiles
        adj_percentiles : np.array
            Reference values at the percentiles
        """

        if frame is None:
            frame = self.df_original

        can = self.get_group_data(group, frame, self.candidate_col_name)
        ref = self.get_group_data(group, frame, self.reference_col_name)

        if self.adjusted_col_name is not None:
            adj = self.get_group_data(group, frame, self.adjusted_col_name)
        else:
            adj = None

        if calc_for == 'mid':
            percentiles = self.percentile_middles
        elif calc_for == 'edge':
            percentiles = self.percentile_edges
        else:
            raise NameError('select mid or edge')

        ref_percentiles = np.array(np.percentile(ref, percentiles))

        can_percentiles = np.array(np.percentile(can, percentiles))

        if adj is not None:
            adj_percentiles = np.array(np.percentile(adj, percentiles))
        else:
            adj_percentiles = None

        return can_percentiles, ref_percentiles, adj_percentiles

    def _ts_props(self):
        """Specific for each child class of adjustment base, allows plotting"""
        props = {'isbreak': None,
                 'breaktype': None,
                 'candidate_name': self.candidate_col_name,
                 'reference_name': self.reference_col_name,
                 'adjusted_name': self.adjusted_col_name,
                 'adjust_failed': False}

        return props

    def get_model_params(self, model_no=None):
        """
        Get the model parameters of the cdfs for the according group
        of the current iteration.

        Parameters
        ----------
        model_no : int (0 or 1) or None
            Number of the model (0=before break, 1=after break)

        Returns
        -------
        model_params: pandas.Series
            Category means for the selected part
        """
        self.model0_params['n_quantiles'] = self.n_quantiles
        self.model1_params['n_quantiles'] = self.n_quantiles

        if model_no is None:
            return {'model0': self.model0_params, 'model1': self.model1_params}
        elif model_no == 0:
            return self.model0_params
        elif model_no == 1:
            return self.model1_params
        else:
            raise ValueError(model_no, 'Invalid value, select 0, 1 or None')

    def split_by_freq(self, df):
        """
        Split a df by the cum freq in the index
        So that each part contains the same range of frequencies.
        The number of values in each group will vary, as there are not equally
        many values in each CF range.

        Parameters
        ----------
        df : pd.DataFrame
            Data over the normalized CF range (0-1)

        Returns
        -------
        parts : list
            List of parts for the data frame for each quantile category
        """

        inc = 1. / self.n_quantiles

        parts = []
        lower = 0
        for i in range(self.n_quantiles):
            upper = lower + inc
            cond1 = (df.index > lower)
            cond2 = (df.index <= upper)

            dat = df.loc[(cond1 & cond2)]
            # add the category as a column
            dat = dat.assign(cat=i)
            parts.append(dat)
            lower = upper

        return parts

    def split_mq(self, data0_cf, data1_cf):
        """
        Split data frames in quantile range parts

        Parameters
        ----------
        data0_cf : pd.DataFrame
            Data before the break
        data1_cf : pd.DataFrame
            Data after the break

        Returns
        -------
        mq_cand_0 : list
            List of DataFrames for each quantile category for data before the break
        my_cand_1 : list
            List of DataFrames for each quantile category for data after the break
        """

        mq_cand_0 = self.split_by_freq(data0_cf)
        mq_cand_1 = self.split_by_freq(data1_cf)

        return mq_cand_0, mq_cand_1

    def _cat_mean(self, mq_cand):
        """
        Calculate category means for the median CF of each category and set
        values at 0 and 1 for interpolation,

        Parameters
        ----------
        mq_cand : list
            List of category DataFrames

        Returns
        -------
        D_mean : pd.DataFrame
            The model DataFrame
        """

        cats = []

        for n_quant, cat in enumerate(mq_cand, start=1):
            if cat.empty:
                raise N_Quantile_Exception(self.n_quantiles,
                                           'Number of passed quantile categories led to an empty category.')
            # the new middle value
            if self.fit == 'mean':
                mean = cat['D'].mean()
            elif self.fit == 'median':
                mean = cat['D'].median()
            else:
                raise Exception('select mean or median for fit')

            cat.loc[:, 'D_mean'] = mean
            df_middle = pd.DataFrame(index=[(float(n_quant) - 0.5) / self.n_quantiles],
                                     data={'adj': mean})

            if df_middle.index[0] in cat.index:
                cat.at[df_middle.index[0], 'adj'] = df_middle['adj'].values[0]
            else:
                cat = pd.concat((cat, df_middle), sort=True).sort_index()

            cats.append(cat)

        # add first and last value to D frame
        d_mean = pd.concat(cats, sort=True, axis=0)
        if self.first_last:
            if self.first_last == 'formula':  # todo: iloc[0] or iloc[1]?
                first = d_mean['adj'].dropna().iloc[0] - (1. / self.n_quantiles)
                last = d_mean['adj'].dropna().iloc[-1] + (1. / self.n_quantiles)
            elif self.first_last == 'equal':  # todo: or formula?
                first = d_mean['adj'].dropna().iloc[0]
                last = d_mean['adj'].dropna().iloc[-1]
            else:
                raise ValueError(self.first_last, 'Unknown input for first_last')

            # D_mean0_last = pd.DataFrame(index=[1], data={col:np.nan for col in D_mean0.columns})
            df_firstlast = pd.DataFrame(index=[0., 1.], data={'adj': [first, last]})

            if df_firstlast.index[0] in d_mean.index.values:
                d_mean.at[df_firstlast.index[0], 'adj'] = df_firstlast['adj'].values[0]
                df_firstlast = df_firstlast.drop(df_firstlast.index[0])

            if df_firstlast.index[-1] in d_mean.index.values:
                d_mean.at[df_firstlast.index[-1], 'adj'] = df_firstlast['adj'].values[-1]
                df_firstlast = df_firstlast.drop(df_firstlast.index[-1])

            if not df_firstlast.empty:
                d_mean = pd.concat((d_mean, df_firstlast), sort=True).sort_index()
        else:
            d_mean = d_mean.sort_index()

        return d_mean

    def calc_qcm_models(self, mq_cand0, mq_cand1):
        """
        Calculates frequencies and difference values for the selected quantiles

        Parameters
        ----------
        mq_cand0 : list
            List of sub-frames for the quantile categories for data before the break time
        mq_cand1 : list
            List of sub-frames for the quantile categories for data before the break time

        Returns
        -------
        d_mean0 : pandas.DataFrame
            Value of data before the break time
        d_mean1 : pandas.DataFrame
            Value of data after the break time
        """

        # differences for data before the break

        d_mean0 = self._cat_mean(mq_cand0)
        d_mean1 = self._cat_mean(mq_cand1)

        return d_mean0, d_mean1

    def adjust(self, values_to_adjust, interpolation_method='cubic'):
        """
        Use the Adjustment class to adjust the candidate data of the object.

        Parameters
        -------
        values_to_adjust : pd.Series, optional
            Candidate values to which the adjustments are applied
        interpolation_method : str, optional (default : 'cubic')
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
            refer to a spline interpolation of zeroth, first, second or third
            order; 'previous' and 'next' simply return the previous or next value
            of the point) or as an integer specifying the order of the spline
            interpolator to use.

        Returns
        -------
        adjusted_values : pd.Series
            The adjusted candidate values

        """
        if values_to_adjust is None:
            # use these values to get adjustments for
            values_to_adjust = self.get_group_data(
                self.adjust_group, self.df_adjust, self.candidate_col_name)

        self.adjusted_col_name = self.candidate_col_name + '_adjusted'

        # use these values to get adjustments for

        before_after_break = 'before' if self.adjust_group == 0 else 'after'

        adjust_obj = QuantileCatMatchAdjust(ds=values_to_adjust,
                                            D0=self.D0, D1=self.D1,
                                            before_after_break=before_after_break)

        self.adjust_obj = adjust_obj

        adjusted_values = adjust_obj.adjust(interpolation_method)  # adjusted values_to_adjust

        self.df_original[self.adjusted_col_name] = self.df_original[self.candidate_col_name]

        # add the adjusted values
        common_index = self.df_original.index.intersection(adjusted_values.index)
        self.df_original.loc[common_index, self.adjusted_col_name] = adjusted_values

        return adjusted_values


class QuantileCatMatchAdjust(object):
    """
    Adjust a time series based on the results from the quantile category matching
    method.
    """

    def __init__(self, ds, D0, D1, before_after_break):
        """

        Parameters
        ----------
        ds: pd.Series
            Values that will be adjusted based on the passed models diffs
        D0: pd.DataFrame
            Difference frame for the data of group0
        D1: pd.DataFrame
            Difference frame for the data of group1
        before_after_break : str
            Identifier if the data that should be adjusted was before or after
            the detected beak (either 'before' or 'after')
        """

        # Values that are used to calculate adjustments from

        self.ds = ds.copy(True)  # type: pd.Series
        df = pd.DataFrame(index=self.ds.index, data={'input': self.ds.values})

        self.D0 = D0
        self.D1 = D1

        self.before_after_break = before_after_break

        self.df, self.df_cf = _init_rank(df, 'input', drop_dupes=True)

        self.cat_diffs = self._corr_params()

    def _corr_params(self):
        """
        Calculate differences in the category means for the 2 groups

        Returns
        -------
        cat_diffs : pd.DataFrame
            Category differences between D0 and D1
        """

        cat_diffs = pd.DataFrame()

        if self.before_after_break == 'before':
            cat_diffs['D'] = self.D1['adj'] - self.D0['adj']  # todo: right order?
        elif self.before_after_break == 'after':
            cat_diffs['D'] = self.D0['adj'] - self.D1['adj']
        else:
            raise Exception("select 'before' or 'after' if tbe passed values "
                            "are before/after the break")

        # drop nans and duplicate indices
        cat_diffs = cat_diffs.dropna()
        return cat_diffs[~cat_diffs.index.duplicated(keep='first')]

    def apply_adjustments(self, df_in, input_col_name, adjustments_col_name):
        # Todo: Do something with the adjustments before applying them?
        # This must support some kind of upsampling, in cases where the temporal
        # resolution of the adjustments does not match the input data.

        df = df_in.copy(True)
        df['adjusted'] = df[input_col_name] + df[adjustments_col_name]
        return df

    def adjust(self, interpolation_method='cubic'):
        """
        Adjust the candidate values based on SM adjustments per CF.
        Connection between adjustments (CF) and candidate values (dates) is
        done via the candidate ranks.

        Parameters
        -------
        interpolation_method : str, optional (default: 'cubic')
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
            refer to a spline interpolation of zeroth, first, second or third
            order; 'previous' and 'next' simply return the previous or next value
            of the point) or as an integer specifying the order of the spline
            interpolator to use.

        Returns
        -------
        ds_adjusted : pd.Series
            The adjusted input series
        """

        adjustments = self._calc_adjustments(interpolation_method)

        df = self.df.copy(True)
        df.loc[:, 'date'] = df.index
        df = df.set_index('input_ranks')

        df['adjustments'] = np.nan

        for ind in adjustments.index.values:
            # TODO: MAKE THIS WITHOUT LOOP
            df.loc[ind, 'adjustments'] = adjustments.loc[ind, ('adj')]

        df.loc[:, 'cand_ranks'] = df.index
        df = df.set_index('date').sort_index()

        df = self.apply_adjustments(df, input_col_name='input',
                                    adjustments_col_name='adjustments')

        self.df = df
        self.adjustments = self.df['adjustments']

        return df['adjusted']

    def _calc_adjustments(self, interpolation_method='cubic'):
        """
        Calculate SM adjustements for for CFs of can_adj by interpolation the
        category means differences.

        Parameters
        -------
        interpolation_method : str, optional (default : 'cubic')
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
            refer to a spline interpolation of zeroth, first, second or third
            order; 'previous' and 'next' simply return the previous or next value
            of the point) or as an integer specifying the order of the spline
            interpolator to use.
        Returns
        -------
        adjustments : pd.DataFrame
            DataFrame that contains the adjustment value for each CF and the
            respective rank of the candidate, that it will adjust.
        """
        if interpolation_method in [False, None]:
            raise ValueError(interpolation_method, 'Select an interpolation method')

        cat_diffs = self.cat_diffs

        x = cat_diffs.index.values
        y = cat_diffs['D'].values

        if x.size == 3:
            # one category means 3 points
            f = interp1d(x, y, kind='linear', fill_value="extrapolate")  # 3 equal points
        else:
            f = interp1d(x, y, kind=interpolation_method, fill_value="extrapolate")

        x_new = self.df_cf.index.values
        f_new = f(x_new)

        self.df_cf['adj'] = f_new

        return self.df_cf.loc[:, ['rank_input', 'adj']].set_index('rank_input')

    def plot_adjustments(self, ax=None):
        """
        Crate a plot of the interpolated corrections

        Parameters
        -------
        ax : plt.Axes, optional (default: None)
            Use this axis object instead to create the plot in
        cf: bool, optional (default: True)
            Plot adjustments over CF, otherwise plot the adjustments over time.
        """
        if ax is None:
            fig, ax = plt.subplots(1,1)

        interpol_diff = self.df_cf['adj']
        # interpolation, if already done
        ax.plot(interpol_diff.index.values, interpol_diff.values, linestyle='--',
                color='red', label='cubic spline')
        return ax

if __name__ == '__main__':
    pass
