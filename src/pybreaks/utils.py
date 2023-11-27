# -*- coding: utf-8 -*-

from datetime import datetime
from collections.abc import Iterable

import pandas as pd
import numpy as np
import calendar
from datetime import timedelta

'''
These functions seem useful for use outside the break detection.
Therefore they are made independent from any classes
'''

def merge_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def conditional_temp_resample(ds, how='M', threshold=0.1):
    '''
    Resample a dataframe to the given temporal resolution ('M', 'D', etc.).
    If the number of valid values (not nans) in a resample period (eg. 'M')
    is smaller than the defined threshold, the resample will be NaN Parameters.
    ----------
    can_adj : pandas.Series
        DataFrame to resample
    how : str
        Time frame for temporal resampling, M = monthly, SMS= semi-month start, SM = semi-month end
        MM=month middle
    threshold : float
        % of valid observations (not nan) in period defined in 'how'
    Returns
    -------
    df_resampled : pd.Series
        The resampled Series

    df_count : pd.DataFrame
        Number of observations per period and threshold that had to be reached
    '''

    if ds.isnull().any():
        raise ValueError('Input Series must not contain nans')
    name = ds.name
    df = ds.copy(True).to_frame(name='input')

    if all(df['input'].isnull()): # if there are only Nons in the can_adj
        df['input'].fillna(np.nan, inplace=True)
        threshold = None

    if how == 'MM':
        month_middle = True
        how = 'MS'
    else:
        month_middle = False

    if not threshold:
        df_resampled = df.resample(how).mean()
        df_count = None
    else:
        if 'M' not in how: # works only for resampling to monthly values
            raise NotImplementedError

        years, months = df.index.year, df.index.month

        if len(years) == 0 or len(months) == 0:
            return None

        startday = datetime(years[0], months[0], 1)
        last_year, last_month = years[-1], months[-1]

        if last_month == 12:
            next_month, next_year = 1, last_year + 1
        else:
            next_month, next_year = last_month + 1, last_year

        days_last_month = (datetime(next_year, next_month, 1) - datetime(last_year, last_month, 1)).days
        endday = datetime(last_year, last_month, days_last_month)

        index_full = pd.date_range(start=startday, end=endday, freq='D')
        df_alldays = pd.DataFrame(index=index_full,
                                  data={'count_should': 1}).resample(how).sum()


        df_mean = df.resample(how).mean()

        df['count'] = 1
        df_mean['count_is'] = df[['count']].resample(how).sum()
        df_mean['count_should'] = df_alldays['count_should'] * threshold

        df_filtered = df_mean.loc[df_mean['count_is'] >= df_mean['count_should']]

        df_count = df_filtered[['count_should', 'count_is']]
        df_resampled = df_filtered.drop(['count_should', 'count_is'], axis=1)

    if month_middle:
        df_resampled = df_resampled.resample(how, loffset=pd.Timedelta(14, 'd')).mean().dropna()

    df_resampled.freq = how

    return df_resampled['input'], df_count


def df_conditional_temp_resample(df_in, resample_to, resample_threshold):
    """
    Wrapper around resample function to resample the input dataframe
    to the selected period, with the selected threshold for minimum
    temporal coverage in % per observation.

    Parameters
    -------
    ds_in : pandas.DataFrame
        DataFrame that should be resampled
    resample_to : 'str'
        Time period to resample to, eg M or D
    resample_threshold : float
        Minimum % of observations in the selected period to calculate a mean
        e.g. 0.1 = 10% valid observations, 1 = 100% valid observations

    Returns
    -------
    df_test_resampled : pd.DataFrame
        The resampled DataFrame
    """

    if df_in.empty:
        return df_in

    df = df_in.copy(True) #type: pd.DataFrame

    resampled_series = []
    for col in df.columns:
        if pd.isnull(df[col]).all():
            resampled = pd.Series(name=col)
        else:
            resampled, _ = conditional_temp_resample(df[col], resample_to, resample_threshold)
            resampled.name = col
        resampled_series.append(resampled)


    df_resampled = pd.concat(resampled_series, axis=1)

    if not df_resampled.columns.size == df.columns.size:
        return df_in

    df_resampled.freq = resample_to
    return df_resampled


def filter_by_quantiles(df_in, filter_col, lower=.1, upper=.9):
    '''
    Filters a data frame by dropping the >upper % and <lower % of all values.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input dataframe, that is filtered
    filter_col : str
        Name of the column based on which the filtering is performed
    lower : float
        Lower threshold quantile
    upper :
        Upper threshold quantile

    Returns
    -------
    masked_values : pd.DataSeries
        Data Series of flags that show which values are outside the defined threshold
    '''

    df_in = df_in[[filter_col]].copy(True).dropna() # type: pd.DataFrame

    upper_threshold = df_in.quantile(upper) #type: float
    lower_threshold = df_in.quantile(lower) #type: float
    df_in.loc[:, 'diff_flag'] = 1  # previous: np.nan
    index_masked = df_in.query('%s <= %f & %s >= % f'
                               % (filter_col, upper_threshold, filter_col, lower_threshold)).index

    df_in.loc[index_masked, ('diff_flag')] = 0

    masked_values = df_in['diff_flag']
    return masked_values


def crosscorr(can, ref, lag=0, method='spearman'):
    """
    Calculate the cross correlation between 2 pandas Series with the passed lag(s)

    Parameters
    ----------
    can : pd.Series
        Candidate data set (that is shifted)
    ref : pd.Series
        Reference data set (that is stationary)
    lag : int or list
        Lag(s) for which the correlations are calculated (can is shifted)
    method : str
        Correlation type, as in pd.corr ('pearson', 'spearman', etc.)

    Returns
    ----------
    ccorr : float or list
        Cross correlations between can and ref for the selected time lag(s)
    """

    if not isinstance(lag, Iterable):
        return ref.corr(can.shift(lag), method=method)
    else:
        return [ref.corr(can.shift(lag), method=method) for lag in lag]


def autocorr(can, lag=0, method='spearman'):
    """
    Calculate the auto correlation for a pandas Series with the passed lag(s)

    Parameters
    ----------
    can : pd.Series
        Candidate data set (that is shifted)
    lag : int or list
        Lag(s) for which the correlations are calculated
    method : str
        Correlation type, as in pd.corr ('pearson', 'spearman', etc.)

    Returns
    ----------
    acorr : float or list
        Auto correlations between can and ref for the selected time lag(s)
    """

    if not isinstance(lag, Iterable):
        return can.corr(can.shift(lag), method=method)
    else:
        return [can.corr(can.shift(lag), method=method) for lag in lag]

def flatten_dict(d):
    '''
    Flattens a dict of dicts so that the keys of 2 levels are merged into 1 key which contains the data of the
    sublevel dictionary.

    Parameters
    ----------
    d : dict
        Dict of dicts that will be flattened.


    Returns
    -------
    flat : dict
        Flattened dictionary

    '''
    def expand(key, value):
        if isinstance(value, dict):
            return [ (key + '_' + str(k), v) for k, v in flatten_dict(value).items() ]
        else:
            return [ (key, value) ]

    items = [ item for k, v in d.items() for item in expand(k, v) ]

    return dict(items)

def days_in_month(month, year, astype=int):
    '''
    Find the
    Parameters
    ----------
    month : int or iterable
        month(s) to look up
    year : int or iterable
        year(s) of the passed month(s)
    astype : dtype, optional (default: int)
        int or float
    Returns
    -------
    days : np.array of float
        Days in the passed months
    '''
    if isinstance(month, Iterable):
        if isinstance(year, int):
            year = np.full(len(month), year)

        days = []
        for m, y in zip(month, year):
            days.append(astype(calendar.monthrange(y, m)[1]))
        return np.array(days)
    else:
        return astype(calendar.monthrange(year, month)[1])


def mid_month_target_values(M):
    '''
    Midmonth target values, that are objectively
    chosen so that the average of the daily adjustments over a given month is
    equal to the monthly adjustment, when performing linear interpolation.

    J. Sheng, F. Zwiers (1998)
    An improved scheme for time-dependent boundary conditions in atmospheric general circulation models
    https://link.springer.com/content/pdf/10.1007%2Fs003820050244.pdf

    LUCIE A. VINCENT AND X. ZHANG (2001)
    Homogenization of Daily Temperatures over Canada (2001)
    https://link.springer.com/article/10.1007/s003820050244

    Parameters
    ---------
    M : np.array
        Monthly values that are used to find the midmonth target values
    '''


    A = np.array([[7./8., 1./8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [1./8., 6./8., 1./8., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 1./8., 6./8., 1./8., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 1./8., 6./8., 1./8., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 1./8., 6./8., 1./8., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 1./8., 6./8., 1./8., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 1./8., 6./8., 1./8., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 1./8., 6./8., 1./8., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 1./8., 6./8., 1./8., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 1./8., 6./8., 1./8., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 1./8., 6./8., 1./8.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1./8., 7./8.]])

    if A.shape[0] != M.shape[0]:
        raise ValueError(M.shape[0], 'Dimension Mismatch between A Matrix and M')

    T = np.matmul(np.linalg.inv(A), M)

    return np.squeeze(np.asarray(T))


def dt_freq(dt, ignore_M=False):
    '''
    Detect the (highest) frequency in the passed date time index, and return it
    in a format the pandas can interpret.
    Also tries to detect whether the found freq is monthly or higher (slow)

    Parameters:
    -------
    dt : pd.DatetimeIndex
        The index that is explored

    Returns:
    --------
    unit : str
        Unit string of the used resolution
    ignore_M : bool
        Do not attempt to find out if the daily freq is monthly or lower (slow)
    '''
    if dt.freq is not None: # if we have the info, it's trivial
        return dt.freq.n, dt.freq.name


    months, years = dt.month.values, dt.year.values
    first_month, first_year = months[0], years[0]
    last_month, last_year =  months[-1], years[-1]


    dm = days_in_month(months, years, astype=float) # days in month
    sm = np.array([int(timedelta(days=d).total_seconds()) for d in dm]) # seconds in month

    seconds = []
    for ns, sec_in_m in zip(np.diff(dt.values), sm):
        # month, day, hour, minute
        i = ns.astype('timedelta64[s]').astype(int)
        seconds.append(i)
    sampled = np.array(seconds)

    # now check if we can resample to months:
    resample = [('m', 60.), ('H', 60.), ('D', 24.)]

    highest_freq = 's'
    for freq, m in resample:
        if all(sampled % m == 0.): # can be resampled, up to daily from seconds
            sampled = sampled / m
            highest_freq = freq

    if highest_freq == 'D' and not any(sampled < 28) and not ignore_M:  # it could be monthly, check
        df_month = pd.DataFrame(index=pd.period_range('{}-{}'.format(first_year, first_month),
                                '{}-{}'.format(last_year, last_month), freq='M'))

        df_month['passed'] = np.nan
        df_month['days'] = np.nan
        for i, (y, m) in enumerate(zip(years, months)):
            df_month.loc['{}-{:02d}'.format(y,m),'passed'] = i
        df_month['passed'] = df_month['passed'].bfill()
        for y, m in zip(df_month.index.year, df_month.index.month):
            df_month.loc['{}-{:02d}'.format(y,m),'days'] = days_in_month(m, y, float)

        df_month = df_month[1:]
        m_should_days = df_month.groupby(['passed']).sum()['days'].values

        if len(m_should_days) == len(sampled) and all(m_should_days == sampled):
            # it is monthly or lower
            m_freq = min(df_month.groupby(['passed']).count()['days'].values)
            return m_freq, 'M'
        else:
            return min(sampled), highest_freq
    else:
        return min(sampled), highest_freq

if __name__ == '__main__':
    pass
