# -*- coding: utf-8 -*-

from datetime import datetime
import pandas as pd
import numpy as np
import calendar

def count_M(dt, first_should, last_should):
    """
    Count the number of valid months instances (not nan) per month in the passed index,
    and how many there should be.

    Parameters
    ----------
    dt : pd.DatetimeIndex
        Valid indices (no nans) in monthly resolution
    first_should : datetime
        The first month that SHOULD have a value (may not be in dt if it is nan)
        Day in datetime is ignored
    last_should : datetime
        The first day that SHOULD have a value (may not be in dt if it is nan)
        Day in datetime is ignored

    Returns
    -------
    df_count: pd.DataFrame
        A data frame that contains the calculated statistics
    """
    years, months = dt.year, dt.month

    if len(years) == 0 or len(months) == 0:
        return None

    index_full = pd.period_range(start=first_should, end=last_should, freq='M')

    max_month_counter, is_months_counter = {}, {}
    for month, indices_per_month in index_full.groupby(index_full.month).items():
        count = indices_per_month.size
        max_month_counter[int(month)] = count

    df_count = pd.DataFrame.from_dict(max_month_counter, orient='index')
    df_count = df_count.rename(columns={0: 'months_max'})
    df_count.index.name = 'month'

    for month, indices_per_month in dt.groupby(dt.month).items():
        count = indices_per_month.size
        is_months_counter[month] = count

    df_max = pd.DataFrame.from_dict(is_months_counter, orient='index')
    df_max.rename(columns={0: 'months_is'}, inplace=True)

    df_count = pd.concat([df_count, df_max], axis=1)

    df_count['coverage'] = df_count['months_is'] / df_count['months_max']


    return df_count


def count_D(dt, first_should, last_should):
    """
    Count the number of days per month in the passed index, and how many there
    should be.

    Parameters
    ----------
    dt : pandas.DateTimeIndex
        Valid index in daily resolution
    first_should : datetime
        The first day that SHOULD have a value (may not be in dt if it is nan)
    last_should : datetime
        The first day that SHOULD have a value (may not be in dt if it is nan)

    Returns
    -------
    df_count: pd.DataFrame
        A data frame that contains the calculated statistics
    """

    years, months, days = dt.year, dt.month, dt.day

    if len(years) == 0 or len(months) == 0:
        return None

    index_full = pd.date_range(start=first_should, end=last_should, freq='D')

    max_days_counter, is_days_counter = {}, {}
    for month, indices_per_month in index_full.groupby(index_full.month).items():
        count = indices_per_month.size
        max_days_counter[int(month)] = count

    df_count = pd.DataFrame.from_dict(max_days_counter, orient='index')
    df_count = df_count.rename(columns={0: 'days_max'})
    df_count.index.name = 'month'

    for month, indices_per_month in dt.groupby(dt.month).items():
        count = indices_per_month.size
        is_days_counter[month] = count

    df_max = pd.DataFrame.from_dict(is_days_counter, orient='index')
    df_max.rename(columns={0: 'days_is'}, inplace=True)

    df_count = pd.concat([df_count, df_max], axis=1)

    df_count['coverage'] = df_count['days_is'] / df_count['days_max']

    return df_count


def compare_temp_cov(ds1, ds2, start1, end1, start2, end2, resolution='D',
                     coverdiff_max=0.5):
    '''
    Compare the coverage of the first series to the second series.
    They are adjacent with the break time as the start of the second series and
    the end of the first series.

    Parameters
    ----------
    ds1 : pd.Series
        The data series before the break time
    ds2 : pd.Series
        The data series after the break time
    start : datetime
        The first date of the first set that COULD/SHOULD have a valid value
    end1 : datetime
        The last date of the first set that COULD/SHOULD have a valid
        observation
    end : datetime
        The last date of the second set that COULD/SHOULD have a valid value
    resolution : str
        D or M, which resolution the input is in
    coverdiff_max : float, between 0 and 1
        THe maximum difference for day/month that the coverage of the 2 groups
        may differ.



    Returns
    -------
    success : bool
        If the test succeeded or not, false means that months are to drop
    df_cover_1 : pd.DataFrame
        The coverage of days/months per month of the first input set
    df_cover_2 : pd.DataFrame
        The coverage of days/months per month of the second input set
    cover_diff : pd.DataFrame
        Differenes in temporal coverage of the 2 groups
    months_to_drop : np.array
        List of months that have to be removed from both sets (if they exist)

    '''


    # 1) for each month in can_adj 2 (independent of the year), find the number
    # of observations (days for daily data, months for monthly)

    # 2) for each month in can_adj 1 (independent of the year), find the number
    # of observations (days for daily data, months for monthly)

    # 3) Check if ALL months in 2) are also covered in 1)
    #   eg all 12 from ds2 are also covered in ds1, or [5,6,7,8] in ds2 is also
    #   exactly [5,6,7,8] in ds2
            # if not this fails, or we could remove the months that are in ds2 but
            # not in ds1 from ds2

    # 4) check if each months from 2) has at least 20% as much values in 1)
        # else fails



    ds1 = ds1.copy(True).dropna()
    ds2 = ds2.copy(True).dropna()

    if resolution == 'D':
        df_count1 = count_D(ds1.index, start1, end1)
        df_count2 = count_D(ds2.index, start2, end2)
    elif resolution == 'M':
        ds1.index = ds1.index.to_period('M')
        ds2.index = ds2.index.to_period('M')
        df_count1 = count_M(ds1.index, start1, end1)
        df_count2 = count_M(ds2.index, start2, end2)
    else:
        raise ValueError('Resolution is not supported')

    if df_count1 is None or df_count2 is None:
        print('One dataset is empty')
        return False, df_count1, df_count2, None, None, None

    cover_diff = pd.DataFrame()

    cover_diff['diff'] = abs(df_count1['coverage'] - df_count2['coverage'])


    # drop the months where 1 of the 2 sets has no coverage
    cond1 = (cover_diff['diff'] > coverdiff_max) # dont select where diff to high
    # if both have none, its ok
    cond2 = pd.isna(df_count1['coverage']) & pd.notna(df_count2['coverage'])
    cond3 = pd.isna(df_count2['coverage']) & pd.notna(df_count1['coverage'])
    #cond0 = cover_diff['diff'].apply(np.isnan) # dont select nans



    cover_diff['drop_month_flag'] = False
    cover_diff.loc[(cond1 | cond2 | cond3), 'drop_month_flag'] = True


    months_to_drop = cover_diff.loc[cover_diff['drop_month_flag']==True].index.values


    if months_to_drop.size != 0:
        return False, df_count1, df_count2, cover_diff, months_to_drop

    else:
        return True, df_count1, df_count2, cover_diff, months_to_drop


def drop_months_data(df, drop_months):
    '''
    Remove all values in the passed months in the dataframe

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        Time series data, that will be filtered
    drop_months : np.array
        List of integers for months that will be removed eg [1,2] for Jan and Feb

    Returns
    -------
    df_filtered : pandas.DataFrame or pandas.Series
        Input data frame without the selected months
    '''
    dat = df.copy(True)
    for month in drop_months:
        dat = dat[dat.index.month != month]
    return dat







if __name__ == '__main__':
    import numpy as np


    monthly= False
    rand1 = 0.6 # percent (0-1)
    rand2 = 0.1 # percent (0-1)

    if monthly:
        start1 = datetime(2000,4,30)
        end1 = datetime(2005,9,30)
        start2 = datetime(2005,10,31)
        end2 = datetime(2008,8, 31)

        index1 = pd.DatetimeIndex(start=start1, end=end1, freq='M')
        index2 = pd.DatetimeIndex(start=start2, end=end2, freq='M')

    else:
        start1 = datetime(2000, 4, 3)
        end1 = datetime(2005, 10, 2)
        start2 = datetime(2005, 10, 3)
        end2 = datetime(2008, 8, 3)

        index1 = pd.DatetimeIndex(start=start1, end=end1, freq='D')
        index2 = pd.DatetimeIndex(start=start2, end=end2, freq='D')

    data1 = np.array([np.random.rand(index1.size)])
    data2 = np.array([np.random.rand(index2.size)])

    rand1 = int(data1.size * rand1)
    rand2 = int(data2.size * rand2)

    data1.ravel()[np.random.choice(data1.size, rand1, replace=False)] = np.nan
    data2.ravel()[np.random.choice(data2.size, rand2, replace=False)] = np.nan


    ds1 = pd.Series(index=index1, data=data1[0]).dropna()
    ds2 = pd.Series(index=index2, data=data2[0]).dropna()


    succ, df_count1, df_count2, cover_diff, drop_months = \
        compare_temp_cov(ds1, ds2, start1, end1, start2, end2,
                         resolution='M' if monthly else 'D',
                         coverdiff_max=0.5)

    ds1 = drop_months_data(ds1, drop_months)
    ds2 = drop_months_data(ds2, drop_months)


    succ, df_count1, df_count2, cover_diff, drop_months = \
        compare_temp_cov(ds1, ds2, start1, end1, start2, end2,
                         resolution='M' if monthly else 'D',
                         coverdiff_max=0.5)

    pass







