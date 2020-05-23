# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
from pybreaks.timeframes import TsTimeFrames
from smecv_grid.grid import SMECV_Grid_v042


def test_std_get_times():
    # default

    intimes = np.array(['2012-07-01', '2011-10-05', '2010-01-15', '2007-10-01', '2007-01-01', '2002-06-19',
                      '1998-01-01', '1991-08-15', '1987-07-09'])

    otime=TsTimeFrames(start=datetime(1978,10,26), end=datetime(2016,12,31),
                       breaktimes=intimes)

    times = otime.get_times()
    assert(all(times['breaktimes'] == intimes))
    assert(all(times['timeframes'][1] == np.array(['2010-01-15','2012-07-01'])))



def test_as_datetime():
    # default

    intimes = np.array(['2012-07-01', '2011-10-05', '2010-01-15', '2007-10-01', '2007-01-01', '2002-06-19',
                      '1998-01-01', '1991-08-15', '1987-07-09'])

    otime=TsTimeFrames(start=datetime(1978,10,26), end=datetime(2016,12,31),
                       breaktimes=intimes)

    times = otime.get_times(as_datetime=True)
    assert(all(times['timeframes'][0] == np.array([datetime(2011,10,5), datetime(2016,12,31)])))


def test_gpi_dep_get_times_no_base():
    # no base break time
    intimes = np.array(['2007-10-01', '2002-06-19', {'lat >= -37 and lat <= 37': ['1998-01-01']}, '1987-07-09'])


    grid = SMECV_Grid_v042()

    otime=TsTimeFrames(start=datetime(1978,10,26), end=datetime(2016,12,31),
                       breaktimes=intimes, min_set_days=None, skip_breaktimes=None, ignore_position=False,
                       grid=grid)
    # Inside
    times = otime.get_times(gpi=384268, as_datetime=False)
    assert(all(times['breaktimes'] == np.array(['2007-10-01', '2002-06-19', '1998-01-01', '1987-07-09'])))
    assert(all(times['timeframes'][1] == np.array(['1998-01-01','2007-10-01'])))
    assert(all(times['timeframes'][2] == np.array(['1987-07-09','2002-06-19'])))
    assert(all(times['ranges'] == np.array(['1978-10-26', '2016-12-31'])))

    #Ouside
    times = otime.get_times(gpi=210678, as_datetime=False)
    assert(all(times['breaktimes'] == np.array(['2007-10-01', '2002-06-19', '1987-07-09'])))
    assert(all(times['timeframes'][1] == np.array(['1987-07-09','2007-10-01'])))


def test_timeframes_gpi_dep_base():
    # base break time
    intimes = np.array(['2010-01-01', '2007-10-01', '2002-06-19', {'lat >= -37 and lat <= 37': ['1998-01-01']}, '1987-07-09'])


    grid = SMECV_Grid_v042()

    otime=TsTimeFrames(start=datetime(1978,10,26), end=datetime(2016,12,31),
                       breaktimes=intimes, min_set_days=None, skip_breaktimes=None, ignore_position=False,
                       grid=grid, base_breaktime=2)
    # Inside
    times = otime.get_times(gpi=384268, as_datetime=False)
    assert(all(times['breaktimes'] == np.array(['2002-06-19', '2007-10-01', '2010-01-01', '1998-01-01', '1987-07-09'])))
    assert(all(times['timeframes'][0] == np.array(['1998-01-01', '2007-10-01'])))
    assert(all(times['timeframes'][1] == np.array(['2002-06-19', '2010-01-01'])))
    assert(all(times['timeframes'][3] == np.array(['1987-07-09', '2002-06-19'])))
    assert(all(times['ranges'] == np.array(['1978-10-26', '2016-12-31'])))

    #Ouside
    times = otime.get_times(gpi=210678, as_datetime=False)
    assert(all(times['breaktimes'] == np.array(['2002-06-19', '2007-10-01', '2010-01-01', '1987-07-09'])))
    assert(all(times['timeframes'][0] == np.array(['1987-07-09', '2007-10-01'])))
    assert(all(times['timeframes'][1] == np.array(['2002-06-19', '2010-01-01'])))
    assert(all(times['timeframes'][2] == np.array(['2007-10-01', '2016-12-31'])))
    assert(all(times['timeframes'][3] == np.array(['1978-10-26', '2002-06-19'])))



def test_timeframes_from_breaktimes_skip():
    # default

    intimes = np.array(['2012-07-01', '2011-10-05', '2010-01-15', '2007-10-01', '2007-01-01', '2002-06-19',
                      '1998-01-01', '1991-08-15', '1987-07-09'])

    otime=TsTimeFrames(start=datetime(1978,10,26), end=datetime(2016,12,31),
                       breaktimes=intimes, skip_breaktimes=[1,2], min_set_days=None, base_breaktime=None)

    times = otime.get_times()
    assert(all(times['breaktimes'] == np.array(['2012-07-01', '2007-10-01', '2007-01-01', '2002-06-19',
                                            '1998-01-01', '1991-08-15', '1987-07-09'])))
    assert(all(times['timeframes'][0] == np.array(['2007-10-01','2016-12-31'])))
    assert(all(times['timeframes'][3] == np.array(['1998-01-01', '2007-01-01'])))
    assert(all(times['ranges'] == np.array(['1978-10-26', '2016-12-31'])))



def test_timeframes_from_breaktimes_mindata():
    # default

    intimes = np.array(['2012-07-01', '2007-10-01', '2007-01-01', '2002-06-19', '1998-01-01'])

    otime=TsTimeFrames(start=datetime(1978,10,26), end=datetime(2016,12,31),
                       breaktimes=intimes, skip_breaktimes=None, min_set_days=365, base_breaktime=None)

    times = otime.get_times()
    assert(all(times['breaktimes'] == np.array(['2012-07-01', '2007-01-01', '2002-06-19', '1998-01-01'])))
    assert(all(times['timeframes'][0] == np.array(['2007-01-01','2016-12-31'])))
    assert(all(times['timeframes'][1] == np.array(['2002-06-19', '2012-07-01'])))
    assert(all(times['timeframes'][2] == np.array(['1998-01-01', '2007-01-01'])))

    assert(all(times['ranges'] == np.array(['1978-10-26', '2016-12-31'])))


def test_adjacent():
    intimes = np.array(['2012-07-01', '2007-10-01', '2007-01-01', '2002-06-19', '1998-01-01'])

    otime = TsTimeFrames(start=datetime(1978,10,26), end=datetime(2016,12,31),
                         breaktimes=intimes, skip_breaktimes=None, min_set_days=None, base_breaktime=None)

    assert(otime.get_adjacent(reference='2007-01-01',shift=-1)=='2007-10-01')
    assert(otime.get_adjacent(reference='2007-01-01',shift=-2)=='2012-07-01')
    assert(otime.get_adjacent(reference='2007-01-01',shift=1)=='2002-06-19')
    assert(otime.get_adjacent(reference='2007-01-01',shift=2)=='1998-01-01')


def test_timeframes_from_breaktimes_and_viceversa_base_breaktime():
    intimes = np.array(['2012-07-01', '2007-10-01', '2007-01-01', '2002-06-19', '1998-01-01'])

    otime=TsTimeFrames(start=datetime(1978,10,26), end=datetime(2016,12,31),
                       breaktimes=intimes, skip_breaktimes=None, min_set_days=None, base_breaktime=1)

    assert(all(otime.timeframe_for_breaktime(None, '2007-01-01')==np.array(['2002-06-19', '2007-10-01'])))
    assert(otime.breaktime_for_timeframe(None, np.array(['2002-06-19', '2007-10-01'])) == '2007-01-01')



if __name__ == '__main__':
    test_timeframes_from_breaktimes_mindata()

    test_std_get_times()
    test_as_datetime()
    test_gpi_dep_get_times_no_base()
    test_timeframes_gpi_dep_base()
    test_timeframes_from_breaktimes_skip()
    test_adjacent()
    test_timeframes_from_breaktimes_and_viceversa_base_breaktime()



