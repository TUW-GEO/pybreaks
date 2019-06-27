# -*- coding: utf-8 -*-

import pandas as pd
from trash.MD_interpolate import MDInterpolate
import numpy.testing as nptest

def create_test_data():
    index = pd.DatetimeIndex(start='2000-01-01', end='2000-12-31', freq='D')
    test_data = pd.Series(index=index, data=[2.] * 366)
    return test_data

def init_obj():
    ts = create_test_data()
    ts_month = ts.resample('M').mean()
    return MDInterpolate(ts_month)


def test_interpolate():
    obj = init_obj()
    inter = obj.interpolate(method='linear')
    nptest.assert_almost_equal(inter, [2] * 366) # [1,2,3] mean = 2 and inter daily
    inter = obj.interpolate(method='nearest')
    nptest.assert_almost_equal(inter, [2] * 366)
    inter = obj.interpolate(method='spline', order=4)
    nptest.assert_almost_equal(inter, [2] * 366)
    inter = obj.interpolate(method='polynomial', order=5)
    nptest.assert_almost_equal(inter, [2] * 366)



if __name__ == '__main__':
    test_interpolate()