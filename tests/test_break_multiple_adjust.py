# -*- coding: utf-8 -*-

from breakadjustment.break_multi import TsRelMultiBreak
import numpy as np
import pandas as pd
from datetime import datetime
import os


def read_test_data():
    '''Read real data from the test data folder'''
    start = datetime(1998,1,1)
    end = datetime(2007,1,1)

    path = os.path.join('test-data', 'csv_ts')
    #for file in os.listdir(path)
    file = 'data_431790.csv'
    ts = pd.read_csv(os.path.join(path, file), index_col=0, parse_dates=True) / 100

    return ts[start:end]

