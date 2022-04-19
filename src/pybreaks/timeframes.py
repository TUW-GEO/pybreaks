# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

class TsTimeFrames(object):
    """
    Module created time frames from passed break times and start / end dates
    Allows location dependent time frames from gpis
    """
    def __init__(self, start, end, breaktimes, grid=None, ignore_position=True,
                 skip_breaktimes=None, min_set_days=None, base_breaktime=None):
        """
        Parameters
        ----------
        start: datetime
            Start date of the time series / data set
        end : datetime
            End date of the time series / data set
        breaktimes : np.array
            Collection of break time strings and conditions, for location
            dependent time frames enter conditions the following way:
                np.array(['2007-10-01', '2002-06-19',
                        {'lat >= -37 and lat <= 37': ['1998-01-01']},
                        '1987-07-09'])
        grid : pygeogrids.BasicGrid or None
            ONLY NECESSARY FOR LOCATION DEPENDANT TIME FRAMES, grid where the
             gpis are on
        ignore_position : bool
            Set true if all break times should be treated as independent from
            their location
        skip_breaktimes : list or None
            Forces list of indices of breaktimes that should be ignored
        min_set_days : int or None
            Set the minimum size of observations per set(
            timeframe_start - breaktime, breaktime - timeframe_end).
            If this value is not reached, the corresponding break time is ignored.
            If skipping is also selected, filtering happens AFTER skipping.
        base_breaktime : int or None

            ... |3      |2      |1      |0

            Identifies the base breaktime, for which the time frame is created first.
            eg 0 would start with last period and return periods towards start.
            eg -1 would start with first period and return periods towards end
            eg 2 would start with 3rd last period, return other periods towards the end,
               then return other periods towards start.

        Attributes
        -------
        breaktimes : np.array of strings
            Collection of breaktimes and according conditions
        ranges : np.array of strings
            Collection of dataset range values (start date and end date)
        skip_breaktimes : list
            List of indices of breaktimes that should be ignored
        ignore_position : bool
            Whether the postional conditions of breaks times should be ignored or not
        grid : pygeogrids.BasicGrid
            Grid that is used for evaluation of conditions for location dependent break times
        """

        self.breaktimes = breaktimes
        self.ranges = [start.strftime('%Y-%m-%d') if isinstance(start, datetime) else start,
                       end.strftime('%Y-%m-%d') if isinstance(end, datetime) else end]
        self.ignore_position = ignore_position

        if not self.ignore_position:
            self.grid = grid

        if not self.gpi_dep_times():
            self.ignore_position = True
        else:
            self.ignore_position = ignore_position

        if self.ignore_position:
            self.breaktimes = self.max_breaktimes()

        self.skip_breaktimes = skip_breaktimes
        if skip_breaktimes:
            self.breaktimes = self.reduce_breaktimes(self.skip_breaktimes)
#
        self.min_set_days = min_set_days
        if self.min_set_days:
            self.breaktimes = self._filter_short_timeframes(self.min_set_days,
                                                            self.breaktimes)

        if self.skip_breaktimes and not self.ignore_position and self.gpi_dep_times():
            raise Exception('Skipping (skipping) breaktimes for gpi dependent '
                            'timeframes ambiguous')

        if self.min_set_days and not self.ignore_position and self.gpi_dep_times():
            raise Exception('Skipping (min_set) breaktimes for gpi dependent'
                            ' timeframes ambiguous')

        self.base_breaktime = base_breaktime

        if (self.base_breaktime is not None) and \
                (skip_breaktimes is not None or skip_breaktimes is not None):
            warnings.warn('Attention: Resorting will be performed AFTER'
                          ' reducing the break times!!')

    def __len__(self):
        return len(self.breaktimes)


    def _base_sort(self, times, base_breaktime):
        '''
        Sort the break times according to the base period (start).

        Parameters
        -------
        times : np.array
            List of break times that should be sorted according to the base period
        base_breaktime : int
            Identifies the base breaktime, for which the time frame is created first.
            eg 0 would start with the last period and return other periods
            towards the beginning
            eg -1 would start with the first period and return other periods
            towards the end
            eg 2 would start with the 3rd last period, return other periods
            towards the end, then return other periods towards the beginning.

        Returns
        -------
        bt_before : np.array
            Break times before the base break time
        bt after : np.array
            Break times after the base break time
        '''

        #sort the break times in descending order
        breaktimes = np.flipud(np.sort(times))
        # take the base break time and all (sorted) bts after it
        breaktimes_before_base = breaktimes[base_breaktime + 1:]
        # take all (sorted) break times before the base break time
        breaktimes_after_base = np.flipud(breaktimes[:base_breaktime + 1])

        return breaktimes_before_base, breaktimes_after_base

    def _filter_short_timeframes(self, days_min, breaktimes):
        '''
        Iterates over break times and removes break times that lead to time frames
        smaller than the passed threshold.

        Parameters
        -------
        breaktimes : list
            List of breaktimes that should be filtered
        days_min : int
            minimum number of days that a set (timeframe_start - breaktime,
            breaktime - timeframe_end) must contain.

        Returns
        -------
        breaktimes : list
            The filtered list of breaktimes
        '''

        candidate_range = [self.ranges[1], self.ranges[0]]
        breaktimes = sorted(candidate_range + list(breaktimes), reverse=True)

        while True:
            restart = False
            for i in range(1, len(breaktimes) - 1):
                end, breaktime, start = breaktimes[i - 1], breaktimes[i], breaktimes[i + 1]
                if pd.date_range(start=str(breaktime), end=str(end), freq='D').size < days_min:
                    breaktimes.remove(breaktime)
                    restart = True
                    break
                if pd.date_range(start=str(start), end=str(breaktime), freq='D').size < days_min:
                    breaktimes.remove(breaktime)
                    restart = True
                    break
            if not restart:
                break

        return breaktimes[1:-1]

    def timeframes_from_breaktimes(self, breaktimes):
        '''
        Takes subsequent breaktimes and creates time frames using the dataset
        start and end

        Parameters
        ----------
        breaktimes : np.array
            List of break times that are used beside the dataset range for
            creating time frames
        base_breaktime : int
            Identifies the base breaktime, for which the time frame is created
            first.
            eg 0 would start with the last period and return other periods towards
            the beginning
            eg -1 would start with the first period and return other periods towards
            the end
            eg 2 would start with the 3rd last period, return other periods
            towards the end, then return other periods towards the beginning.

        Returns
        -------
        timeframes_left : np.array
            Tuples of datetimes that indicate the start and end of each time
            frame left of the base break time
        timeframes_right : np.array
            Tuples of datetimes taht indicat the start and end of each time
            frame right of the base break time
        breaktimes : np.array
            The (new) breaktimes (in case they were re-ordered), otherwise the
            input breaktimes
        '''
        if isinstance(breaktimes, list):
            breaktimes = np.array(breaktimes)
        if self.base_breaktime is None:
            times = {'left': [self.ranges[1]] + breaktimes.tolist() + [self.ranges[0]], 'right':None}
        else:
            bt_before, bt_after = self._base_sort(breaktimes, self.base_breaktime)
            breaktimes = np.concatenate((bt_after, bt_before))
            times = {'right': [bt_before[0]] + bt_after.tolist() + [self.ranges[1]],
                     'left': [bt_after[0]] + bt_before.tolist() + [self.ranges[0]]}

        timeframes = {'left':[], 'right':[]}
        for side, time in times.items():
            if time is None:
                timeframes[side] = None
                continue
            for i in range(len(time)-2):
                timeframe = time[i:i+3]
                if side == 'right':
                    [starttime, breaktime, endtime] = timeframe[0:3]
                else:
                    [endtime, breaktime, starttime] = timeframe[0:3]
                timeframes[side].append([starttime, endtime])

        timeframes_left = np.array(timeframes['left'])
        timeframes_right =  None if timeframes['right'] is None else np.array(timeframes['right'])

        return timeframes_left, timeframes_right, breaktimes

    def reduce_breaktimes(self, indices_to_ignore):
        '''
        Reduces cci_breaks version breaktimes to the breaktimes that are not
        in skip_breaktimes

        Parameters
        -------
        indices_to_ignore : list
            Integeres of breaktimes that should be ignored

        Returns
        -------
         breaktimes : np.array
            The reduced break times
        '''
        return np.array([self.breaktimes[i] for i in range(len(self.breaktimes))
                if i not in indices_to_ignore])

    def filter_breaktimes(self, gpi):
        '''
        Takes the objects break times and filters them according to the postion
        of the passed gp on the objects grid

        Parameters
        ----------
        gpi : int
            Filtering of the objects break times is done based on this GPI

        Returns
        -------
        filtered_breaktimes : np.array
            The list of filtered break times
        '''

        (lon, lat) = self.grid.gpi2lonlat(gpi)
        timeset = self.breaktimes
        return_times = []
        for i, time in enumerate(timeset):
            if isinstance(time, dict):
                for condition, value in time.items():
                    if eval(condition):
                        for v in value:
                            if v not in return_times:
                                return_times.append(v)
            else:
                return_times.append(time)
        return np.array(return_times)

    @staticmethod
    def as_string(datetimes):
        '''
        Turns datetime objects to strings

        Parameters
        -------
        datetimes : list or datetime
            Dates to convert

        Returns
        -------
        datestrings: list
            The converted dates
        '''
        return [str(time.date()) for time in datetimes]

    @staticmethod
    def as_datetimes(datestrings):
        '''
        Turns string objects to datetimes

        Parameters
        -------
        datestrings : list or str
            Strings to convert

        Returns
        -------
        datetimes: list
            The converted strings
        '''
        return [datetime.strptime(time, '%Y-%m-%d') for time in datestrings]

    def gpi_dep_times(self):
        """
        Checks if current break times and time frames are positional dependent

        Returns
        -------
        status : bool
            True if breaktimes are position dependent else False
        """

        if any([isinstance(x, dict) for x in self.breaktimes]):
            return True
        else:
            return False

    def max_breaktimes(self):
        '''
        Return the maximum of break times (if there are multiple, contradicting
        conditions, select the condition that contains most values)

        Returns
        -------
        breaktimes : np.array
            The maximum breaktimes
        '''

        union = []
        for i, d in enumerate(self.breaktimes):
            if isinstance(d, dict):
                longest_condition_name = None
                longest_condition_size = 0
                for condition, values in d.items():
                    if len(values) >= longest_condition_size:
                        longest_condition_name = condition
                        longest_condition_size = len(values)
                for value in d[longest_condition_name]:
                    union.append(value)
            else:
                union.append(d)

        return np.array(union)

    def get_times(self, gpi=None, as_datetime=False):
        '''
        Get the break times, time frames and range (for the passed gpi, if they
        are location dependent)

        Parameters
        -------
        gpi : int, optional (default: None)
            GPI to get times for
        as_datetime : bool
            If True, return times as datetime objects

        Returns
        -------
        times : dict
            Break times, time frames and range for the GPI
        '''

        return_times = {'ranges': self.ranges}

        if not self.gpi_dep_times():
            return_times['breaktimes'] = self.breaktimes
        else:
            if not gpi:
                breaktimes = self.max_breaktimes()
                return_times['breaktimes'] = breaktimes
            else:
                breaktimes = self.filter_breaktimes(gpi)
                return_times['breaktimes'] = breaktimes


        timeframes_left, timeframes_right, breaktimes = \
            self.timeframes_from_breaktimes(return_times['breaktimes'])

        if timeframes_right is None:
            return_times['timeframes'] = timeframes_left
        else:
            return_times['timeframes'] = np.concatenate((timeframes_right, timeframes_left))

        return_times['breaktimes'] = breaktimes


        if as_datetime:
            return_times['breaktimes'] = self.as_datetimes(return_times['breaktimes'])
            return_times['timeframes'] = [self.as_datetimes(timeframe) for timeframe in return_times['timeframes']]
            return_times['ranges'] = self.as_datetimes(return_times['ranges'])

        for name, data in return_times.items():
            return_times[name] = np.array(data)

        return return_times

    def get_index(self, gpi, time, gpi_times=None):
        '''
        Find the index of a time for a gpi in the current objects breaktimes or timeframes
        Find out where a certain time lays in the objects times

        Parameters
        ----------
        gpi : int
            GPI for location dependent times
        time : np.array or datetime or str
            The break time or time frame the index should be found for
        gpi_times : dict
            A time collection which is searched for the passed time to get the index

        Returns
        -------
        index : int
            The index of the time in the times collection
        '''

        if type(time) in (np.ndarray, list):
            if all([isinstance(t, datetime) for t in time]):
                time = self.as_string(time)
            if gpi_times:
                times = gpi_times['timeframes']
            else:
                times = self.get_times(gpi, as_datetime=False)['timeframes']
            return np.where((times == time)[:, 1])[0][0]
        else:
            if isinstance(time, datetime):
                time = self.as_string([time])[0]
            if gpi_times:
                times = gpi_times['breaktimes']
            else:
                times = self.get_times(gpi, as_datetime=False)['breaktimes']
            return np.where(times == time)[0][0]

    def breaktime_for_timeframe(self, gpi, timeframe):
        '''
        Get the according break time for the passed time frame

        Parameters
        ----------
        gpi : int
            GPI to get breaktime for
        timeframe : tuple
            timeframe to get breaktime for

        Returns
        -------
        breaktime : datetime or str
            Breaktime for the selected timeframe at the location
        '''
        times = self.get_times(gpi, as_datetime=False)['breaktimes']
        if all([isinstance(time, datetime) for time in timeframe]):
            return self.as_datetimes([times[self.get_index(gpi, np.array(self.as_string(timeframe)))]])[0]
        else:
            return times[self.get_index(gpi, timeframe)]

    def timeframe_for_breaktime(self, gpi, breaktime):
        '''
        Get the according time frame for the passed break time at the passed location

        Parameters
        ----------
        gpi : int or None
            GPI to get time frame for
        breaktime : datetime.datetime or str
            breaktime to get timeframe for

        Returns
        -------
        timeframe : tuple
            Time frame for the selected breaktime at the location
        '''
        times = self.get_times(gpi, as_datetime=False)['timeframes']
        if isinstance(breaktime, datetime):
            return self.as_datetimes(times[self.get_index(gpi, breaktime)])
        else:
            return times[self.get_index(gpi, breaktime)]

    def get_adjacent(self, reference, shift, gpi=None):
        '''
        Gets the adjacent times for the passed gpi

        Parameters
        -------
        reference : datetime or str or tuple
            the reference break time or time frame
        shift : int
            index change for time to return,
                eg 1 -> return next time (AFTER passed one (earlier in time)),
                -1 -> return previous time (BEFORE passed one (later in time)) etc..
        gpi : int, optional
             GPI to get times for

        Returns
        ------
        times : list
            The adjacent break times if reference was a break time, time frame
            if reference was a time frame
        '''

        times = self.get_times(gpi, as_datetime=False)
        if self.base_breaktime is not None:
            times['breaktimes'] = self._sort(times['breaktimes']) # sorted
            times['timeframes'] = self._sort(times['timeframes']) # sorted
        else:
            pass # unsorted


        if type(reference) in (np.ndarray, list): # timeframe
            index = self.get_index(gpi, reference, times)

            if all([isinstance(time, datetime) for time in reference]):
                return self.as_datetimes(times['timeframes'][index + shift])
            else:
                return times['timeframes'][index + shift]
        else: # breaktime
            # extend with the first and last date
            breaktimes = np.concatenate([[times['ranges'][1]],
                                         times['breaktimes'],
                                         [times['ranges'][0]]])

            times['breaktimes'] = breaktimes

            index = self.get_index(gpi, reference, times)

            if isinstance(reference, datetime):
                return self.as_datetimes([breaktimes[index + shift]])[0]
            else:
                return breaktimes[index + shift]

    def _sort(self, times):
        '''
        Sorts the passed array of breaktimes or timeframes in ASCENDING ORDER

        Parameters
        ----------
        times : np.array
            Array of breaktimes, or array of lists of time frames.

        Returns
        -------
        sorted : np.array
            Sorted array of breaktimes or timeframes

        '''
        times = np.copy(times)

        if len(times.shape) == 1:
            sorted = np.sort(times)
        else:
            sorted = times[times[:, 0].argsort()]

        return sorted


if __name__ == '__main__':
    from datetime import datetime
    times = np.array(['2012-07-01', '2011-10-05', '2010-01-15', '2007-10-01', '2007-01-01', '2002-06-19',
                      '1998-01-01', '1991-08-15', '1987-07-09'])

    otime=TsTimeFrames(start=datetime(1978,10,26), end=datetime(2016,12,31),
                       breaktimes=times, min_set_days=None, skip_breaktimes=None, base_breaktime=1)
    times = otime.get_times()

    otime.breaktime_for_timeframe(None, timeframe=np.array(['2007-10-01','2011-10-05']))
    otime.timeframe_for_breaktime(None, breaktime='2011-10-05')