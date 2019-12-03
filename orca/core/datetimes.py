import itertools

import pandas as pd

from .common import default_session
from .internal import _ConstantSP
from .utils import sql_select, get_orca_obj_from_script


def _orca_unary_op(func):
    @property
    def ufunc(self):
        return self._unary_op(func)
    return ufunc


def _orca_logical_unary_op(func):
    @property
    def lufunc(self):
        return self._logical_unary_op(func)
    return lufunc


class Timestamp(object):
    
    def __init__(self, ts_input, freq=None, tz=None, unit=None, year=None, month=None, day=None, hour=None,
                 minute=None, second=None, microsecond=None, nanosecond=None, tzinfo=None, session=default_session()):
        if isinstance(ts_input, pd.Timestamp):
            data = ts_input
        else:
            data = pd.Timestamp(ts_input, freq, tz, unit, year, month, day, hour, minute, second, microsecond, nanosecond, tzinfo)
        self._session = session
        self._internal = data.to_numpy()
        self._var = _ConstantSP.upload_obj(session, self._internal)

    @property
    def _var_name(self):
        return self._var._var_name

    def __repr__(self):
        return self._internal.__repr__()

    def __str__(self):
        return self._internal.__str__()


class DatetimeProperties(object):

    date = _orca_unary_op("date")
    time = _orca_unary_op("second")
    year = _orca_unary_op("year")
    month = _orca_unary_op("monthOfYear")
    day = _orca_unary_op("dayOfMonth")
    hour = _orca_unary_op("(t->datetime(t).hour())")
    minute = _orca_unary_op("(t->datetime(t).minuteOfHour())")
    second = _orca_unary_op("(t->datetime(t).secondOfMinute())")
    microsecond = _orca_unary_op("(t->nanotimestamp(t).microsecond())")
    nanosecond = _orca_unary_op("(t->nanotimestamp(t).nanosecond())")    # TODO: nanosecond support other dtypes
    dayofyear = _orca_unary_op("dayOfYear")
    weekofyear = _orca_unary_op("weekOfYear")
    dayofweek = _orca_unary_op("weekday{,false}")
    weekday = dayofweek
    quarter = _orca_unary_op("quarterOfYear")

    is_month_start = _orca_logical_unary_op("isMonthStart")
    is_month_end = _orca_logical_unary_op("isMonthEnd")
    is_quarter_start = _orca_logical_unary_op("isQuarterStart")
    is_quarter_end = _orca_logical_unary_op("isQuarterEnd")
    is_year_start = _orca_logical_unary_op("isYearStart")
    is_year_end = _orca_logical_unary_op("isYearEnd")
    is_leap_year = _orca_logical_unary_op("isLeapYear")


class DatetimeMethods(DatetimeProperties):

    def __init__(self, s):
        self._s = s

    def _logical_unary_op(self, func):
        from .operator import BooleanExpression
        return BooleanExpression(self._s, None, func, 1)

    def _unary_op(self, func):
        from .operator import ArithExpression
        return ArithExpression(self._s, None, func, 0)

    # def _unary_op(self, func):
    #     s = self._s
    #     data_column = s._data_columns[0]
    #     select_list = [f"{func}({data_column}) as {data_column}"]
    #     index_columns = s._index_columns
    #     select_list = itertools.chain(index_columns, select_list)
    #     script = sql_select(select_list, s._var_name, is_exec=True)
    #     return get_orca_obj_from_script(s._session, script, s._index_map, name=s.name, squeeze=1)
