import abc
import itertools
from typing import Iterable

from .internal import _InternalAccessor
from .operator import DataFrameLike, SeriesLike, GroupByOpsMixin, StatOpsMixin, ArithExpression
from .series import Series
from .utils import sql_select, _infer_level


class GroupBy(_InternalAccessor, GroupByOpsMixin, metaclass=abc.ABCMeta):

    def __init__(self, session, internal, index, by, level, as_index, sort, ascending, where_expr, name,
                 groupkeys=None, groupby_list=None, orderby_list=None, result_index_map=None,
                 contextby_result_index_map=None):
        self._session = session
        self._internal = internal
        self._index = index
        self._as_index = as_index
        self._sort = sort
        self._ascending = ascending
        self._where_expr = where_expr
        self._name = name
        
        if (groupkeys is not None and groupby_list is not None
                and orderby_list is not None and result_index_map is not None
                and contextby_result_index_map is not None):
            self._groupkeys = groupkeys
            self._groupby_list = groupby_list
            self._orderby_list = orderby_list
            self._result_index_map = result_index_map
            self._contextby_result_index_map = contextby_result_index_map
            return

        index_names = []
        groupkeys = []
        if by is None and level is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        if level is not None:
            groupkeys, _, index_names, __ = _infer_level(level, self._index_map)
        else:
            for column in by:
                if isinstance(column, str):
                    groupkeys.append(column)
                    index_names.append(column)
                elif isinstance(column, Series):
                    if column._var_name != self._var_name:
                        raise NotImplementedError("Unable to groupby with a external Series")
                    groupkeys.append(column._data_columns[0])
                    index_names.append(column._data_columns[0])
                elif isinstance(column, ArithExpression):
                    if not column.is_series_like:
                        raise ValueError("Grouper is not 1-dimensional")
                    groupkeys.append(column._get_data_select_list()[0])
                    index_names.append(column._data_columns[0])
                else:
                    raise ValueError("Each element in by must be a label")
        self._groupkeys = groupkeys

        self._groupby_list, self._orderby_list, \
        self._result_index_map, self._contextby_result_index_map = \
            GroupByOpsMixin._get_groupby_list_orderby_list_and_index_map(
                groupkeys, index_names, sort, resample=False)

    @property
    @abc.abstractmethod
    def is_series_like(self):
        pass

    @property
    @abc.abstractmethod
    def is_dataframe_like(self):
        pass

    def __getitem__(self, key):
        if isinstance(key, str):
            klass = SeriesGroupBy
            name = key
        elif isinstance(key, Iterable):
            klass = DataFrameGroupBy
            name = self._name
        else:
            raise KeyError(key)
        new_odf = self._internal[key]
        return klass(self._session, new_odf, self._index, None, None,
                     self._as_index, self._sort, self._ascending, self._where_expr, name,
                     self._groupkeys, self._groupby_list, self._orderby_list, self._result_index_map, self._contextby_result_index_map)

    def _groupby_op(self, *args, **kwargs):
        return GroupByOpsMixin._groupby_op(self, *args, **kwargs)
    
    def _contextby_op(self, *args, **kwargs):
        return GroupByOpsMixin._contextby_op(self, *args, **kwargs)
    
    def resample(self, rule, how=None, axis=0, fill_method=None, closed=None,
                 label=None, convention='start', kind=None, loffset=None,
                 limit=None, base=0, on=None, level=None, lazy=False, **kwargs):
        from .resample import SeriesResampler, DataFrameResampler
        klass = SeriesResampler if self.is_series_like else DataFrameResampler
        StatOpsMixin._validate_resample_arguments(how=how, axis=axis, fill_method=fill_method, closed=closed,
                                                  label=label, convention=convention, kind=kind, loffset=loffset,
                                                  limit=limit, base=base, on=on, level=level)
        return klass(self._session, self._internal, self._index, rule, on=on, level=level,
                     where_expr=self._where_expr, name=self._name, groupkeys=self._groupkeys, sort=self._sort)

    def filter(self, func, dropna=True, *args, **kwargs):
        if not dropna:
            raise NotImplementedError()
        if not isinstance(func, str):
            raise ValueError("Orca does not support callable func; func must be a string representing a HAVING condition")
        select_list = self._get_data_select_list()
        script = sql_select(select_list, self._var_name, self._where_expr,
                            groupby_list=self._groupby_list, is_groupby=False,
                            having_list=[func])
        # print(script)    # TODO: debug info
        return self._run_groupby_script("", script, [])

    # def size(self):
    #     groupkeys = self._groupkeys
    #     select_list = ["count(*)"]
    #     script = sql_select(select_list, self._var_name, self._where_expr,
    #                         groupby_list=groupkeys, orderby_list=groupkeys)
    #     return self._run_groupby_script(script, groupkeys, groupby_size=True)


class DataFrameGroupBy(DataFrameLike, GroupBy):

    pass


class SeriesGroupBy(SeriesLike, GroupBy):

    pass


class HavingGroupBy(_InternalAccessor, GroupByOpsMixin, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def is_series_like(self):
        pass

    @property
    @abc.abstractmethod
    def is_dataframe_like(self):
        pass


class DataFrameHavingGroupBy(DataFrameLike, HavingGroupBy):

    pass


class SeriesHavingGroupBy(DataFrameLike, HavingGroupBy):

    pass
