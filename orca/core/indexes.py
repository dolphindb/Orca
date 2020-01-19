import abc
import itertools
from typing import Iterable

import dolphindb as ddb
import numpy as np
import pandas as pd

from .common import default_session
from .datetimes import DatetimeProperties
from .internal import _ConstantSP, _InternalFrame, _InternalAccessor
from .operator import IndexLike, SeriesLike, ArithOpsMixin, StatOpsMixin, LogicalOpsMixin, IOOpsMixin
from .utils import (
    _to_freq, dolphindb_temporal_types, _to_numpy_dtype,
    is_dolphindb_uploadable, sql_select,
    get_orca_obj_from_script)


class IndexOpsMixin(ArithOpsMixin, LogicalOpsMixin, metaclass=abc.ABCMeta):

    def __init__(self, internal, session):
        self._internal = internal
        self._session = session
        if isinstance(internal, _ConstantSP):
            self._name = None
        else:
            names = [name[0] if name is not None else None
                    for _, name in internal.index_map]
            if len(names) == 0:
                self._name = None
                self._names = None
            elif len(names) == 1:
                self._name = names[0]
                self._names = None
            elif len(names) > 1:
                self._name = None
                self._names = names
        # self._dtype = internal.dtype

    def __len__(self):
        return len(self._internal)

    # @property TODO: name getter and setter
    # def name(self):
    #     return self.name

    # @name.setter
    # def name(self, name):
    #     self.rename(inplace=True)

    @property
    def _ddb_dtype(self):
        if isinstance(self._internal, _ConstantSP):
            return self._type
        else:
            index_column = self._index_column
            return self._ddb_dtypes[index_column]

    @property
    def ndim(self):
        return 1

    @property
    def size(self):
        return len(self)

    @property
    def dtype(self):
        return _to_numpy_dtype(self._ddb_dtype)

    @abc.abstractmethod
    def to_numpy(self):
        pass

    @abc.abstractmethod
    def to_pandas(self):
        pass

    @property
    def is_monotonic(self):
        return self._unary_agg_op("isSorted", axis=None, level=None, numeric_only=False)

    @property
    def is_monotonic_increasing(self):
        return self._unary_agg_op("isSorted", axis=None, level=None, numeric_only=False)

    @property
    def is_monotonic_decreasing(self):
        return self._unary_agg_op("isSorted{,false}", axis=None, level=None, numeric_only=False)

    @property
    def is_unique(self):
        len_self = len(self)
        return len_self == 1 or self.nunique == len_self

    @property
    def hasnans(self):
        return self._unary_agg_op("hasNull", axis=None, level=None, numeric_only=False)

    def _unary_op(self, *args, **kwargs):
        return ArithOpsMixin._unary_op(self, *args, **kwargs)

    def _binary_op(self, *args, **kwargs):
        return ArithOpsMixin._binary_op(self, *args, **kwargs)

    def _extended_binary_op(self, *args, **kwargs):
        return ArithOpsMixin._extended_binary_op(self, *args, **kwargs)

    def _logical_op(self, *args, **kwargs):
        return LogicalOpsMixin._logical_op(self, *args, **kwargs)

    def _logical_unary_op(self, *args, **kwargs):
        return LogicalOpsMixin._logical_unary_op(self, *args, **kwargs)

    def _to_script(self):
        odf = self._internal
        if isinstance(odf, _ConstantSP):
            return self._var_name
        select_list = self._index_columns
        return sql_select(select_list, self._var_name)
        # elif self._segmented:
        #     select_list = self._index_columns
        #     return sql_select(select_list, self._var_name, is_exec=True)
        # else:
        #     assert len(self._index_columns) == 1
        #     var_name, column_name = self._var_name, self._index_column
        #     return f"{var_name}.{column_name}"

    def _binary_op_on_different_indices(self, other, func, axis):
        """
        Implementation of binary operator between Series on different
        indices. A new Series representing an in-memory DolphinDB table
        is returned. It is garenteed that both Series have no where_expr.
        
        Parameters
        ----------
        other : _Frame
            Right hand side of the operator.
        func : str
            Fuction name.
        
        Returns
        -------
        orca.DataFrame
            The result of the operation.
        
        Raises
        ------
        NotImplementedError
            To be implemented.
        """
        from .merge import _generate_joiner
        _COLUMN_NAME = "ORCA_DIFFERENT_INDICES_COLUMN"

        if other._is_series_like:
            session = self._session
            self_var_name, other_var_name = self._var_name, other._var_name
            self_column_name = self._data_columns[0]
            other_column_name = other._data_columns[0]
            select_list = [f"{func}({self_var_name}.{self_column_name}, {other_var_name}.{other_column_name}) as {_COLUMN_NAME}"]
            index_list, from_clause = _generate_joiner(
                self_var_name, other_var_name, self._index_columns, other._index_columns)
            select_list = itertools.chain(index_list, select_list)
            script = sql_select(select_list, from_clause)
            index_map = [(s_map[0], None if s_map[1] != o_map[1] else s_map[1])
                         for s_map, o_map
                         in zip(self._internal.index_map, other._internal.index_map)]
            return self._get_from_script(
                session, script, data_columns=[_COLUMN_NAME], index_map=index_map)
        elif other._is_dataframe_like:
            raise NotImplementedError()

class Index(IndexLike, _InternalAccessor, IndexOpsMixin, IOOpsMixin):

    """
    Accessor for DataFrame and Series.
    
    When calling get_select_list, a specific identifier is added before the
    column.

    When names are not given, a specific identifier is used instead.
    """

    def __init__(self, data, dtype=None, copy=False, name=None, tupleize_cols=None, session=default_session()):
        if isinstance(data, _ConstantSP):
            assert dtype is None
            assert not copy
            assert tupleize_cols is None
            IndexOpsMixin.__init__(self, data, session)
            self._name = name
        elif isinstance(data, _InternalFrame):
            assert dtype is None
            assert name is None
            assert not copy
            assert tupleize_cols is None
            IndexOpsMixin.__init__(self, data, session)
        else:
            if isinstance(data, (pd.Index, pd.Series)):
                idx = (data if dtype is None and name is None and tupleize_cols is None
                       else pd.Index(data, dtype=dtype, name=name, tupleize_cols=tupleize_cols))
            else:
                idx = pd.Index(data=data, dtype=dtype, copy=False, name=name,
                               tupleize_cols=tupleize_cols)    # TODO: copy = True or False ?, freq?
            # var = _ConstantSP.upload_obj(session, idx.to_numpy())
            # var._framize(name=idx.name)
            # IndexOpsMixin.__init__(self, var, session)
            # self._name = idx.name
            odf = _InternalFrame.from_pandas(session, idx)
            IndexOpsMixin.__init__(self, odf, session)
        self._where_expr = None

    def __repr__(self):
        if self._segmented:
            return "<.index.Index object representing a column in a DolphinDB segmented table>"
        else:
            return self.to_pandas().__repr__()

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return (self._var_name == other._var_name
                    and self._index_columns == other._index_columns)
    
    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def _from_internal(cls, odf, index=None):
        """
        Create an orca Index indicated by an _InternalFrame and another
        pandas or orca Index.
        
        Parameters
        ----------
        odf : _InternalFrame
            odf provides the metadata of the represented DolphinDB table
            and servers as the _internal attribute of the Index
        index : pd.Index or orca.Index, optional
            index provides the metadata such as name, frequency, etc. of
            the Index, by default None
        """
        session = odf._session
        if index is None or not isinstance(index, pd.DatetimeIndex):
            if odf.is_any_vector:
                index = Index(index, session=session)
            elif len(odf.index_map) == 1:
                if odf._ddb_dtypes[odf._index_columns[0]] in dolphindb_temporal_types:
                    index = DatetimeIndex._from_internal(odf, index)
                else:
                    index = Index(odf, session=session)
            elif len(odf.index_map) == 0:
                index = Index([], session=session)
            else:
                index = MultiIndex(odf, session=session)
        elif isinstance(index, pd.DatetimeIndex):
            index = DatetimeIndex._from_internal(odf, index)
        else:
            raise TypeError("Unsupported index type")
        return index

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError("Index.name must be a string")
        self._name = value

    @property
    def names(self):
        return self._names

    def rename(self, value, inplace=False):
        raise NotImplementedError()

    @property
    def _index_column(self):
        assert isinstance(self._internal, _InternalFrame)
        return self._index_columns[0]
    
    @property
    def _index(self):
        return self

    def _get_data_select_list(self):
        if isinstance(self._internal, _ConstantSP):
            return [self._var_name]
        else:
            return self._index_columns

    def _to_script_list(self):
        if isinstance(self._internal, _ConstantSP):
            assert(self._form != ddb.settings.DF_TABLE)
            return [self._var_name]
        else:
            return [sql_select([col], self._var_name, is_exec=True)
                    for col in self._index_columns]

    def to_pandas(self):
        if isinstance(self._internal, _ConstantSP):
            df = self._session.run(self._to_script())
            return pd.Index(df).rename(self._name)
        elif len(self._index_columns) == 0:
            raise ValueError("Frame has no default index if it is not in memory")
        else:
            df = self._session.run(self._to_script())
            return pd.Index(df.iloc[:,0]).rename(self._name)

    def to_numpy(self):
        return self.to_pandas().to_numpy()

    def _get_data_select_list(self):
        if isinstance(self._internal, _ConstantSP):
            return [self._var_name]
        else:
            return [f"{self._var_name}.{self._index_column}"]

    def _unary_agg_op(self, func, *args, **kwargs):
        if isinstance(self._internal, _ConstantSP):
            script = f"{func}({self._var_name})"
        else:
            index_column = self._index_column
            select_list = [f"{func}({index_column})"]
            script = sql_select(select_list, self._var_name, is_exec=True)
        return get_orca_obj_from_script(self._session, script, [], as_index=True)

    def min(self, axis=None, skipna=True, *args, **kwargs):
        return self._unary_agg_op("min")

    def max(self, axis=None, skipna=True, *args, **kwargs):
        return self._unary_agg_op("max")

    def unique(self, level=None):
        pass

    def nunique(self, dropna=True):
        pass

    isna = LogicalOpsMixin.isna
    notna = LogicalOpsMixin.notna
    isnull = LogicalOpsMixin.isnull
    notnull = LogicalOpsMixin.notnull
    fillna = StatOpsMixin.fillna
    dropna = StatOpsMixin.dropna

    # def _binary_op(self, other, func):
    #     from .frame import DataFrame
    #     from .series import Series
    #     if is_dolphindb_uploadable(self):
    #         raise NotImplementedError()
    #     elif not isinstance(self, Index):
    #         raise TypeError("Operand must be a Series")
    #     elif is_dolphindb_uploadable(other):
    #         raise NotImplementedError()
    #     elif isinstance(other, DataFrame):
    #         raise NotImplementedError()
    #     elif isinstance(other, Series):
    #         raise NotImplementedError()
    #     else:
    #         raise TypeError("Operand must be a Series or DataFrame")

    # def _logical_op(self, other, func):
    #     raise NotImplementedError()

    # def _logical_unary_op(self, func):
    #     raise NotImplementedError()
    @property
    def values(self):
        #warnings.warn("orca objects does not store data in numpy arrays. Accessing values will retrive whole data from the remote node.", Warning)
        return self.to_numpy()

    @property
    def shape(self):
        return (len(self),)

    @property
    def nbytes(self):
        session = self._session
        script = sql_select(["bytes"], "objs()", where_expr=f"name='{self._var_name}'", is_exec=True)
        script += "[0]"
        return session.run(script)

    @property
    def ndim(self):
        return 1

    @property
    def T(self):
        return self

    @property
    def is_all_dates(self):
        return False

class MultiIndex(Index):

    def __init__(self, data, names=None, session=default_session()):
        if isinstance(data, _InternalFrame):
            assert names is None
            Index.__init__(self, data, session=session)
        elif isinstance(data, pd.MultiIndex):
            assert names is None
            frame = data.to_frame()
            var = _ConstantSP.upload_obj(session, frame)
            Index.__init__(self, var, session=session)
            self._names = list(data.names)

    @staticmethod
    def _from_pandas_multiindex(session, index):
        from .frame import DataFrame
        return DataFrame(data=None, session=session, index=index).index

    @classmethod
    def from_arrays(cls, arrays, sortorder=None, names=None, session=default_session()):
        return cls._from_pandas_multiindex(session, pd.MultiIndex.from_arrays(arrays, sortorder, names))
    
    @classmethod
    def from_tuples(cls, tuples, sortorder=None, names=None, session=default_session()):
        return cls._from_pandas_multiindex(session, pd.MultiIndex.from_tuples(tuples, sortorder, names))

    @classmethod
    def from_product(cls, iterables, sortorder=None, names=None, session=default_session()):
        return cls._from_pandas_multiindex(session, pd.MultiIndex.from_product(iterables, sortorder, names))

    @classmethod
    def from_frame(cls, df, sortorder=None, names=None, session=default_session()):    # TODO: directly upload frame
        return cls._from_pandas_multiindex(session, pd.MultiIndex.from_frame(df, sortorder, names))

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        raise NotImplementedError()

    # def _to_script(self):
    #     select_list = self._index_columns
    #     return sql_select(select_list, self._var_name)

    def to_pandas(self):    # TODO: dealing with where clause
        df = self._session.run(self._to_script())
        return pd.MultiIndex.from_frame(df).rename(self.names)

    def _unary_op(self, func):
        raise TypeError(f"cannot perform {func} with thid index type: MultiIndex")

    def _binary_op(self, other, func):
        raise TypeError(f"cannot perform {func} with thid index type: MultiIndex")

    def _logical_op(self, other, func):
        raise TypeError(f"cannot perform {func} with thid index type: MultiIndex")

    def _logical_unary_op(self, func):
        raise TypeError(f"cannot perform {func} with thid index type: MultiIndex")


class RangeIndex(Index):

    def __init__(self, start=None, stop=None, step=1, name=None, session=default_session()):
        self._start = start
        self._stop = stop
        self._step = step
        self._name = name
        self._session = session
        self._internal = None

    @classmethod
    def _from_internal(cls, frame):
        odf = frame._internal
        session = frame._session
        names = odf.index_map[0][1]
        name = names if names is None else names[0]
        obj = cls(start=0, stop=len(odf), name=name, session=session)
        obj._internal = odf
        return obj

    @classmethod
    def _from_internal_frame_and_range(cls, session, internal, index):
        assert isinstance(index, RangeIndex)
        assert isinstance(internal, _InternalFrame)
        start, stop, step, name = index.start, index.stop, index.step, index.name
        range_index = RangeIndex(start, stop, step, name, session)
        range_index._internal = internal
        return range_index

    def __len__(self):
        # TODO: step
        return max(self.stop - self.start, 0)

    def __eq__(self, other):
        if isinstance(other, RangeIndex):
            return (self.start == other.start
                    and self.stop == other.stop
                    and self.step == other.step)
        else:
            return False

    @property
    def start(self):
        return self._start
    
    @property
    def stop(self):
        return self._stop

    @property
    def step(self):
        return self._step

    @property
    def dtype(self):
        return np.dtype(np.int64)

    @property
    def id(self):     # Pseudo id to be used for reference
        return f"_orca_range_index_{self.start}_{self.stop}"

    def _to_script(self):
        return f"({self.start}..{self.stop - 1})"

    def __repr__(self):
        if self.name is None:
            return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step})"
        else:
            return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step}, name='{self.name}')"

    def to_pandas(self):
        return pd.RangeIndex(self.start, self.stop, self.step, name=self.name)

    def _unary_func(self, func):
        script = f"{func}({self.start}..{self.stop - 1})"
        return self._session.run(script)

    def _binary_func(self, other, func):
        session = self._session
        script = f"{func}({self.start}..{self.stop - 1}, {other})"
        data = session.run(script)
        return Index(data, session=session)

    # def _get_select_list(self, sql=False):
    #     if self._internal is None and sql:
    #         return []
    #     elif self._internal is None:
    #         name = self._real_column_names[0]
    #         script = f"{self._to_script()} as {name}"
    #         return [script]
    #     else:
    #         return Index._get_select_list(self)

    # def _get_update_list(self):
    #     if self._internal is None:
    #         name = self._real_column_names[0]
    #         script = f"{name} = {self._to_script()}"
    #         return (script,)
    #     else:
    #         return Index._get_update_list(self)

    # def append(self, other):
    #     session = self._session
    #     if isinstance(other, RangeIndex) and other.start == self.stop:
    #         return RangeIndex(self.start, other.end, session=session)
    #     script = f"join({self._to_script()}, {other._getter_script})"
    #     data = run_script(session, script)
    #     return Index(data, session=session)

    def head(self, n=5):
        if n == 0:
            return RangeIndex(0, 0, self.name)
        elif n > 0:
            if len(self) <= n:
                return RangeIndex(self.start, self.stop, name=self.name)
            else:
                return RangeIndex(self.start, self.start + n, name=self.name)
        else:
            if len(self) <= abs(n):
                return RangeIndex(self.start, self.start, name=self.name)
            else:
                return RangeIndex(self.start, self.stop + n, name=self.name)

    def tail(self, n=5):
        if n == 0:
            return RangeIndex(0, 0, self.name)
        elif n > 0:
            if len(self) <= n:
                return RangeIndex(self.start, self.stop, name=self.name)
            else:
                return RangeIndex(self.stop - n, self.stop, name=self.name)
        else:
            if len(self) <= abs(n):
                return RangeIndex(self.stop, self.stop, name=self.name)
            else:
                return RangeIndex(self.start - n, self.stop, name=self.name)


class DatetimeIndex(DatetimeProperties, Index):

    def __init__(self, data, freq=None, dtype=None, tz=None, session=default_session()):
        if isinstance(data, _InternalFrame):
            assert freq is None
            assert dtype is None
            assert tz is None

            index_columns = data._index_columns
            ddb_dtype = data._ddb_dtypes[index_columns[0]]
            assert len(index_columns) == 1
            assert ddb_dtype in dolphindb_temporal_types
            Index.__init__(self, data, session=session)
            self._freq = _to_freq(ddb_dtype)
            self._dtype = _to_numpy_dtype(ddb_dtype)
            self._tz = None
        elif isinstance(data, pd.DatetimeIndex):
            data = (data if freq is None and dtype is None and tz is None
                    else pd.DatetimeIndex(data, freq=freq, dtype=dtype, tz=tz))
            Index.__init__(self, data, session=session)
            self._freq = data.freq
            self._dtype = data.dtype
            self._tz = data.tz
        else:
            raise NotImplementedError()

    # def __eq__(self, other):
    #     if isinstance(other, DatetimeIndex):
    #         raise NotImplementedError()
    #     else:
    #         return False

    @classmethod
    def _from_internal(cls, odf, index):
        obj = cls(odf, session=odf._session)
        if index is None:
            # sample_script = sql_select(odf._index_columns, odf._var_name, is_exec=True, limit=3)
            # sample_data = odf._session.run(sample_script)
            # try:
            #     obj._freq = pd.infer_freq(pd.DatetimeIndex(sample_data))
            # except ValueError:
            #     obj._freq = None
            obj._freq = None
            obj._dtype = None
            obj._tz = None
        else:
            try:
                # FIXME:
                # oidx = orca.to_datetime(['20130101 09:00:00','20130101 09:00:02','20130101 09:00:03','20130101 09:00:05','20130101 09:00:06'])
                # odf = orca.DataFrame({'A': ["a", "c", "w", "f", "f"], 'B': [0, 1, 2, np.nan, 4]}, index=orca.Index(data=oidx,name='time'))
                obj._freq = index.freq
            except ValueError:
                obj._freq = None
            obj._dtype = index.dtype
            obj._tz = index.tz
        return obj

    @property
    def freq(self):
        return self._freq

    @property
    def tz(self):
        return self._tz

    @property
    def is_all_dates(self):
        return True

    # @property
    # def dtype(self):
    #     return self._dtype     # TODO: sophisticated datetime

    def to_pandas(self):
        odf = self._internal
        if isinstance(odf, _ConstantSP):
            data = self._session.run(self._to_script())
        else:
            pdf = self._session.run(self._to_script())
            data = pdf[pdf.columns[0]]
        return pd.DatetimeIndex(data=data, freq=self._freq, tz=self._tz).set_names(self._name)
        # return pd.DatetimeIndex(data=pdf, freq=self._freq, dtype=self._dtype)

    def _logical_unary_op(self, func):
        from .operator import BooleanExpression
        return BooleanExpression(self, None, func, 1)

    def _unary_op(self, func, infered_ddb_dtypestr):
        from .operator import ArithExpression
        return ArithExpression(self, None, func, 0,
                               infered_ddb_dtypestr=infered_ddb_dtypestr)

class PeriodIndex(Index):

    def __init__(self):
        pass

