from typing import Iterable

import pandas as pd

from .base import _Frame
from .common import default_session
from .datetimes import Timestamp
from .indexes import Index, IndexOpsMixin
from .indexing import _iLocIndexer, _LocIndexer
from .internal import _ConstantSP, _InternalFrame
from .operator import BooleanExpression, IOOpsMixin, SeriesLike
from .utils import (ORCA_COLUMN_NAME_FORMAT, _infer_axis,
                    _to_numpy_dtype, dolphindb_literal_types,
                    dolphindb_numeric_types, dolphindb_temporal_types,
                    is_datetimelike, is_dolphindb_identifier, sql_select,
                    to_dolphindb_literal)


class Series(SeriesLike, _Frame, IndexOpsMixin, IOOpsMixin):

    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False, session=default_session()):
        from .frame import DataFrame
        if isinstance(data, _ConstantSP):
            assert isinstance(index, _ConstantSP)
            odf = _InternalFrame.create_any_vector(data, index)
            self._internal = odf
            self._where_expr = None
            self._name = None
            self._session = session
        elif isinstance(data, _InternalFrame):
            assert index is None
            assert dtype is None
            assert not copy
            assert not fastpath
            _Frame.__init__(self, data, session)
        elif isinstance(data, Series):
            assert index is None
            assert dtype is None
            name = data.name
            odf = data._internal
            new_odf = odf.copy_as_in_memory_table() if copy else odf
            _Frame.__init__(self, new_odf, session)
            self._where_expr = data._where_expr
        elif isinstance(data, Index):
            raise NotImplementedError()
        elif isinstance(data, pd.Index):
            raise NotImplementedError()
        else:
            if isinstance(data, pd.Series):    # TODO: index necessary ?
                s = (data if index is None and dtype is None and name is None
                     else pd.Series(data, index=index, dtype=dtype, fastpath=fastpath))    # TODO: copy, fastpath?
            else:
                pd_index = index.to_pandas() if isinstance(index, Index) else index
                s = pd.Series(data=data, index=pd_index, dtype=dtype, name=name,
                              copy=copy, fastpath=fastpath)    # TODO: write your own Series parsing function
            name = s.name
            df = DataFrame(s)    # TODO: directly create _InternalFrame?
            _Frame.__init__(self, df._internal, session)
        self._name = name
        self._update_metadata(index)

    def _update_metadata(self, index):
        self._index = Index._from_internal(self._internal, index)

    def __getitem__(self, key):
        if isinstance(key, BooleanExpression):
            return self._get_rows_from_boolean_expression(key)
        elif isinstance(key, tuple):
            if not all(isinstance(k, BooleanExpression) for k in key):
                raise TypeError("All elements in the key tuple must be a boolean expression")
            return self._get_rows_from_boolean_expression(key)
        else:
            try:
                return self.iloc[key]
            except TypeError:
                return self.loc[key]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            return self.iloc.__setitem__(key, value)
        else:
            return self.loc.__setitem__(key, value)

    @property
    def iloc(self):
        return _iLocIndexer(self)

    @property
    def loc(self):
        return _LocIndexer(self)

    @property
    def _ddb_dtype(self):
        return self._ddb_dtypes[self._data_columns[0]]

    @property
    def dtypes(self):
        return self.dtype

    @property    # TODO: @lazyproperty
    def dtype(self):
        return _to_numpy_dtype(self._ddb_dtype)

    @property
    def ndim(self):
        """
        Return an int representing the number of axes / array dimensions.
        
        Returns
        -------
        int
            Return 1 if Series. Otherwise return 2 if DataFrame.
        """
        return 1

    @property
    def name(self):
        """
        Return name of the Series.
        """
        return self._name

    @name.setter
    def name(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError("Series.name must be a string")
        self._name = value

    def rename(self, index=None, inplace=False):
        if index == self._name:
            if inplace:
                return
            else:
                return self.copy()
        if index is None or not is_dolphindb_identifier(index):
            column_name = ORCA_COLUMN_NAME_FORMAT(0)
        else:
            column_name = index
        if not isinstance(column_name, str):
            raise NotImplementedError()
        data_column = self._data_columns[0]
        columns = {data_column: column_name}
        if not inplace:    # TODO: copy?
            odf = self._internal.copy_as_in_memory_table()
            odf.rename(columns=columns, level=None)
            return self._with_where_expr(self._where_expr, odf, name=index)
        else:
            self._internal.rename(columns=columns, level=None)
            self._name = index

    # def _to_script(self):
    #     odf = self._internal
    #     if isinstance(odf, _ConstantSP):
    #         return self._var_name
    #     elif self._segmented:
    #         select_list = self._index_columns
    #         return sql_select(select_list, self._var_name, is_exec=True)
    #     else:
    #         assert len(self._index_columns) == 1
    #         var_name, column_name = self._var_name, self._index_columns[0]
    #         return f"{var_name}.{column_name}"

    def to_pandas(self):
        odf = self._internal
        if odf.is_any_vector:
            session = self._session
            s = session.run(odf.var_name)
            idx = session.run(odf._index.var_name)
            return pd.Series(data=s, index=idx, name=self._name)
        data = _Frame.to_pandas(self)
        s = data[self._data_columns[0]]
        s.rename(self._name, inplace=True)
        return s

    def to_dataframe(self):
        from .frame import DataFrame
        return DataFrame._with_where_expr(self._where_expr, self._internal)

    def unique(self):
        odf = self._internal
        if odf.is_any_vector:
            raise NotImplementedError("mixed-typed Series does not support unique")
        else:
            column_name = self._data_columns[0]
            select_list = [column_name]
            where_expr = [f"isDuplicated({column_name})=false"]
            script = sql_select(select_list, self._var_name, where_expr, is_exec=True)
        return self._session.run(script)

    def nunique(self, dropna=False):
        odf = self._internal
        if dropna:
            raise NotImplementedError()
        if odf.is_any_vector:
            raise NotImplementedError("mixed-typed Series does not support nunique")
        else:
            column_name = self._data_columns[0]
            select_list = [f"nunique({column_name})"]
            script = sql_select(select_list, self._var_name, is_exec=True)
        return self._session.run(script)

    def value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
        if normalize:
            raise NotImplementedError()
        if bins is not None:
            raise NotImplementedError()
        if dropna:
            res = self[self.notna()].groupby(self, sort=ascending).size()
        else:
            res = self.groupby(self, sort=False).size()
        if sort:
            return res.sort_values(ascending=ascending)
        else:
            return res

    def to_csv(self, engine="dolphindb", append=False, *args, **kwargs):
        names = [
            "path_or_buf",
            "sep",
            "na_rep",
            "float_format",
            "columns",
            "header",
            "index",
            "index_label",
            "mode",
            "encoding",
            "compression",
            "quoting",
            "quotechar",
            "line_terminator",
            "chunksize",
            "date_format",
            "doublequote",
            "escapechar",
            "decimal",
        ]

        old_names = [
            "path_or_buf",
            "index",
            "sep",
            "na_rep",
            "float_format",
            "header",
            "index_label",
            "mode",
            "encoding",
            "compression",
            "date_format",
            "decimal",
        ]

        if "path" in kwargs:
            kwargs["path_or_buf"] = kwargs.pop("path")

        if len(args) > 1:
            # Either "index" (old signature) or "sep" (new signature) is being
            # passed as second argument (while the first is the same)
            maybe_sep = args[1]
            if not (isinstance(maybe_sep,str) and len(maybe_sep) == 1):
                names = old_names

        pos_args = dict(zip(names[: len(args)], args))

        for key in pos_args:
            if key in kwargs:
                raise ValueError(
                    "Argument given by name ('{}') and position "
                    "({})".format(key, names.index(key))
                )
            kwargs[key] = pos_args[key]

        if kwargs.get("header", None) is None:
            kwargs["header"] = False  # Backwards compatibility.

        if engine == "pandas":
            return self.to_pandas().to_frame().to_csv(*args, **kwargs)
        elif engine == "dolphindb":
            from orca import DataFrame
            return self.to_dataframe().to_csv(engine=engine, append=append, *args, **kwargs)
        else:
            raise ValueError("Unsupport type engine "+engine)

        return

    @property
    def shape(self):
        return (len(self),)

    @property
    def T(self):
        return self

    @property
    def nbytes(self):
        session = self._session
        script = sql_select(["bytes"],"objs()",where_expr=f"name = '{self._var_name}'" ,is_exec=True)
        script += "[0]"
        return session.run(script)

    def __iter__(self):
        column = f"{self._var_name}.{self._data_columns[0]}"
        for i in range(len(self)):
            if is_datetimelike(self._ddb_dtype):
                yield Timestamp(self._session.run(column+f"[{i}]"))
            else:
                yield self._session.run(column+f"[{i}]")

    def __index_iter__(self):
        column = f"{self._var_name}.{self._index_columns[0]}"
        for i in range(len(self)):
            if is_datetimelike(self._ddb_dtype):
                yield Timestamp(self._session.run(column+f"[{i}]"))
            else:
                yield self._session.run(column+f"[{i}]")

    def items(self):
        return zip(iter(self.__index_iter__()), iter(self))

    def describe(self,  percentiles=None, include=None, exclude=None):
        s = self
        session = self._session
        data_list = []
        index_list = []
        if s._ddb_dtype in dolphindb_numeric_types:
            index_list = ['count', 'mean', 'std', 'min']
            data_list = [s.count(), s.mean(), s.std(), s.min()]
            if percentiles is None:
                percentiles = [.25, .5, .75]
            else:
                if .5 not in percentiles:
                    percentiles.append(.5)
                percentiles.sort()
            #ref = _ConstantSP.upload_obj(session, percentiles)
            for i in percentiles:
                index_list.append(str(i*100)[:2] + '%')
            #data_list.extend(session.run(f"quantile({self._var_name}.{self._data_columns[0]}, {to_dolphindb_literal(percentiles)})"))
            data_list.extend(s.quantile(percentiles))
            index_list.append("max")
            data_list.append(s.max())
        elif s._ddb_dtype in dolphindb_literal_types:
            index_list = ['count', 'unique', 'top', 'freq']
            data_list = [s.count(), len(s.unique())]
            x = s.value_counts()
            data_list.append(session.run(f"{x._var_name}.{x._index_columns[0]}[0]"))
            data_list.append(x.iloc[0])
        elif s._ddb_dtype in dolphindb_temporal_types:
            index_list = ['count', 'unique', 'top', 'freq', 'first', 'last']
            data_list = [s.count(), len(s.unique())]
            x = s.value_counts()
            data_list.append(session.run(f"{x._var_name}.{x._index_columns[0]}[0]"))
            data_list.append(x.iloc[0])
            data_list.append(s.first())
            data_list.append(s.last())

        return pd.Series(data=data_list, index=index_list)

    def set_axis(self, labels, axis=0, inplace=None):
        axis = _infer_axis(self, axis)
        new_index = dict()
        if axis == 0:
            if len(self.index.values) != len(labels):
                raise ValueError(f" Length mismatch: Expected axis has {len(self.index.values)} elements, new values have {len(labels)} elements")
            for idx, new_idx in zip(self.index.values, labels):
                new_index[idx] = new_idx
        elif axis not in [0, 1]:
            raise ValueError('axis must be either 0 or 1')
        if not inplace:
            odf = self._internal.copy_as_in_memory_table()
            return self._with_where_expr(self._where_expr, odf, name=new_index)
        else:
            self._name = new_index

    def rename_axis(self, mapper=None, index=None, axis=0, copy=True, inplace=False):
        axis = _infer_axis(self, axis)
        if axis == 0:
            if mapper is not None:
                index = mapper
        elif axis not in [0, 1]:
            raise ValueError('axis must be either 0 or 1')
        if index == str.upper:
            index = str.upper(self._internal.index_name)
        elif index == str.lower:
            index = str.lower(self._internal.index_name)
        if index is not None:    # TODO: support more args; implement rename index
            if isinstance(index, dict):
                if not is_dolphindb_identifier(index):
                    column_name = ORCA_COLUMN_NAME_FORMAT(0)
                else:
                    column_name = index
                if not isinstance(column_name, str):
                    raise NotImplementedError()
                data_column = self._data_columns[0]
                columns = {data_column: column_name}
                if not inplace:  # TODO: copy?
                    odf = self._internal.copy_as_in_memory_table()
                    odf.rename(columns=columns, level=None)
                    return self._with_where_expr(self._where_expr, odf, name=index)
                else:
                    self._internal.rename(columns=columns, level=None)
                    self._name = index
            else:
                if isinstance(index, str) or isinstance(index, int) or isinstance(index, float) or isinstance(index, bool) or index is None:
                    index = [index]
                if len(index) != 1:
                    raise ValueError(f"Length of new names must be 1, got {len(index)}")
                if not inplace:
                    os = self._internal.copy_as_in_memory_table()
                    new_os = self._with_where_expr(self._where_expr, os, session=self._session)
                    new_os.rename_axis(index[0], inplace=True)
                    return new_os
                else:
                    index_map = [i for i in self._index_map]
                    index_map.remove(index_map[0])
                    tup = (self._index_map[0][0], (index[0],))
                    index_map.insert(0, tup)
                    new_os = _InternalFrame(self._session, self._var, index_map=index_map)
                    self._internal = new_os
                    self._index = Index._from_internal(new_os)
