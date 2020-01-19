import itertools
from typing import Callable

import dolphindb as ddb
import numpy as np

from .internal import _ConstantSP
from .operator import ArithExpression, BooleanExpression
from .merge import _generate_joiner
from .utils import (_get_where_list, _infer_dtype, check_key_existence,
                    is_dolphindb_integral, is_dolphindb_scalar,
                    is_dolphindb_vector, sql_select,
                    to_dolphindb_literal, _try_convert_iterable_to_list)


def _unfold(key):
    from .series import Series
    if (isinstance(key, tuple) and len(key) == 2
            and (isinstance(key[0], (slice, tuple, list, BooleanExpression, Series))
                 or is_dolphindb_scalar(key[0]))):
        rows_sel, cols_sel = key
    else:
        rows_sel = key
        cols_sel = None
    return rows_sel, cols_sel


def _check_is_integral(ddb_dtype):
    if ddb_dtype not in (ddb.settings.DT_BYTE,
                         ddb.settings.DT_SHORT,
                         ddb.settings.DT_INT,
                         ddb.settings.DT_LONG):
        raise ValueError(
            "Location based indexing can only have [integer, integer "
            "slice (START point is INCLUDED, END point is EXCLUDED), "
            "listlike of integers, boolean array] types")


class _LocIndexer(object):

    def __init__(self, df_or_s):
        from .series import Series
        from .frame import DataFrame
        assert isinstance(df_or_s, (Series, DataFrame)), \
            f"unexpected argument type: {type(df_or_s)}"
        self._df = df_or_s

    def _get_rows_cond_from_tuple(self, key):
        assert isinstance(key, tuple)

        odf = self._df._internal
        if len(key) > len(odf.index_map):
            raise ValueError("Too many indexers")
        var_name = odf.var_name
        rows_cond = " and ".join(
            f"{var_name}.{index[0]} == {to_dolphindb_literal(k)}"
            for index, k in zip(odf.index_map, key)
        )
        return rows_cond

    def _get_data_columns(self, cols_sel):
        df = self._df
        old_data_columns = df._data_columns

        if df._is_series_like and cols_sel is not None:
            raise IndexError("Too many indexers")

        if cols_sel is None:
            data_columns = None
        elif isinstance(cols_sel, slice):
            if cols_sel == slice(None):
                data_columns = None
            elif cols_sel.step is not None and cols_sel.step != 1:
                raise NotImplementedError("Cannot use step with DolphinDB.")
            else:
                if cols_sel.start is None:
                    col_start = 0
                else:
                    col_start = old_data_columns.index(cols_sel.start)
                if cols_sel.stop is None:
                    col_end = len(old_data_columns)
                else:
                    col_end = old_data_columns.index(cols_sel.stop) + 1
                data_columns = old_data_columns[col_start:col_end]
        elif isinstance(cols_sel, str):
            data_columns, _ = check_key_existence(cols_sel, old_data_columns)
        elif is_dolphindb_vector(cols_sel):
            if not all(isinstance(col, str) for col in cols_sel):
                raise TypeError("Every value in column index must be a string")
            data_columns, _ = check_key_existence(cols_sel, old_data_columns)
        else:
            raise KeyError(cols_sel)
        return data_columns

    def _any_vector_getitem(self, rows_sel):
        from .series import Series

        df = self._df
        session = df._session
        index_column = df._index._var_name

        if is_dolphindb_scalar(rows_sel):
            rows_cond = session.run(f"find({index_column},{rows_sel})")
            value = session.run(f"{df._var_name}[{rows_cond}]")
        elif isinstance(rows_sel, slice):
            if rows_sel.step is not None and rows_sel.step != 1:
                raise KeyError("slice with step != 1 is not supported")
            len_data = len(df)
            if rows_sel.start is None:
                start = 0
            else:
                start = session.run(f"find({index_column},{to_dolphindb_literal(rows_sel.start)})")
            if rows_sel.stop is None:
                stop = len_data
            else:
                stop = session.run(f"find({index_column},{to_dolphindb_literal(rows_sel.stop)})")

            stop = min(stop, len_data)
            start = min(start, stop)
            # rows_cond = f"{start}:{stop}"    # TODO: start = stop = 0
            rows_cond = f"{start}..{stop}"
            data_col = session.run(f"{df._var_name}[{rows_cond}]")
            index_col = session.run(f"{index_column}[{rows_cond}]")
            value = Series(data=list(data_col), index=list(index_col), session=session)
        elif is_dolphindb_vector(rows_sel):
            rows_cond = session.run(f"find({index_column},{to_dolphindb_literal(rows_sel)})")
            data_col = session.run(f"{df._var_name}[{rows_cond}]")
            index_col = session.run(f"{index_column}[{rows_cond}]")
            value = Series(data=list(data_col), index=list(index_col), session=session)
        else:
            raise NotImplementedError()

        return value

    def __getitem__(self, key):  # TODO: handle Series._internal is _ConstantSP
        from .series import Series
        from .frame import DataFrame

        df = self._df
        index_columns = df._index_columns
        session = df._session
        rows_sel, cols_sel = _unfold(key)

        if df._internal.is_any_vector:
            return self._any_vector_getitem(rows_sel)

        data_columns = self._get_data_columns(cols_sel)
        if isinstance(rows_sel, slice) and rows_sel == slice(None):
            return df[data_columns]
        elif isinstance(rows_sel, (list, tuple)) and len(rows_sel) == 0:
            return df[data_columns].iloc[0:0]

        if isinstance(rows_sel, BooleanExpression):
            if rows_sel._var_name == df._var_name:
                return df[data_columns][rows_sel]
        elif isinstance(rows_sel, Series):
            if rows_sel._ddb_dtype == ddb.settings.DT_BOOL:
                return df[data_columns][rows_sel]

        ref = df.compute(data_columns=data_columns, as_non_segmented=True)
        var_name = ref._var_name

        rows_cond = None
        squeeze = False
        # TODO: dealing with boolean indices
        if isinstance(rows_sel, Callable):
            raise NotImplementedError("orca does not support function-like keys")
        elif isinstance(rows_sel, BooleanExpression):
            rows_cond = _get_where_list(rows_sel)[0]
            # raise NotImplementedError()
        elif isinstance(rows_sel, Series):
            if rows_sel.dtype != np.dtype(bool):
                raise ValueError("...")
            else:
                rowslen = len(rows_sel)
                reflen = len(ref)
                var_name = rows_sel._var_name
                data_column = rows_sel._data_columns[0]
                data = f"{var_name}.{data_column}"
                if rowslen < reflen:
                    padding = rowslen - reflen
                    rows_cond = f"join({data}, take(false, {padding}))"
                elif rowslen > reflen:
                    rows_cond = f"{data}[0:{reflen}]"
                else:
                    rows_cond = data
                # TODO: if Series is not in memory
                # rows_cond = rows_sel.
        elif isinstance(rows_sel, DataFrame):
            raise ValueError("Cannot index with multidimensional key")
        elif isinstance(rows_sel, slice):
            index_column = index_columns[0]
            if rows_sel.step is not None and rows_sel.step != 1:
                raise NotImplementedError("Cannot use step with DolphinDB.")
            elif rows_sel == slice(None):
                pass
            elif rows_sel.start is None:
                stop_rows = session.run(f"find({var_name}.{index_column},{to_dolphindb_literal(rows_sel.stop)})")
                rows_cond = f"rowNo({var_name}.{index_column}) <= {stop_rows}"
                # rows_cond = f"{var_name}.{index_column} <= " \
                #           f"{to_dolphindb_literal(rows_sel.stop)}"
            elif rows_sel.stop is None:
                start_rows = session.run(f"find({var_name}.{index_column},{to_dolphindb_literal(rows_sel.start)})")
                rows_cond = f"rowNo({var_name}.{index_column}) >= {start_rows}"

                # rows_cond = f"{var_name}.{index_column} >= " \
                #           f"{to_dolphindb_literal(rows_sel.start)}"
            else:
                start_rows = session.run(f"find({var_name}.{index_column},{to_dolphindb_literal(rows_sel.start)})")
                stop_rows = session.run(f"find({var_name}.{index_column},{to_dolphindb_literal(rows_sel.stop)})")
                rows_cond = f"between(rowNo({var_name}.{index_column}), " \
                            f"({start_rows}:{stop_rows}))"
                # rows_cond = f"between({var_name}.{index_column}, " \
                #            f"({to_dolphindb_literal(rows_sel.start)}:" \
                #            f"{to_dolphindb_literal(rows_sel.stop)}))"
        elif isinstance(rows_sel, tuple):
            squeeze = True
            rows_cond = self._get_rows_cond_from_tuple(rows_sel)
        elif is_dolphindb_scalar(rows_sel):
            squeeze = True
            rows_cond = self._get_rows_cond_from_tuple((rows_sel,))
        elif is_dolphindb_vector(rows_sel):
            var = _ConstantSP.upload_obj(session, rows_sel)
            if var.type == ddb.settings.DT_BOOL:
                len_var, len_ref = len(var), len(ref)
                if len_var != len_ref:
                    raise IndexError("indices are out-of-bounds")
                # elif len_var < len_ref:
                #    padding = len_ref - len_var
                #   rows_cond = f"join({var.var_name},take(false,{padding}))"
                else:
                    rows_cond = var.var_name
            else:
                rows_cond = f"find({var_name}.{index_columns[0]}, {var.var_name})"
        else:
            raise KeyError(rows_sel)

        if rows_cond is None:
            script = var_name
        elif is_dolphindb_vector(rows_sel) and var.type != ddb.settings.DT_BOOL:
            select_list = itertools.chain([f"{var.var_name} as {index_columns[0]}"], df._data_columns)
            from_clause = f"{var_name}[{rows_cond}]"
            script = sql_select(select_list, from_clause)
        else:
            script = f"{var_name}[{rows_cond}]"
            # script = var_name if rows_cond is None else f"select {var.var_name}, {rows_cond} from "

        return ref._get_from_script(session, script, ref, squeeze=squeeze, squeeze_axis=None)

    def __setitem__(self, key, value):
        from .series import Series
        from .frame import DataFrame
        from .groupby import ContextByExpression

        df = self._df
        session = df._session
        index_columns = df._index_columns

        df._prepare_for_update()
        var = df._var

        rows_sel, cols_sel = _unfold(key)
        cols_cond = self._get_data_columns(cols_sel)
        cols_cond = df._data_columns if cols_cond is None else cols_cond

        if (isinstance(rows_sel, BooleanExpression)
                and isinstance(value, (Series, DataFrame, ArithExpression,
                                       BooleanExpression, ContextByExpression))
                and rows_sel._var_name == value._var_name
                and (rows_sel is value._where_expr or value._where_expr is None)):
            column_names = _try_convert_iterable_to_list(cols_cond)
            new_values = value._get_data_select_list()
            if isinstance(value, ContextByExpression):
                contextby_list = value._get_contextby_list()
            else:
                contextby_list = None
            return var._sql_update(column_names, new_values, where_expr=rows_sel,
                                    contextby_list=contextby_list)

        from_clause = None
        lencols = len(cols_cond)
        value_list = []
        if isinstance(value, Series):
            res = self[rows_sel, cols_sel]
            rows = session.run(f"{res._var_name}.{res._index_columns[0]}")
            lenrows = len(rows)
            ref = value.compute(as_non_segmented=True)
            df_var_name, ref_var_name = df._var_name, ref._var_name
            if (cols_sel is None) or isinstance(cols_sel, slice):
                find = session.run(
                    f"find({ref._var_name}.{ref._index_columns[0]}, {res._var_name}.{res._index_columns[0]})")
                for i in range(0, lenrows):
                    if find[i] == -1:
                        value_list.append('NULL')
                    else:
                        value_list.append(value[rows[i]])
            elif is_dolphindb_vector(cols_sel):
                _, from_clause = _generate_joiner(df_var_name, ref_var_name, df._index_columns,
                                                  ref._index_columns)
                for i in range(0, lencols):
                    value_list.append(f"{ref_var_name}.{ref._data_columns[0]}")
            elif isinstance(cols_sel, str):
                _, from_clause = _generate_joiner(df_var_name, ref_var_name, df._index_columns,
                                                  ref._index_columns)
                value_list.append(f"{ref_var_name}.{ref._data_columns[0]}")
        elif isinstance(value, DataFrame):
            ref = value.compute(as_non_segmented=True)
            df_var_name, ref_var_name = df._var_name, ref._var_name
            _, from_clause = _generate_joiner(df_var_name, ref_var_name, df._index_columns,
                                              ref._index_columns)
            for i in range(0, lencols):
                value_list.append(f"{ref_var_name}.{cols_cond[i]}")
        elif isinstance(value, ContextByExpression):
            if (value._var_name == df._var_name
                    and value._where_expr is df._where_expr):
                value_list = value._value_list
            else:
                self[key] = value.compute()
        elif isinstance(value, dict):
            if (cols_sel is None) or isinstance(cols_sel, slice):
                for i in range(0, lencols):
                    if value.get(cols_cond[i], None) is None:
                        value_list.append('NULL')
                    else:
                        value_list.append(value[cols_cond[i]])
            elif is_dolphindb_vector(cols_sel):
                s = Series(value)
                ref = s.compute(as_non_segmented=True)
                df_var_name, ref_var_name = df._var_name, ref._var_name
                _, from_clause = _generate_joiner(df_var_name, ref_var_name, df._index_columns,
                                                  ref._index_columns)
                for i in range(0, lencols):
                    value_list.append(f"{ref_var_name}.{ref._data_columns[0]}")

            elif isinstance(cols_sel, str):
                s = Series(value)
                ref = s.compute(as_non_segmented=True)
                df_var_name, ref_var_name = df._var_name, ref._var_name
                _, from_clause = _generate_joiner(df_var_name, ref_var_name, df._index_columns,
                                                  ref._index_columns)
                value_list.append(f"{ref_var_name}.{ref._data_columns[0]}")
        elif isinstance(value, set):
            if len(value) == lencols:
                value_list = list(value)
            else:
                ref = _ConstantSP.upload_obj(session, list(value))
                value_list.append(f"{ref.var_name}")
        elif is_dolphindb_scalar(value):
            for i in range(0, lencols):
                value_list.append(to_dolphindb_literal(value))
        elif is_dolphindb_vector(value):
            if len(value) == lencols:
                if isinstance(value[0], set):
                    valuel = []
                    for i in range(0, lencols):
                        valuel.append(list(value[i]))
                    ref = _ConstantSP.upload_obj(session, valuel)
                    for i in range(0, lencols):
                        value_list.append(f"{ref.var_name}[{i}]")
                else:
                    value_list = value
            else:
                t = np.array(value)
                if t.ndim == 1:
                    ref = _ConstantSP.upload_obj(session, value)
                    value_list.append(f"{ref.var_name}")
                elif t.ndim == 2:
                    ref = _ConstantSP.upload_obj(session, t)
                    for i in range(0, lencols):
                        value_list.append(f"{ref.var_name}[{i}]")
        elif isinstance(value, np.ndarray):
            if value.ndim == 0:
                value_list = [value.item()]
        else:
            raise NotImplementedError()

        index_column = df._index_columns[0]
        var_name = var.var_name
        if isinstance(rows_sel, Callable):
            raise NotImplementedError("orca does not support function-like keys")
        elif isinstance(rows_sel, BooleanExpression):
            rows_cond = rows_sel
        elif isinstance(rows_sel, slice):
            index_column = index_columns[0]
            if rows_sel.step is not None and rows_sel.step != 1:
                raise NotImplementedError("Cannot use step with DolphinDB.")
            elif rows_sel == slice(None):
                rows_cond = None
            elif rows_sel.start is None:
                stop_rows = session.run(f"find({var_name}.{index_column},{to_dolphindb_literal(rows_sel.stop)})")
                rows_cond = f"rowNo({var_name}.{index_column}) <= {stop_rows}"
            elif rows_sel.stop is None:
                start_rows = session.run(f"find({var_name}.{index_column},{to_dolphindb_literal(rows_sel.start)})")
                rows_cond = f"rowNo({var_name}.{index_column}) >= {start_rows}"
            else:
                start_rows = session.run(f"find({var_name}.{index_column},{to_dolphindb_literal(rows_sel.start)})")
                stop_rows = session.run(f"find({var_name}.{index_column},{to_dolphindb_literal(rows_sel.stop)})")
                rows_cond = f"between(rowNo({var_name}.{index_column}), " \
                            f"({start_rows}:{stop_rows}))"
        elif is_dolphindb_scalar(rows_sel):
            rows_cond = f"{index_column} = {to_dolphindb_literal(rows_sel)}"
        elif is_dolphindb_vector(rows_sel):
            upload_var = _ConstantSP.upload_obj(session, rows_sel)
            if upload_var.type == ddb.settings.DT_BOOL:
                len_var, len_df = len(upload_var), len(self._df)
                if len_var != len_df:
                    raise IndexError("indices are out-of-bounds")
                else:
                    rows_cond = f"{upload_var.var_name} = true"
            else:
                rows_cond = f"{index_column} in {upload_var.var_name}"
        if from_clause is None:
            var._sql_update(cols_cond, value_list, where_expr=rows_cond)
        else:
            var._sql_update(cols_cond, value_list, where_expr=rows_cond, from_table_joiner=from_clause)


class _iLocIndexer(object):

    def __init__(self, data):
        self._df = data

    def _get_data_columns(self, cols_sel):
        df = self._df
        old_data_columns = df._data_columns

        if df._is_series_like and cols_sel is not None:
            raise IndexError("Too many indexers")

        if cols_sel is None:
            data_columns = None
        elif isinstance(cols_sel, slice):
            if cols_sel == slice(None):
                data_columns = None
            else:
                data_columns = old_data_columns[cols_sel]
        elif is_dolphindb_integral(cols_sel):
            data_columns = [old_data_columns[cols_sel]]
        elif is_dolphindb_vector(cols_sel):
            if _infer_dtype(value=cols_sel) != "bool":
                data_columns = [old_data_columns[col] for col in cols_sel]
            else:
                l = len(cols_sel)
                if l != len(old_data_columns):
                    raise IndexError("indices are out-of-bounds")
                data_columns = [col for col, col_sel in zip(old_data_columns, cols_sel) if col_sel]
        else:
            raise KeyError(cols_sel)
        return data_columns

    @staticmethod
    def _check_negative_index(idx, length):
        if idx < 0:
            idx += length
            if idx < 0:
                raise IndexError("single positional indexer is out-of-bounds")
        return idx

    def _get_rows_cond(self, rows_sel, ref):
        """
        Deduce rows selection condition from row index in the key.

        Returns
        -------
        Tuple[str, _ConstantSP]
            The condition script and the temporary variable storing the
            uploaded key.
        """
        var = None
        if isinstance(rows_sel, Callable):
            raise NotImplementedError("orca does not support function-like keys")
        elif isinstance(rows_sel, slice):
            if rows_sel.step is not None and rows_sel.step != 1:
                raise KeyError("slice with step != 1 is not supported")
            len_data = len(ref)
            if rows_sel.start is None:
                start = 0
            elif rows_sel.start < 0:
                start = max(len_data + rows_sel.start, 0)
            else:
                start = rows_sel.start

            if rows_sel.stop is None:
                stop = len_data
            elif rows_sel.stop < 0:
                stop = max(len_data + rows_sel.stop, 0)
            else:
                stop = rows_sel.stop
            stop = min(stop, len_data)
            start = min(start, stop)
            rows_cond = f"{start}:{stop}"    # TODO: start = stop = 0
        elif is_dolphindb_integral(rows_sel):
            rows_sel = self._check_negative_index(rows_sel, len(self._df))
            rows_cond = f"[{rows_sel}]"
        elif is_dolphindb_vector(rows_sel):
            len_self = len(self._df)
            rows_sel = np.array([self._check_negative_index(idx, len_self) for idx in rows_sel])
            var = _ConstantSP.upload_obj(ref._session, rows_sel)
            if var.type not in (ddb.settings.DT_BYTE,
                                ddb.settings.DT_SHORT,
                                ddb.settings.DT_INT,
                                ddb.settings.DT_LONG,
                                ddb.settings.DT_BOOL):
                raise ValueError(
                    "Location based indexing can only have [integer, integer "
                    "slice (START point is INCLUDED, END point is EXCLUDED), "
                    "listlike of integers, boolean array] types")
            rows_cond = var.var_name
        else:
            raise TypeError("Cannot index by location index with a non-integer key")
        return rows_cond, var

    def _any_vector_getitem(self, rows_sel):
        from .series import Series

        df = self._df
        session = df._session
        index_column = df._index._var_name

        if is_dolphindb_integral(rows_sel):
            value = session.run(f"{df._var_name}[{rows_sel}]")
        elif isinstance(rows_sel, slice):
            if rows_sel.step is not None and rows_sel.step != 1:
                raise KeyError("slice with step != 1 is not supported")
            len_data = len(df)
            if rows_sel.start is None:
                start = 0
            elif rows_sel.start < 0:
                start = max(len_data + rows_sel.start, 0)
            else:
                start = rows_sel.start
            if rows_sel.stop is None:
                stop = len_data
            elif rows_sel.stop < 0:
                stop = max(len_data + rows_sel.stop, 0)
            else:
                stop = rows_sel.stop
            stop = min(stop, len_data)
            start = min(start, stop)
            if not isinstance(start, int) or not isinstance(stop, int):
                raise TypeError("cannot do slice indexing with these indexers [{start}:{stop}]")
            rows_cond = f"{start}:{stop}"    # TODO: start = stop = 0
            # rows_cond = f"{start}..{stop - 1}"
            data_col = session.run(f"{df._var_name}[{rows_cond}]")
            index_col = session.run(f"{index_column}[{rows_cond}]")
            value = Series(data=list(data_col), index=list(index_col), session=session)
        elif is_dolphindb_vector(rows_sel):
            data_col = session.run(f"{df._var_name}[{rows_sel}]")
            index_col = session.run(f"{index_column}[{rows_sel}]")
            value = Series(data=list(data_col), index=list(index_col), session=session)
        else:
            raise NotImplementedError()

        return value

    def __getitem__(self, key):
        df = self._df
        session = df._session
        rows_sel, cols_sel = _unfold(key)

        if df._internal.is_any_vector:
            return self._any_vector_getitem(rows_sel)

        data_columns = self._get_data_columns(cols_sel)
        if isinstance(rows_sel, slice) and rows_sel == slice(None):
            return df[data_columns]
        elif isinstance(rows_sel, (list, tuple)) and len(rows_sel) == 0:
            return df[data_columns].iloc[0:0]

        if df._segmented:
            raise ValueError("A segmented table does not support direct access with iloc")

        ref = df.compute(data_columns=data_columns,
                         as_non_segmented=True)  # TODO: is this necessary if I use a sql_select later?
        rows_cond, _ = self._get_rows_cond(rows_sel, ref)
        squeeze = is_dolphindb_integral(rows_sel)

        script = f"{ref._var_name}[{rows_cond}]"
        name = ref.name if ref._is_series_like else None
        return ref._get_from_script(session, script, ref, name=name, squeeze=squeeze, squeeze_axis=0)

    def __setitem__(self, key, value):
        from .series import Series
        from .frame import DataFrame
        df = self._df
        if df._segmented:
            raise ValueError("A segmented table does not support direct access.")

        df._prepare_for_update()

        odf = df._internal.var
        rows_sel, cols_sel = _unfold(key)
        data_columns = self._get_data_columns(cols_sel)
        data_columns = data_columns or df._data_columns

        session = df._session

        from_clause = None
        lencols = len(data_columns)

        value_list = []
        if isinstance(value, Series):
            res = self[rows_sel, cols_sel]
            rows = session.run(f"{res._var_name}.{res._index_columns[0]}")

            lenrows = len(rows)

            ref = value.compute(as_non_segmented=True)
            df_var_name, ref_var_name = df._var_name, ref._var_name
            if (cols_sel is None) or isinstance(cols_sel, slice):
                find = session.run(
                    f"find({ref._var_name}.{ref._index_columns[0]}, {res._var_name}.{res._index_columns[0]})")
                for i in range(0, lenrows):
                    if find[i] == -1:
                        value_list.append('NULL')
                    else:
                        value_list.append(value.loc[rows[i]])
            elif is_dolphindb_vector(cols_sel):
                _, from_clause = _generate_joiner(df_var_name, ref_var_name, df._index_columns,
                                                  ref._index_columns)
                for i in range(0, lencols):
                    value_list.append(f"{ref_var_name}.{ref._data_columns[0]}")

            elif is_dolphindb_scalar(cols_sel):
                _, from_clause = _generate_joiner(df_var_name, ref_var_name, df._index_columns,
                                                  ref._index_columns)
                value_list.append(f"{ref_var_name}.{ref._data_columns[0]}")

        elif isinstance(value, DataFrame):
            ref = value.compute(as_non_segmented=True)
            df_var_name, ref_var_name = df._var_name, ref._var_name
            _, from_clause = _generate_joiner(df_var_name, ref_var_name, df._index_columns,
                                              ref._index_columns)
            for i in range(0, lencols):
                value_list.append(f"{ref_var_name}.{data_columns[i]}")


        elif is_dolphindb_scalar(value):
            # create an ANY vector
            col_num = len(data_columns)
            value_tuple = np.array([value] * col_num, dtype=np.object)
            var = _ConstantSP.upload_obj(session, value_tuple)
            value_script = [var.var_name]
            for i in range(0, lencols):
                value_list.append(to_dolphindb_literal(value))
        elif is_dolphindb_vector(value):
            if len(value) == lencols:
                if isinstance(value[0], set):
                    valuel = []
                    for i in range(0, lencols):
                        valuel.append(list(value[i]))
                    ref = _ConstantSP.upload_obj(session, valuel)
                    for i in range(0, lencols):
                        value_list.append(f"{ref.var_name}[{i}]")
                else:
                    value_list = value
            # ref = _ConstantSP.upload_obj(session, value)
            else:
                t = np.array(value)
                if t.ndim == 1:
                    ref = _ConstantSP.upload_obj(session, value)
                    value_list.append(f"{ref.var_name}")
                elif t.ndim == 2:
                    ref = _ConstantSP.upload_obj(session, t)
                    for i in range(0, lencols):
                        value_list.append(f"{ref.var_name}[{i}]")
        elif isinstance(value, dict):

            if (cols_sel is None) or isinstance(cols_sel, slice):
                for i in range(0, lencols):
                    if value.get(data_columns[i], None) is None:
                        value_list.append('NULL')
                    else:
                        value_list.append(value[data_columns[i]])
        elif isinstance(value, set):
            if len(value) == lencols:
                value_list = list(value)
            else:
                ref = _ConstantSP.upload_obj(session, list(value))
                value_list.append(f"{ref.var_name}")
        elif isinstance(value, np.ndarray):
            if value.ndim == 0:
                value_list = [value.item()]
        else:
            raise NotImplementedError()

        index_column = df._index_columns[0]
        var_name = odf.var_name
        if isinstance(rows_sel, Callable):
            raise NotImplementedError("orca does not support function-like keys")
        elif isinstance(rows_sel, BooleanExpression):
            rows_cond = rows_sel
        elif isinstance(rows_sel, slice):
            index_column = df._index_columns[0]
            if rows_sel.step is not None and rows_sel.step != 1:
                raise NotImplementedError("Cannot use step with DolphinDB.")
            elif rows_sel == slice(None):
                rows_cond = None
                pass
            elif rows_sel.start is None:
                stop_rows = rows_sel.stop - 1
                rows_cond = f"rowNo({var_name}.{index_column}) <= {stop_rows}"

            elif rows_sel.stop is None:
                start_rows = rows_sel.start
                rows_cond = f"rowNo({var_name}.{index_column}) >= {start_rows}"

            else:
                start_rows = rows_sel.start
                stop_rows = rows_sel.stop - 1
                rows_cond = f"between(rowNo({var_name}.{index_column}), " \
                            f"({start_rows}:{stop_rows}))"
        elif is_dolphindb_scalar(rows_sel):
            rows_cond = f"rowNo({var_name}.{index_column}) = {to_dolphindb_literal(rows_sel)}"
            # rows_cond = f"{index_column} = {to_dolphindb_literal(rows_sel)}"
        elif is_dolphindb_vector(rows_sel):
            var = _ConstantSP.upload_obj(session, rows_sel)
            if var.type == ddb.settings.DT_BOOL:
                len_var, len_df = len(var), len(self._df)
                if len_var != len_df:
                    raise IndexError("indices are out-of-bounds")
                # elif len_var < len_df:
                #    padding = len_df - len_var
                #    rows_cond = f"join({var.var_name},take(false,{padding})) = true"
                else:
                    rows_cond = f"{var.var_name} = true"
            else:
                rows_cond = f"{index_column} in {var.var_name}"

        if from_clause is None:
            odf._sql_update(data_columns, value_list, where_expr=rows_cond)
        else:
            odf._sql_update(data_columns, value_list, where_expr=rows_cond, from_table_joiner=from_clause)
