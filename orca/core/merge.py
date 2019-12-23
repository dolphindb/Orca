import itertools

from .base import _Frame
from .common import _raise_must_compute_error
from .operator import ArithExpression, BooleanExpression
from .utils import (_merge_where_expr, _try_convert_iterable_to_list,
                    check_key_existence, get_orca_obj_from_script,
                    is_dolphindb_integral, sql_select, to_dolphindb_literal)
from .window import WindowJoiner


def _generate_joiner(left_var_name, right_var_name,
                     left_join_columns, right_join_columns, 
                     how="outer", sort=False, use_right_index=False):
        """
        Returns the index_list and from_clause used in binary operations on
        Frames with different indices.
        """
        if len(left_join_columns) != len(right_join_columns):
            raise ValueError("cannot join with no overlapping index names")
        if sort:
            methods = {"left_semi": "slsj", "left": "slj", "right": "slj",
                       "outer": "sfj", "inner": "sej"}
        else:
            methods = {"left_semi": "lsj", "left": "lj", "right": "lj",
                       "outer": "fj", "inner": "ej", "asof": "aj"}
        method = methods.get(how, None)
        if method is None:
            raise ValueError(f"do not recognize join method {how}")

        left_join_literal = to_dolphindb_literal(left_join_columns)
        right_join_literal = to_dolphindb_literal(right_join_columns)
        if how == "right":
            index_list = (f"{right_var_name}.{col} as {col}" for col in right_join_columns)
        elif how in ("left", "left_semi"):
            index_list = (f"{left_var_name}.{col} as {col}" for col in left_join_columns)
        else:
            index_list = (
                f"iif(isValid({left_var_name}.{left_col}), "
                f"{left_var_name}.{left_col}, "
                f"{right_var_name}.{right_col}) "
                f"as {right_col if use_right_index else left_col}"
                for left_col, right_col
                in zip(left_join_columns, right_join_columns)
            )
        if how == "right":
            from_clause = f"{method}({right_var_name}, {left_var_name}, " \
                          f"{right_join_literal}, {left_join_literal})"
        else:
            from_clause = f"{method}({left_var_name}, {right_var_name}, " \
                          f"{left_join_literal}, {right_join_literal})"
        return index_list, from_clause


def _get_left_right_data_join_columns(left, right, on, left_on, right_on,
                                      left_index, right_index,
                                      by=None, left_by=None, right_by=None):
    """
    Return (left_data_columns, left_join_columns, right_data_columns, right_join_columns)
    """
    left_data_columns = left._data_columns
    right_data_columns = right._data_columns
    if by is not None:
        if left_by is not None or right_by is not None:
            raise ValueError('Can only pass argument "by" OR "left_by" and "right_by", not a combination of both.')
        left_by = right_by = _try_convert_iterable_to_list(by)
    else:
        left_by = _try_convert_iterable_to_list(left_by)
        right_by = _try_convert_iterable_to_list(right_by)
    if on is not None:
        if left_on is not None or right_on is not None:
            # TODO: pandas.errors.MergeError
            raise ValueError('Can only pass argument "on" OR "left_on" and "right_on", not a combination of both.')
        left_on = right_on = _try_convert_iterable_to_list(on)
    else:
        left_on = _try_convert_iterable_to_list(left_on)
        right_on = _try_convert_iterable_to_list(right_on)
    left_on = left_by + left_on
    right_on = right_by + right_on
    if left_index:
        left_join_columns = left._index_columns
    else:
        left_join_columns, left_data_columns = check_key_existence(left_on, left_data_columns)
    if right_index:
        right_join_columns = right._index_columns
    else:
        right_join_columns, right_data_columns = check_key_existence(right_on, right_data_columns)
    if len(left_join_columns) != len(right_join_columns):
        raise ValueError("len(right_on) must equal len(left_on)")
    return left_data_columns, left_join_columns, right_data_columns, right_join_columns


def merge(left, right, how='inner', on=None, left_on=None, right_on=None,
          left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'),
          copy=True, indicator=False, validate=None,
          by=None, left_by=None, right_by=None):
    # TODO: return internal expression
    if isinstance(left, (ArithExpression, BooleanExpression)):
        _raise_must_compute_error("Can only merge Series or DataFrame objects")
    if not isinstance(left, _Frame) or not left.is_dataframe_like:
        raise TypeError(f"Can only merge Series or DataFrame objects, a {type(left)} was passed")
    if sort:    # TDOO: sort
        if how not in ('inner'):
            raise NotImplementedError()
    if not isinstance(right, _Frame):
        raise TypeError('other must be a DataFrame or a Series')
    elif right.is_series_like and right.name is None:
        raise TypeError('Other Series must have a name')
    if how != 'asof':
        if by is not None:
            raise TypeError("merge() got an unexpected keyword argument 'by'")
        if left_by is not None:
            raise TypeError("merge() got an unexpected keyword argument 'left_by'")
        if right_by is not None:
            raise TypeError("merge() got an unexpected keyword argument 'right_by'")
    lsuf, rsuf = suffixes
    if not isinstance(lsuf, str) or not isinstance(rsuf, str):
        raise TypeError("Both lsuffix and rsuffix must be strings")
    # left_ref = left.compute()
    # right_ref = right.compute()
    left_ref, right_ref = left, right
    left_var_name, right_var_name = left._var_name, right._var_name
    left_data_columns, left_join_columns, right_data_columns, right_join_columns = \
        _get_left_right_data_join_columns(
            left_ref, right_ref, on, left_on, right_on,
            left_index, right_index, by, left_by, right_by)

    index_list, from_clause = _generate_joiner(
        left_ref._var_name, right_ref._var_name,
        left_join_columns, right_join_columns, how=how, sort=sort)
    overlap_columns = set(left_data_columns) & set(right_data_columns)
    if overlap_columns and not lsuf and not rsuf:
        raise ValueError(f"columns overlap but no suffix specified: "
                         f"{list(overlap_columns)}")

    def get_join_list(tb_name, suf, data_columns, join_columns=None):
        def overlap_checked_col(col):
            return f"{tb_name}.{col} as " + (col + suf if col in overlap_columns else col)
        if join_columns is None:
            return (overlap_checked_col(col) for col in data_columns)
        else:
            join_list = []
            for col in data_columns:
                try:
                    idx = join_columns.index(col)
                    column_alias_script = index_list[idx]
                    column_alias_script = column_alias_script[:column_alias_script.index(" as ")+4] + col
                    join_list.append(column_alias_script)
                except ValueError:
                    join_list.append(overlap_checked_col(col))
            return join_list

    index_list = list(index_list)
    if left_index and not right_index:
        select_list = itertools.chain(
            (f"{right_var_name}.{col} as {col}" for col in right_ref._index_columns),
            get_join_list(left_var_name, lsuf, left_data_columns),
            get_join_list(right_var_name, rsuf, right_ref._data_columns, right_join_columns)
        )
        index_map = right_ref._index_map
    elif right_index and not left_index:
        select_list = itertools.chain(
            (f"{left_var_name}.{col} as {col}" for col in left_ref._index_columns),
            get_join_list(left_var_name, lsuf, left_ref._data_columns, left_join_columns),
            get_join_list(right_var_name, rsuf, right_data_columns)
        )
        index_map = left_ref._index_map
    elif not left_index and not right_index:
        select_list = itertools.chain(
            get_join_list(left_var_name, lsuf, left_ref._data_columns),
            get_join_list(right_var_name, rsuf, right_data_columns)
        )
        index_map = None
    else:
        select_list = itertools.chain(
            index_list,
            get_join_list(left_var_name, lsuf, left_data_columns),
            get_join_list(right_var_name, rsuf, right_data_columns)
        )
        index_map = left_ref._index_map

    where_list = _merge_where_expr(left._where_expr, right._where_expr)

    script = sql_select(select_list, from_clause, where_list)
    return get_orca_obj_from_script(left._session, script, index_map=index_map)


def merge_asof(left, right, on=None, left_on=None, right_on=None,
               left_index=False, right_index=False,
               by=None, left_by=None, right_by=None,
               suffixes=('_x', '_y'), tolerance=None,
               allow_exact_matches=True, direction='backward'):
    if tolerance is not None:
        raise NotImplementedError()
    if not allow_exact_matches:
        raise NotImplementedError()
    if direction != "backward":
        raise NotImplementedError()
    return merge(left, right, how="asof", on=on, left_on=left_on, right_on=right_on,
                 left_index=left_index, right_index=right_index,
                 by=by, left_by=left_by, right_by=right_by, suffixes=suffixes)


def merge_window(left, right, window_lower, window_upper, prevailing=False, on=None,
                 left_on=None, right_on=None, left_index=False, right_index=False):
    if not isinstance(right, _Frame):
        raise TypeError('other must be a DataFrame or a Series')
    elif right.is_series_like and right.name is None:
        raise TypeError('Other Series must have a name')
    if not is_dolphindb_integral(window_lower) or not is_dolphindb_integral(window_upper):
        raise TypeError("Both window_lower and window_upper must be integers")
    left_ref = left.compute()
    right_ref = right.compute()
    _, left_join_columns, __, right_join_columns = \
        _get_left_right_data_join_columns(
            left_ref, right_ref, on, left_on, right_on, left_index, right_index)

    window = f"{window_lower}:{window_upper}"
    method = "pwj" if prevailing else "wj"
    return WindowJoiner(left._session, method, window, left_ref._internal,
                        right_ref._internal, left_join_columns, right_join_columns)
