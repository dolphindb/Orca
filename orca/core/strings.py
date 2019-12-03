import itertools

from .series import Series
from .utils import sql_select, get_orca_obj_from_script


def _orca_unary_op(func):
    def ufunc(self):
        return self._unary_op(func)
    return ufunc


class StringMethods(object):

    def __init__(self, s):
        self._s = s

    lower = _orca_unary_op("lower")

    def _unary_op(self, func):
        s = self._s
        data_column = s._data_columns[0]
        select_list = [f"{func}({data_column}) as {data_column}"]
        index_columns = s._index_columns
        select_list = itertools.chain(index_columns, select_list)
        script = sql_select(select_list, s._var_name, is_exec=True)
        return get_orca_obj_from_script(s._session, script, s._index_map, name=s.name, squeeze=1)
