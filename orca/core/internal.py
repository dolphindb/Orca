import itertools
import warnings
from typing import Iterable, List, Optional, Tuple, Union

import dolphindb as ddb
import pandas as pd
from pandas.api.types import is_list_like

from .common import (AttachDefaultIndexWarning, _get_verbose,
                     _warn_not_dolphindb_identifier)
from .utils import (ORCA_COLUMN_NAME_FORMAT, ORCA_INDEX_NAME_FORMAT,
                    _new_orca_identifier, _to_column_index, _to_index_map,
                    check_key_existence, is_dolphindb_identifier, sql_select,
                    sql_update, to_dolphindb_literal)

IndexMap = Tuple[str, Optional[Tuple[str, ...]]]

class _ConstantSP(object):
    """
    The internal object which represents DolphinDB objects and works like
    a smart pointer. When its reference count is reduced to zero (which
    is automatically controlled by Python's memory management system),
    the represented object's memory is released.

    Scripts which modify DolphinDB variables (i.e. assignments and updates)
    should be encapsulated in methods of this class and called from the
    _InternalFrame which owns the instance of this class.

    .. note:: this is an internal class. It is not supposed to be exposed to
        users and users should not directly access to it.
    """
    ORCA_IDENTIFIER = "ORCA_"

    def __init__(self, session, id):
        self._id = id
        self._session = session
        self._form, self._type, self._segmented, self._in_memory = None, None, False, False
        self._update_metadata()

    def __del__(self):
        var_name = self.var_name
        script = f"{var_name} = NULL; undef('{var_name}', VAR)"
        self._session.run(script)

    def __len__(self):
        if not self.segmented:
            script = f"size({self.var_name})"
        else:
            script = f"exec count(*) from {self.var_name}"
        return self._session.run(script)

    @property
    def form(self):
        return self._form

    @property
    def type(self):
        return self._type

    @property
    def schema(self):
        return self._schema

    @property
    def in_memory(self):
        return self._in_memory

    @property
    def segmented(self):
        return self._segmented

    @property
    def id(self):
        return self._id

    @property
    def var_name(self):    # TODO: lazyproperty?
        return _ConstantSP.ORCA_IDENTIFIER + self._id

    @property
    def _var_name(self):    # TODO: lazyproperty?
        return _ConstantSP.ORCA_IDENTIFIER + self._id

    @classmethod
    def upload_obj(cls, session, obj):
        obj_id = _new_orca_identifier()
        var_name = _ConstantSP.ORCA_IDENTIFIER + obj_id
        session.upload({var_name: obj})
        return cls(session, obj_id)

    @classmethod
    def run_script(cls, session, script):
        obj_id = _new_orca_identifier()
        var_name = _ConstantSP.ORCA_IDENTIFIER + obj_id
        if _get_verbose():
            print(script)
        session.run(f"&{var_name} = ({script})")
        return cls(session, obj_id)

    def _to_script(self):
        return self.var_name

    def _update_metadata(self):
        var_name, session = self.var_name, self._session
        form, dtype, typestr = session.run(f"[form({var_name}), type({var_name}), typestr({var_name})]")
        form, dtype, typestr = int(form), int(dtype), str(typestr)
        self._form, self._type = form, dtype
        self._segmented = typestr.find("SEGMENTED") >= 0
        self._in_memory = typestr.find("IN-MEMORY") >= 0 or not self._segmented
        self._schema = (session.run(f"schema({self.var_name}).colDefs").set_index("name")
                        if form == ddb.settings.DF_TABLE else None)

    def _sql_update(self, column_names: List[str],
                    new_values: List[str],
                    from_table_joiner: Optional[str] = None,
                    where_expr=None,
                    contextby_list: Optional[List[str]] = None):
        session = self._session
        table_name = self.var_name
        script = sql_update(table_name, column_names, new_values,
                            from_table_joiner, where_expr, contextby_list)
        if _get_verbose():
            print(script)
        session.run(script)
        self._update_metadata()

    def squeeze(self, index_columns=[], data_columns=None, name=None, as_index=False, squeeze_axis=None):
        """
        Reduce the dimension of a DataFrame or Series if possible.
        
        Parameters
        ----------
        index_columns : list[str], optional
            The index columns of the input DataFrame or Series, used as
            name of the result
        data_columns : list[str], optional
            The data columns of the input DataFrame or Series. Only
            these columns will be used, by default None, that is, to use
            all columns
        name : str, optional
            The name of the returned Series if squeezed to a Series,
            by default None
        as_index : bool, optional
            Whether to return a Index instead of a Series, by default
            False
        squeeze_axis : 0, 1 or None
            A specific axis to squeeze. 0 for index, 1 for column, None
            for both. By default None
        """
        from .indexes import Index
        from .series import Series

        session = self._session
        var_name, form = self.var_name, self.form

        if form == ddb.settings.DF_SCALAR:
            return session.run(var_name)
        elif form == ddb.settings.DF_VECTOR:
            return session.run(f"{var_name}[0]")
        elif form == ddb.settings.DF_TABLE:
            all_columns = self.schema.index
            if index_columns:    # TODO: MultiIndex
                index_column = index_columns[0]
                name = session.run(f"(exec {index_column} from {var_name})[0]")
            data_columns = data_columns or [col for col in all_columns if col not in index_columns]
            assert all(col in self.schema.index for col in data_columns)
            script = sql_select(data_columns, var_name)
            if squeeze_axis is None and len(data_columns) == 1:
                return session.run(f"values({script})[0][0]")
            index = _ConstantSP.upload_obj(session, data_columns)
            if len(set(self.schema.typeInt[data_columns])) > 1:    # has mixed type
                self.reset_with_script(f"loop(first, values({script}))")
                if as_index:
                    return Index(self, name=name, session=session)
                else:
                    return Series(self, index=index, name=name, session=session)
            else:
                if as_index:
                    self.reset_with_script(f"each(first, values({script}))")
                    return Index(self, name=name, session=session)
                else:
                    self.reset_with_script(
                        f"table({index.var_name} as {ORCA_INDEX_NAME_FORMAT(0)}, "
                        f"each(first, values({script})) as ORCA_EXPRESSION_COLUMN)")
                    index_map = [(ORCA_INDEX_NAME_FORMAT(0), None)]
                    data_columns = ["ORCA_EXPRESSION_COLUMN"]
                    odf = _InternalFrame(session, self, index_map, data_columns)
                    return Series(odf, name=name, session=session)
        else:
            raise ValueError(f"Unsupported form: {form}")

    def rename(self, columns):
        old_names, new_names = zip(*columns.items())
        old_names_literal = to_dolphindb_literal(old_names)
        new_names_literal = to_dolphindb_literal(new_names)
        session, var_name = self._session, self.var_name
        if _get_verbose():
            print(f"rename!({var_name}, {old_names_literal}, {new_names_literal})")
        session.run(f"rename!({var_name}, {old_names_literal}, {new_names_literal})")
        self._update_metadata()

    def reset_with_script(self, script):
        session, var_name = self._session, self.var_name
        session.run(f"&{var_name} = ({script})")
        self._update_metadata()

    def append(self, script):
        session, var_name = self._session, self.var_name
        if _get_verbose():
            print(f"append!({var_name}, {script})")
        session.run(f"append!({var_name}, {script})")
        self._update_metadata()

    def attach_index(self, index, index_map):
        script_list = index._to_script_list()
        replace_column_scripts = []
        update_column_list = []
        update_column_scripts = []
        for (col, _), script in zip(index_map, script_list):
            if col in self.schema:
                replace_column_scripts.append(f"replaceColumn!({self.var_name}, {to_dolphindb_literal(col)}, {script})")
            else:
                update_column_list.append(col)
                update_column_scripts.append(script)
        self._sql_update(update_column_list, update_column_scripts)
        self._session.run(";".join(replace_column_scripts))
        self._update_metadata()

    def drop_columns(self, column_names):
        session, var_name = self._session, self.var_name
        column_names_literal = to_dolphindb_literal(column_names)
        script = f"{var_name}.drop!({column_names_literal})"
        if _get_verbose():
            print(script)
        session.run(script)
        self._update_metadata()

    def attach_default_index(self):
        if self.segmented:
            warnings.warn("Unable to attach an default index to segmented table.", AttachDefaultIndexWarning)
            return False
        form, var_name = self._form, self.var_name
        size = len(self)
        if size <= 0:
            return False
        if form == ddb.settings.DF_TABLE:
            column_names = [ORCA_INDEX_NAME_FORMAT(0)]
            new_values = [f"0..{size-1}"]
            try:
                self._sql_update(column_names, new_values)
            except RuntimeError as ex:
                ex_msg = str(ex)
                if (ex_msg.startswith("The table is not allowed to update")
                        or ex_msg.startswith("<Server Exception> in run: The table is not allowed to update")
                        or ex_msg.startswith("<Server Exception> in run: The category")
                        or ex_msg.startswith("<Server Exception> in run: The data type")
                        or ex_msg.endswith("the table shouldn't be shared and the size of the new column must equal to the size of the table.")):
                    warnings.warn("Unable to attach an default index to the table.", AttachDefaultIndexWarning)
                    return False
                else:
                    raise
        elif form == ddb.settings.DF_VECTOR:
            script = f"table({var_name}, 0..{size-1} as {ORCA_INDEX_NAME_FORMAT(0)})"
            self.reset_with_script(script)
        return True

    def as_frame(self, name):
        form, var_name = self._form, self.var_name
        assert form == ddb.settings.DF_VECTOR
        if name is None or not is_dolphindb_identifier(name):
            _warn_not_dolphindb_identifier()
            name = ORCA_INDEX_NAME_FORMAT(0)
        script = f"table({var_name} as {name})"
        self.reset_with_script(script)


class _InternalAccessor(object):

    def __init__(self):
        self._internal = None

    @property
    def _var_name(self):
        """
        The variable name of the DolphinDB object represented by this
        DataFrame or Series.
        """
        return self._internal._var_name

    @property
    def _in_memory(self):
        """
        Whether the DolphinDB object represented by the orca object is in memory.

        If in_memory is True, modifications to the object are allowed.
        """
        return self._internal.in_memory

    @property
    def _segmented(self):
        """
        Whether the DolphinDB object represented by the orca object is segmented.

        If segmented is True, direct access to the object is not allowed. A SQL
        query is used instead.
        """
        return self._internal.segmented

    @property
    def _data_columns(self):
        """
        The real data column names in the DolphinDB table.
        """
        return self._internal._data_columns

    @property
    def _index_columns(self):
        """
        The real index column names in the DolphinDB table.
        """
        return self._internal._index_columns

    @property
    def _column_index(self):
        return self._internal._column_index

    @property
    def _column_index_names(self):
        return self._internal._column_index_names

    @property
    def _column_index_level(self):
        return self._internal._column_index_level

    def _column_name_for(self, column_name_or_index):
        return self._internal.column_name_for(column_name_or_index)

    @property
    def _index_map(self):
        return self._internal._index_map

    @property
    def _index_names(self):
        return self._internal._index_names

    @property
    def _index_name(self):
        return self._internal._index_name

    @property
    def _schema(self):
        """
        The schema of the DolphinDB table with the column names as the index.
        """
        return self._internal.schema

    @property
    def _form(self):
        return self._internal.form

    @property
    def _ddb_dtypes(self):
        return self._internal._ddb_dtypes

    @property
    def _ddb_dtypestr(self):
        return self._internal._ddb_dtypestr

    @property
    def _type(self):
        """
        The type of the DolphinDB object.
        """
        return self._internal.type

    @property
    def _var(self):
        return self._internal.var


class _InternalFrame(object):
    """
    The internal DataFrame which represents DolphinDB objects (a DolphinDB
    table or vector) and manage indices.

    .. note:: this is an internal class. It is not supposed to be exposed to
        users and users should not directly access to it.
    """

    def __init__(self, session: ddb.session,
                 var: _ConstantSP,
                 index_map: Optional[List[IndexMap]] = None,
                 data_columns: Optional[List[str]] = None,
                 column_index: Optional[List[Tuple[str, ...]]] = None,
                 column_index_names: Optional[List[str]] = None, index_of_any_vector=None):
                #  index=None, is_any_vector=False):
        from .indexes import Index
        self._session = session
        self._var = var
        if index_of_any_vector:
            self._is_any_vector = True
            self._index = index_of_any_vector
            return
        else:
            self._is_any_vector = False
        if index_map is None or index_map == []:     # TODO: create RangeIndex
            index_map = [(ORCA_INDEX_NAME_FORMAT(0), None)] if var.attach_default_index() else []
        else:
            self.check_index_map_validity(index_map)

        assert data_columns is None or all(isinstance(col, str) for col in data_columns)

        self._index_map = index_map
        if data_columns is None:
            index_columns = {index_column for index_column, _ in index_map}
            self._data_columns = [col for col in var.schema.index if col not in index_columns]
        else:
            self._data_columns = data_columns

        if column_index is None:
            self._column_index = _to_column_index(self._data_columns)
        else:
            assert len(column_index) == len(self._data_columns)
            assert all(isinstance(i, tuple) for i in column_index), column_index
            assert len({len(i) for i in column_index}) <= 1, column_index
            self._column_index = column_index

        if len(self._column_index) != len(set(self._column_index)):
            raise ValueError("DolphinDB does not support duplicated column names")
        
        if column_index_names is not None and not is_list_like(column_index_names):
            raise ValueError('column_index_names should be list-like or None for a MultiIndex')
        
        if (isinstance(column_index_names, list)
            and all(name is None for name in column_index_names)):
            self._column_index_names = None
        else:
            self._column_index_names = column_index_names

        self._update_data_select_list()

    def _update_data_select_list(self):
        self._data_select_list = [f"{self.var_name}.{col}" for col in self.data_columns]

    def __len__(self):
        return len(self.var)

    def __getitem__(self, key):
        keys, _ = check_key_existence(key, self.data_columns)
        return _InternalFrame(self._session, self.var, index_map=self.index_map, data_columns=keys)
                            #   data_columns=keys, index=self.index)

    @classmethod
    def create_any_vector(cls, var, index):
        return cls(var._session, var, index_of_any_vector=index)

    @classmethod
    def from_upload_obj(cls, session, obj):
        var = _ConstantSP.upload_obj(session, obj)
        return cls(session, var)

    @classmethod
    def from_run_script(cls, session, script):
        var = _ConstantSP.run_script(session, script)
        return cls(session, var)

    @classmethod
    def from_pandas(cls, session, pdf: Union[pd.DataFrame, pd.Index]):
        if isinstance(pdf, pd.Index):
            var = _ConstantSP.upload_obj(session, pdf.to_numpy())
            var.as_frame(name=pdf.name)
            index_map = _to_index_map(pdf)
            return cls(session, var, index_map)
        columns = pdf.columns
        if len(columns) == 0 and len(pdf) == 0:    # trivial case
            pdf.index = pd.RangeIndex(0)
        if isinstance(columns, pd.RangeIndex):
            _warn_not_dolphindb_identifier()
            data_columns = [f"{ORCA_COLUMN_NAME_FORMAT(i)}" for i, _ in enumerate(columns)]
        else:
            data_columns = ["_".join(column) if isinstance(column, tuple)
                                else column if is_dolphindb_identifier(column)
                                else f"{ORCA_COLUMN_NAME_FORMAT(i)}"
                            for i, column in enumerate(columns)]
        column_index = _to_column_index(columns)
        column_index_names = columns.names

        index_map = _to_index_map(pdf.index)
        index_columns = [index_column for index_column, _ in index_map]

        reset_index = pdf.reset_index()
        reset_index.columns = index_columns + data_columns
        # TODO: koalas check is datatime
        try:
            var = _ConstantSP.upload_obj(session, reset_index)
        except RuntimeError as e:
            ex_msg = str(e)
            if ex_msg.startswith("Unable to cast Python instance of type"):
                raise RuntimeError(ex_msg + "; You might have created a table with non-string as column names")
            elif ex_msg.startswith("All columns must have the same size"):
                raise RuntimeError(ex_msg + "; You might have passed duplicated column names")
            else:
                raise
        return cls(session, var, index_map, data_columns, column_index, column_index_names)

    @staticmethod
    def check_index_map_validity(index_map):
        if all(isinstance(index_field, str)
               and (index_name is None or isinstance(index_name, tuple))
               for index_field, index_name in index_map):
            return
        else:
            raise ValueError(f"Invalid column map: '{index_map}'")

    @property
    def _column_index_level(self):
        """ Return the level of the column index. """
        column_index = self._column_index
        if len(column_index) == 0:
            return 1
        else:
            levels = set(0 if idx is None else len(idx) for idx in column_index)
            assert len(levels) == 1, levels
            return list(levels)[0]

    @property    # TODO: @lazyproperty
    def _column_index_to_name(self):
        return dict(zip(self.column_index, self.data_columns))

    def column_name_for(self, column_name_or_index):
        if column_name_or_index in self._column_index_to_name:
            return self._column_index_to_name[column_name_or_index]
        else:
            if not isinstance(column_name_or_index, str):
                raise KeyError(column_name_or_index)
            return column_name_or_index

    @property
    def data_columns(self):
        return self._data_columns

    @property    # TODO: @lazyproperty
    def _index_columns(self):
        return [col for col, _ in self.index_map]

    @property
    def column_index(self):
        return self._column_index
    
    @property
    def column_index_names(self):
        return self._column_index_names

    @property
    def index_map(self):
        return self._index_map

    @property
    def index_names(self):
        return [name[0] if isinstance(name, tuple) else name
                for _, name in self._index_map]
        # name = self.index_map[0][1]
        # if name is not None:
        #     return name[0]
        # else:
        #     return name

    @property
    def index_name(self):
        name = self.index_map[0][1]
        if name is not None:
            return name[0]
        else:
            return name

    @property
    def _index_names(self):
        return self.index_names
    
    @property
    def _index_name(self):
        return self.index_name

    # @property
    # def use_range_index(self):
    #     return self._use_range_index

    @property
    def is_any_vector(self):
        return self._is_any_vector

    @property
    def in_memory(self):
        return self.var.in_memory

    @property
    def segmented(self):
        return self.var.segmented

    @property
    def id(self):
        return self.var.id

    @property
    def _var_name(self):    # TODO: lazyproperty
        return self.var.var_name
    
    @property
    def var_name(self):
        return self._var_name

    @property
    def var(self):
        return self._var

    @property
    def form(self):
        return self.var.form

    @property
    def type(self):
        return self.var.type

    @property
    def schema(self):
        return self.var.schema

    @property
    def _ddb_dtypes(self):
        return self.schema.typeInt

    @property
    def _ddb_dtypestr(self):
        class _Typestr(object):
            def __getitem__(this, key):
                return self.schema.typeString[key].lower()
        return _Typestr()

    # @property
    # def index(self):
    #     return self._index

    # @property
    # def dtype(self):
    #     return self._dtype

    @property
    def data_select_list(self):
        return self._data_select_list

    def _to_script(self, ignore_index=False):
        index_columns = [] if ignore_index else self._index_columns
        select_list = itertools.chain(index_columns, self.data_columns)
        return sql_select(select_list, self.var_name)

    def set_columns(self, columns):
        assert len(columns) == len(self.data_columns)
        column_index = _to_column_index(columns)
        if len(column_index) != len(set(column_index)):
            raise ValueError("DolphinDB does not support duplicated column names")
        if isinstance(columns, pd.Index):
            column_index_names = columns.names
        else:
            column_index_names = None
        self._column_index = column_index
        self._column_index_names = column_index_names

    def copy_as_in_memory_table(self, inplace=False):
        session = self._session
        data_columns = self.data_columns
        column_index = self.column_index
        column_index_names = self.column_index_names
        select_list = itertools.chain(self._index_columns, data_columns)
        script = sql_select(select_list, self._var_name)

        if inplace:
            if self.segmented:
                script = f"loadTableBySQL(<{script}>)"
                self._var.reset_with_script(script)
                return
            else:
                return
        if self.segmented:
            script = f"loadTableBySQL(<{script}>)"
        var = _ConstantSP.run_script(session, script)
        return _InternalFrame(session, var, self.index_map, data_columns,
                              column_index, column_index_names)

    def attach_index(self, index):
        from .indexes import Index
        assert isinstance(index, Index)
        var = self.var
        all_columns = self._data_columns + self._index_columns
        index_map = _to_index_map(index, all_columns)
        var.drop_columns(self._index_columns)
        var.attach_index(index, index_map)
        self._index_map = index_map
        self._index = Index._from_internal(self, index)

    def append(self, other, ignore_index, sort):
        # TODO: align columns and sort
        select_list = self.schema.index
        other_script = sql_select(select_list, other._var_name)
        self.var.append(other_script)
        if ignore_index:
            index_map = [(ORCA_INDEX_NAME_FORMAT(0), None)] if self.var.attach_default_index() else []
            self._index_map = index_map

    def rename(self, columns, level):
        self._var.rename(columns=columns)
        column_index_level = self._column_index_level
        new_data_columns = []
        new_column_index = []
        # new_column_index_names = []    # TODO: check correctness
        # for data_column, col_idx, name in \
        #     zip(self._data_columns, self._column_index, self._column_index_names):

        for data_column, col_idx in zip(self._data_columns, self._column_index):
            new_col = columns.get(data_column)
            if new_col is not None:
                if level is None:
                    new_col_idx = tuple([new_col] * column_index_level)
                else:
                    new_col_idx = list(col_idx)
                    new_col_idx[level] = new_col
                    new_col_idx = tuple(new_col_idx)
                new_data_columns.append(new_col)
                new_column_index.append(new_col_idx)
                # new_column_index_names.append(name)
            else:
                new_data_columns.append(data_column)
                new_column_index.append(col_idx)
                # new_column_index_names.append(name)
        self._data_columns = new_data_columns
        self._column_index = new_column_index
        self._update_data_select_list()

    def get_script_with_unary_op(self, func):
        data_columns = self.data_columns
        select_list = (f"{func}({col}) as {col}"
                       for col in data_columns)
        return sql_select(select_list, self._var_name, is_exec=True)


    # def get_select_clause(self):
    #     return ",".join(self.column_names)

    # def get_to_pandas_script(self, select=True):
    #     if not self.segmented and self.is_table_like:
    #         return "{}.{}".format(self.var_name, self.column_names[0])
    #     elif self.is_table_like:
    #         select_or_exec = "select" if select else "exec"
    #         return "{select_or_exec} {select_clause} from {var_name}".format(
    #             select_or_exec=select_or_exec,
    #             select_clause=self.get_select_clause(),
    #             var_name=self.var_name
    #         )
    #     else:
    #         return self.var_name
