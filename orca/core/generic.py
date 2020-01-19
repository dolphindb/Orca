import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from .common import default_session, _set_verbose
from .frame import DataFrame
from .indexes import Index, IndexOpsMixin
from .internal import _ConstantSP, _InternalFrame
from .operator import ArithExpression, BooleanExpression
from .series import Series
from .utils import (ORCA_INDEX_NAME_FORMAT, _infer_dtype, _to_data_column,
                    _unsupport_columns_axis, to_dolphindb_literal,
                    to_dolphindb_type_string)


def connect(host, port, user="admin", passwd="123456", session=default_session()):
    session.connect(host, port, user, passwd)


def set_verbose(verbose=False):
    _set_verbose(verbose)


def read_pickle(path, compression="infer", session=default_session(),  *args, **kwargs):
    pdf = pd.read_pickle(path=path, compression=compression, *args, **kwargs)
    return DataFrame(pdf, session=session)


def read_fwf(filepath_or_buffer, colspecs='infer', widths=None, infer_nrows=100,
             session=default_session(), *args, **kwargs):
    pdf = pd.read_fwf(filepath_or_buffer=filepath_or_buffer,colspecs=colspecs,
                      widths=widths, infer_nrows=infer_nrows, *args, **kwargs)
    return DataFrame(pdf, session=session)


def read_msgpack(path_or_buf, encoding='utf-8', iterator=False, session=default_session(),
                 *args, **kwargs):
    pdf = pd.read_msgpack(path_or_buf=path_or_buf, encoding=encoding, iterator=iterator,
                          *args, **kwargs)
    return DataFrame(pdf, session=session)


def read_clipboard(sep=r'\s+', session=default_session(), **kwargs):
    pdf = pd.read_clipboard(sep=sep, **kwargs)
    return DataFrame(pdf, session=session)


def read_excel(io, sheet_name=0, header=0, names=None, index_col=None, usecols=None,
               squeeze=False, dtype=None, engine=None, converters=None, true_values=None, false_values=None,
               skiprows=None, nrows=None, na_values=None, keep_default_na=True, verbose=False, parse_dates=False,
               date_parser=None, thousands=None, comment=None, skip_footer=0, skipfooter=0, convert_float=True,
               mangle_dupe_cols=True, session=default_session(), *args, **kwargs):
    pdf = pd.read_excel(io, sheet_name=sheet_name, header=header, names=names, index_col=index_col, usecols=None,
                        squeeze=squeeze, dtype=dtype, engine=engine, converters=converters, true_values=true_values,
                        false_values=false_values, skiprows=skiprows, nrows=nrows, na_values=na_values, keep_default_na=keep_default_na,
                        verbose=verbose, parse_dates=parse_dates,
                        date_parser=date_parser, thousands=thousands, comment=comment, skip_footer=skip_footer,
                        skipfooter=skipfooter, convert_float=convert_float,
                        mangle_dupe_cols=mangle_dupe_cols, *args, **kwargs)
    return DataFrame(pdf, session=session)


def ExcelWriter(path, engine=None, date_format=None, datetime_format=None, mode='w', *args, **kwargs):
    return pd.ExcelWriter(path=path, engine=engine, date_format=date_format, datetime_format=datetime_format,
                          mode=mode, *args, **kwargs)


def read_json(path_or_buf=None, orient=None, typ='frame', dtype=None, convert_axes=None,
              convert_dates=True, keep_default_dates=True, numpy=False,
              precise_float=False, date_unit=None, encoding=None, lines=False,
              chunksize=None, compression='infer', session=default_session(), *args, **kwargs):
    pdf = pd.read_json(path_or_buf=path_or_buf, orient=orient, typ=typ, dtype=dtype,
                       convert_axes=convert_axes, convert_dates=convert_dates,
                       keep_default_dates=keep_default_dates, numpy=numpy,
                       precise_float=precise_float, date_unit=date_unit,
                       encoding=encoding, lines=lines, chunksize=chunksize,
                       compression=compression, *args, **kwargs)
    return DataFrame(pdf, session=session)


def json_normalize(data, record_path = None, meta = None, meta_prefix = None,
                   record_prefix = None, errors='raise', sep='.', max_level=None,
                   session=default_session(), *args, **kwargs):
    from pandas.io.json import json_normalize as pdjson_normalize
    pdf = pd.json_normalize(data=data,record_path=record_path,meta=meta,meta_prefix=meta_prefix,record_prefix=record_prefix,
                            errors=errors, sep=sep, max_level=max_level, *args, **kwargs)
    return DataFrame(pdf, session=session)


def build_table_schema(data, index=True, primary_key=None, version=True,
                       session=default_session(), *args, **kwargs):
    from pandas.io.json import build_table_schema as pdbuild_table_schema
    pdf = pd.build_table_schema(data=data, index=index, primary_key=primary_key,
                                version=version, *args, **kwargs)
    return DataFrame(pdf, session=session)


def read_html(io, match=".+", flavor=None, header=None, index_col=None, skiprows=None,
              attrs=None, parse_dates=False, thousands=",", encoding=None, decimal=".",
              converters=None, na_values=None, keep_default_na=True, displayed_only=True,
              session=default_session(), *args, **kwargs):
    pdf = pd.read_html(io, match=match, flavor=flavor, header=header, index_col=index_col,
                       skiprows=skiprows, attrs=attrs, parse_dates=parse_dates, thousands=thousands,
                       encoding=encoding, decimal=decimal, converters=converters, na_values=na_values,
                       keep_default_na=keep_default_na, displayed_only=displayed_only, *args, **kwargs)
    return DataFrame(pdf, session=session)


def read_hdf(path_or_buf, key=None, mode='r', session=default_session(), *args, **kwargs):
    pdf = pd.read_hdf(path_or_buf, key=key, mode=mode, *args, **kwargs)
    return DataFrame(pdf, session=session)


def read_feather(path, columns=None, use_threads=True, session=default_session(), *args, **kwargs):
    pdf = pd.read_feather(path, columns=columns, use_threads=use_threads, *args, **kwargs)
    return DataFrame(pdf, session=session)


def read_parquet(path, engine='auto', columns=None, session=default_session(), *args, **kwargs):
    pdf = pd.read_parquet(path, engine=engine, columns=columns, *args, **kwargs)
    return DataFrame(pdf, session=session)


def read_sas(filepath_or_buffer, format=None, index=None, encoding=None, chunksize=None,
             iterator=False, session=default_session(), *args, **kwargs):
    pdf = pd.read_sas(filepath_or_buffer,format=format,index=index,encoding=encoding,
                       chunksize=chunksize,iterator=iterator,*args, **kwargs)
    return DataFrame(pdf, session=session)


def read_sql_table(table_name, con, schema=None, index_col=None,
                   coerce_float=True, parse_dates=None, columns=None,
                   chunksize=None, session=default_session(), *args, **kwargs):
    pdf = pd.read_sql_table(table_name, con, schema=schema, index_col=index_col,
                            coerce_float=coerce_float, parse_dates=parse_dates,
                            columns=columns, chunksize=chunksize, *args, **kwargs)
    return DataFrame(pdf, session=session)


def read_sql_query(sql, con, index_col=None, coerce_float=True, params=None,
                   parse_dates=None, chunksize=None, session=default_session(),
                   *args, **kwargs):
    pdf = pd.read_sql_query(sql, con, index_col=index_col, coerce_float=coerce_float, params=params,
                            parse_dates=parse_dates, chunksize=chunksize,*args, **kwargs)
    return DataFrame(pdf, session=session)


def read_sql(sql, con, index_col=None, coerce_float=True, params=None,
             parse_dates=None, columns=None, chunksize=None,
             session=default_session(), *args, **kwargs):
    pdf = pd.read_sql(sql, con, index_col=index_col, coerce_float=coerce_float, params=params,
                      parse_dates=parse_dates, columns=columns, chunksize=chunksize,
                      *args, **kwargs)
    return DataFrame(pdf, session=session)


def read_gbq(query, project_id=None, index_col=None, col_order=None, reauth=False,
             auth_local_webserver=False, dialect=None, location=None, configuration=None,
             credentials=None, use_bqstorage_api=None, private_key=None, verbose=None,
             session=default_session(), *args, **kwargs):
    pdf = pd.read_gbq(query, project_id=project_id, index_col=index_col, col_order=col_order,
                      reauth=reauth, auth_local_webserver=auth_local_webserver,
                      dialect=dialect, location=location, configuration=configuration,
                      credentials=credentials, use_bqstorage_api=use_bqstorage_api,
                      private_key=private_key, verbose=verbose, *args, **kwargs)
    return DataFrame(pdf, session=session)


def read_stata(filepath_or_buffer, convert_dates=True, convert_categoricals=True, encoding=None, index_col=None,
               convert_missing=False, preserve_dtypes=True, columns=None, order_categoricals=True,
               chunksize=None, iterator=False, session=default_session(), *args, **kwargs):
    pdf = pd.read_stata(filepath_or_buffer, convert_dates=convert_dates,
                        convert_categoricals=convert_categoricals, encoding=encoding,
                        index_col=index_col, convert_missing=convert_missing,
                        preserve_dtypes=preserve_dtypes, columns=columns,
                        order_categoricals=order_categoricals, chunksize=chunksize,
                        iterator=iterator, *args, **kwargs)
    return DataFrame(pdf, session=session)


def from_pandas(pdf, session=default_session()):
    if isinstance(pdf, pd.DataFrame):
        return DataFrame(pdf, session=session)
    elif isinstance(pdf, pd.Series):
        return Series(pdf, session=session)
    else:
        raise TypeError("pdf must be a pandas DataFrame or pandas Series")


def read_csv(path, sep=',', delimiter=None, names=None,  index_col=None,
             engine='dolphindb', usecols=None, squeeze=False, prefix=None, dtype=None,
             partitioned=True, db_handle=None, table_name=None, partition_columns=None,
             session=default_session(), *args, **kwargs):
    """
    :param partitioned: If False, then load with the function loadText
    :param path: file path
    :param sep:str, default ','
                If the sep is not char type, you must set the engine as 'c' or 'python'.
                Delimiter to use. If sep is None, the C engine cannot automatically detect
                the separator, but the Python parsing engine can, meaning the latter will
                be used and automatically detect the separator by Python's builtin sniffer
                tool, ``csv.Sniffer``. In addition, separators longer than 1 character and
                different from ``'\s+'`` will be interpreted as regular expressions and
                will also force the use of the Python parsing engine. Note that regex
                delimiters are prone to ignoring quoted data. Regex example: ``'\r\t'``.
    :param delimiter:
                str, default ``None``
                Alias for sep.
    :param names:
                array-like, optional
                List of column names to use. If file contains no header row. Duplicates in this list are not
                allowed.
    :param index_col:
                int, str, or False, default ``None``
              Column(s) to use as the row labels of the ``DataFrame``, either given as
              string name or column index.
    :param engine:
             {{'c', 'python','dolphindb'}}, optional
                Parser engine to use. The C engine is faster while the python engine is
                currently more feature-complete. And when sep or delimiter is not char type,
                you must choose 'c' or 'python'.
    :param usecols:
             list-like or callable, optional
            Return a subset of the columns. If list-like, all elements must either
            be positional (i.e. integer indices into the document columns) or strings
            that correspond to column names provided either by the user in `names` or
            inferred from the document header row(s). For example, a valid list-like
            `usecols` parameter would be ``[0, 1, 2]`` or ``['foo', 'bar', 'baz']``.
            Element order is ignored, so ``usecols=[0, 1]`` is the same as ``[1, 0]``.
    :param squeeze:
            bool, default False
            If the parsed data only contains one column then return a Series.
    :param prefix:
            str, optional
            Prefix to add to column numbers when no header, e.g. 'X' for X0, X1, ...
    :param dtype:
    :param session:
    :param args:
    :param kwargs:
    :return: DataFrame
             When squeeze is true, return value may be Series
    :raise: RuntimeError, File may not exist
            ValueError, the param pattern may not right.
    """
    # check engine if engige is 'python' or 'c', just use the pandas function to read csv file. Notice that pandas read file from the local address.
    # If you must set the engine type to assure the program run, set it to 'c'.
    if engine == 'python' or engine == 'c':
        df = pd.read_csv(path, sep=sep, delimiter=delimiter,names=names,index_col=index_col,
                         engine=engine,usecols=usecols,squeeze=squeeze,prefix=prefix,dtype=dtype,*args,**kwargs)
        return DataFrame(df, session=session)
    elif engine != 'dolphindb':
        raise ValueError("The engine just support 'python', 'c', 'dolphindb'. Please check the engine type")

    path_var = _ConstantSP.upload_obj(session, path)
    path_name = path_var.var_name
    # support param sep and delimiter
    if delimiter is None:
        delimiter = sep

    if delimiter is not None:
        if len(delimiter) > 1:
            raise ValueError("'dolphindb' engine only supports char type.")
    else:
        delimiter = ','

    try:
        schema = _ConstantSP.run_script(session, f"extractTextSchema({path_name},'{delimiter}')")
    except RuntimeError:
        raise RuntimeError("The file might not exsit")
    schema_name = schema.var_name
    # dolphindb will set the char to int8, set the right dtype
    session.run(f"update {schema_name} set type = 'SYMBOL' where type = 'CHAR'")

    if dtype is not None:
        update_script = ";".join(
            "update {} set type = {} where name = {}".format(
                schema_name,
                to_dolphindb_literal(to_dolphindb_type_string(np_type)),
                to_dolphindb_literal(name)
            )
            for name, np_type in dtype.items()
        )
        session.run(update_script)

    # when name is None, set the prefix
    update_script_appendix = None
    if names is None:
        if  prefix is not None:
            update_script_appendix = to_dolphindb_literal([f"{prefix}{i}" for i in range(len(schema))])
            # TODO: limit the length of column name?
    else:
        # names length may not match the schema
        if len(names) >= len(schema):
            names_list = names[0:len(schema)]
            update_script_appendix = to_dolphindb_literal(names_list)
        elif len(names) > len(schema):
            warnings.warn("DolphinDB does not support changing part of the column names")

    # if should update schema
    if update_script_appendix is not None:
        update_script = f"update {schema_name} set name = {update_script_appendix}"
        session.run(update_script)
    
    if db_handle is not None or table_name is not None or partition_columns is not None:
        if db_handle is None or table_name is None or partition_columns is None:
            raise ValueError("'db_handle', 'table_name' and 'partition_columns' should be all specified or all unspecified")
        if not partitioned:
            raise ValueError("partitioned must be True if db_handle, table_name, and partition_columns are specified")
        try:
            db_handle = to_dolphindb_literal(db_handle)
            db_handle_var = _ConstantSP.run_script(session, f"database({db_handle})")
        except RuntimeError:
            raise RuntimeError(f"The database {db_handle} does not exist")
        table_name = to_dolphindb_literal(table_name)
        partition_columns = to_dolphindb_literal(partition_columns)
        script = f"loadTextEx({db_handle_var._var_name}, {table_name}, {partition_columns}, {path_name}, '{delimiter}', {schema_name})"
    elif partitioned:
        script = f"ploadText({path_name}, '{delimiter}', {schema_name})"
    else:
        script = f"loadText({path_name}, '{delimiter}', {schema_name})"

    try:
        odf = _InternalFrame.from_run_script(session, script)     # TODO: In-memory?
    except RuntimeError:
        raise RuntimeError("The file might not exsit or the schema format might be invalid")

    data = DataFrame(odf, session=session)

    # param index_col
    if index_col is True:
        raise ValueError("The value of index_col couldn't be True")
    elif index_col is not None and index_col is not False:
        if not isinstance(index_col, (list, tuple, np.ndarray)):
            index_col = [index_col]
        if isinstance(index_col[0],int):
            column_names=[data.columns[i] for i in index_col] # index may out of range
        else:
            column_names=index_col

        data.set_index(column_names,inplace=True)

    # param usecol, may be str
    msg = (
        "'usecols' must either be list-like of all strings, all unicode, "
        "all integers."
    )
    if usecols is not None:
        if not isinstance(usecols, (list, tuple, np.ndarray)):
            usecols = [usecols]
        column_names= usecols
        if _infer_dtype(usecols) not in ('int','str','unicode','empty'):
            raise ValueError(msg)
        if _infer_dtype(usecols) == 'int':
            column_names = [data._data_columns[i] for i in usecols]
        data = data[column_names]

    # param squeeze
    if squeeze:
        if data.columns.__len__() == 1:
            data = data.squeeze()

    return data


def read_shared_table(table_name, session=default_session()):
    data = _InternalFrame.from_run_script(session, table_name)
    return DataFrame(data, session=session)


def read_table(database, table_name, partition=None, session=default_session()):
    partition = "NULL" if partition is None else partition
    script = "loadTable({}, {}, {})".format(
        to_dolphindb_literal(database),
        to_dolphindb_literal(table_name),
        partition
    )
    data = _InternalFrame.from_run_script(session, script)     # TODO: In-memory?
    return DataFrame(data, session=session)


def concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=None, copy=True):
    axis = _unsupport_columns_axis(None, axis)

    if (isinstance(objs, (DataFrame, IndexOpsMixin))
            or not isinstance(objs, Iterable)):
        raise TypeError(f"first argument must be be an iterable of orca objects; "
                        f"you passed an object of type '{type(objs).__name__}'")
    if join not in ('inner', 'outer'):
        raise ValueError("Only can inner (intersect) or outer (union) join the other axis")
    if len(objs) == 0:
        raise ValueError("no objects to concatenate")
    objs = [obj for obj in objs if obj is not None]
    if len(objs) == 0:
        raise ValueError("all objects passed were None")
    for obj in objs:
        if not isinstance(obj, (Series, DataFrame)):
            if isinstance(obj, (ArithExpression, BooleanExpression)):
                raise TypeError(f"cannot concatenate object of type '{type(obj)}.__name__'; "
                                f"try converting an expression to a DataFrame by calling compute()")
            else:
                raise TypeError(f"cannot concatenate object of type '{type(obj)}.__name__'; "
                                f"only orca.Series and orca.DataFrame are valid")
    should_return_series = all(isinstance(obj, Series) for obj in objs)

    column_index_levels = {obj._column_index_level for obj in objs}
    if len(column_index_levels) != 1:
        raise ValueError("MultiIndex columns should have the same levels")

    if not ignore_index:
        indices = [obj.index for obj in objs]
        first_index = indices[0]
        for index in indices:
            if index.names != first_index.names:
                raise ValueError(
                    f"Index type and names should be same in the objects to concatenate. "
                    f"You passed different indices {first_index.names} and {index.names}")

    column_indices = [obj._column_index for obj in objs]
    if ignore_index:
        index_names = [[] for _ in objs]
    else:
        index_names = [obj._index_names for obj in objs]
    if (all(name == index_names[0] for name in index_names)
            and all(index == column_indices[0] for index in column_indices)):
        merged_columns = column_indices[0]
        column_index = merged_columns
        inner_join = True
    else:
        if join == "inner":
            interested_columns = set.intersection(*map(set, column_indices))
            merged_columns = [column for column in column_indices[0] if column in interested_columns]
            column_index = merged_columns
            inner_join = True
        elif join == "outer":
            # get the map {column: column_typestr}
            merged_columns = {}
            for column_index, obj in zip(column_indices, objs):
                for column in column_index:
                    if column not in merged_columns:
                        data_column = _to_data_column(column)
                        merged_columns[column] = obj._ddb_dtypestr[data_column]
            merged_columns = sorted(merged_columns.items()) if sort else merged_columns.items()    # TODO: check correctness
            column_index = [index for index, _ in merged_columns]
            inner_join = False
    session = objs[0]._session
    if len(merged_columns) == 0 and ignore_index:
        # a trivial case where the returned df contains one RangeIndex and no data columns
        df_length = sum(map(len, objs))
        script = f"table(0..{df_length-1} as {ORCA_INDEX_NAME_FORMAT(0)})"
        return DataFrame._get_from_script(session, script, index_map=[(ORCA_INDEX_NAME_FORMAT(0), None)])
    table_selections = ",".join(obj._concat_script(merged_columns, ignore_index, inner_join) for obj in objs)
    index_map = None if ignore_index else objs[0]._index_map
    script = f"unionAll([{table_selections}], false)"    # TODO: partitioned=true or false?
    df = DataFrame._get_from_script(session, script, index_map=index_map, column_index=column_index)
    if should_return_series:
        s = df[objs[0]._data_columns[0]]
        s.rename(None, inplace=True)    # unionAll uses the first object's column name
        return s
    else:
        return df


def save_table(db_path, table_name, df, ignore_index=True, session=default_session()):
    if not isinstance(df, (DataFrame, ArithExpression, BooleanExpression)):
        raise TypeError("df must be a DataFrame")
    df_script = df._to_script(ignore_index=ignore_index)
    if db_path.startswith("dfs://"):
        db_path = to_dolphindb_literal(db_path)
        table_name = to_dolphindb_literal(table_name)
        session.run(f"tableInsert(loadTable({db_path},{table_name}),{df_script})")
    else:
        db_handle = to_dolphindb_literal(db_path)
        db_handle_var = _ConstantSP.run_script(session, f"database({db_handle})")
        table_name = to_dolphindb_literal(table_name)
        session.run(f"saveTable({db_handle_var._var_name},{df_script},{table_name})")


def isna(obj):
    if isinstance(obj, (Series, DataFrame, Index)):
        return obj.isna()
    else:
        return pd.isna(obj)


def notna(obj):
    if isinstance(obj, (Series, DataFrame, Index)):
        return obj.notna()
    else:
        return pd.notna(obj)


isnull = isna
notnull = notna


def to_datetime(arg, session=default_session(), *args, **kwargs):
    if isinstance(arg, DataFrame):
        raise NotImplementedError()
    if isinstance(arg, Series):
        raise NotImplementedError()

    data = pd.to_datetime(arg, *args, **kwargs)
    return data


def date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed=None, session=default_session(), **kwargs):
    # TODO: write your own parse function
    # pdate_range = pd.date_range(start=start, end=end, periods=periods, freq=freq, tz=tz, normalize=normalize, name=name, closed=closed, **kwargs)
    # return DatetimeIndex(pdate_range, session=session)
    return pd.date_range(start=start, end=end, periods=periods, freq=freq, tz=tz, normalize=normalize, name=name, closed=closed, **kwargs)


def qcut(x, q, labels=None, retbins=False, precision=3, duplicates='raise', session=default_session()):
    if not isinstance(x, Series):
        return pd.qcut(x=x, q=q, labels=labels, retbins=retbins,
                       precision=precision, duplicates=duplicates)
    if not isinstance(q, int):
        raise NotImplementedError()
    if q <= 0:
        raise ValueError("q must be greater than 0")
    if labels is not None:
        raise NotImplementedError()
    if retbins:
        raise NotImplementedError()
    var = x._to_script(ignore_index=True, is_exec=True)
    script = f"rank({var},,{q})"
    return session.run(script)