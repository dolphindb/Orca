# Orca User Guide

 - [1 Installation](#1-installation)   
 - [2 Quick start](#2-quick-start)
 - [3 Architecture](#3-architecture)
 - [4 Limitations](#4-limitations)
 - [5 Best practices](#5-best-practices) 
 - [6 How to implement functionalities not supported by orca](#6-how-to-implement-functionalities-not-supported-by-orca)

## 1 Installation

Orca supports Linux and Windows. It requires Python version 3.6 and above, and pandas version 0.25.1 and above.

The orca project has been integrated into [DolphinDB Python API] (https://github.com/dolphindb/Tutorials_EN/blob/master/python_api.md). 

To use orca, we need to install DolphinDB Python API:
```
pip install dolphindb
```

Open a DolphinDB server, connect to this server with function `connect(host, port, username, password)`:
```python
>>> import dolphindb.orca as orca
>>> orca.connect("localhost", 8848, "admin", "123456")
```

Now orca is ready to use. 

If you would like to use a pandas program in orca, you can change the following statement in pandas
```python
import pandas as pd
```
to the following orca statements:
```python
import dolphindb.orca as pd
pd.connect("localhost", 8848, "admin", "123456")
```

## 2 Quick start

Create an orca Series with a list of values. Orca automatically generates a default index for it:
```python
>>> s = orca.Series([1, 3, 5, np.nan, 6, 8])
>>> s

0 1.0
1 3.0
2 5.0
3 NaN
4 6.0
5 8.0
dtype: float64
```

An orca DataFrame can be created with a dictionary. Each element in the dictionary must be a Series or an object that can be converted to a Series:
```python
>>> df = orca.DataFrame(
... {"a": [1, 2, 3, 4, 5, 6],
... "b": [100, 200, 300, 400, 500, 600],
... "c": ["one", "two", "three", "four", "five", "six"]},
... index = [10, 20, 30, 40, 50, 60])
>>> df
    a b c
10 1 100 one
20 2 200 two
30 3 300 three
40 4 400 four
50 5 500 five
60 6 600 six
```

An orca DataFrame can also be created with a pandas DataFrame:
```python
>>> dates = pd.date_range('20130101', periods=6)
>>> pdf = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
>>> df = orca.DataFrame(pdf)
>>> df
                   A         B         C         D
2013-01-01  0.758590 -0.180460 -0.066231  0.259408
2013-01-02  1.165941  0.961164 -0.716258  0.143499
2013-01-03  0.441121 -0.232495 -0.275688  0.516371
2013-01-04  0.281048 -0.782518 -0.683993 -1.474788
2013-01-05 -0.959676  0.860089  0.374714 -0.535574
2013-01-06  1.357800  0.729484  0.142948 -0.603437
```

Check whether 'df' is an orca DataFrame:
```
>>> type(df)
<class 'orca.core.frame.DataFrame'>
```

Use function `head` to view the top few lines of an orca object:
```python
>>> df.head()
                   A B C D
2013-01-01 0.758590 -0.180460 -0.066231 0.259408
2013-01-02 1.165941 0.961164 -0.716258 0.143499
2013-01-03 0.441121 -0.232495 -0.275688 0.516371
2013-01-04 0.281048 -0.782518 -0.683993 -1.474788
2013-01-05 -0.959676 0.860089 0.374714 -0.535574
```

View the index and column name of an orca object with `index` and `columns`:
```python
>>> df.index
DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype = 'datetime64[ns]', freq = 'D')

>>> df.columns
Index(['A', 'B', 'C', 'D'], dtype = 'object')
```

Convert an orca DataFrame to a pandas DataFrame with `to_pandas`:
```
>>> pdf1 = df.to_pandas()
>>> type(pdf1)
<class 'pandas.core.frame.DataFrame'>
```

Load a CSV file with `read_csv`. As the CSV file must be located on DolphinDB server, the file path in `read_csv` is the path on the server. 
```python
>>> df = orca.read_csv("/home/DolphinDB/orca/databases/USPrices.csv")
```

## 3 Architecture

The top layer of orca is DolphinDB pandas API and the bottom layer is DolphinDB database. DolphinDB Python API implements the communication between orca client and DolphinDB server. DolphinDB script is generated on the client with pandas API, and then sent to the DolphinDB server to execute through DolphinDB Python API. An orca DataFrame only keeps the metadata of the corresponding DolphinDB table. All columns are stored on DolphinDB server and calculation is also conducted on DolphinDB server.

### How data is stored in orca

A DataFrame or a Series in orca is stored as a table in DolphinDB. Columns and indices are stored in the same table. A DolphinDB table represented by an orca DataFrame contains one or multiple data columns and index columns. The DolphinDB table represented by an orca Series contains a column and one or multiple indices. This makes operations such as index alignment, grouping and aggregation, and calculation involving multiple columns very efficient. 

An orca DataFrame only stores metadata of the corresponding DolphinDB table, including table names, column names, index names, etc. When we fetch a column from a DataFrame, a Series is returned. The Series corresponds to the same DolphinDB table as the DataFrame. Only the metadata of the 2 orca objects are different.

## 4 Limitations

Currently orca has the following limitations:

### 4.1 Data type

As a data type must be specified for each column of a DolphinDB table and an ANY vector cannot be used as a column of a DolphinDB table, a column of an orca DataFrame cannot have mixed data types. In addition, the elements in a column must be scalars of Python's int, float, string types and cannot be Python's list, dict types. Functions designed for list or dict types, such as `DataFrame.explode`, are not supported by orca.

### 4.2 Column name

The column names in an orca DataFrame must be valid DolphinDB variable names. They should contain only letters, numbers or underscores; start with a letter; are not reserved words for DolphinDB such as 'if'.

DolphinDB does not allow duplicate column names. So an orca DataFrame cannot have 2 columns with identical column names.

For certain special columns (such as index), orca assigns column names that begin with ORCA_. We should avoid using strings that begin with ORCA_ as column names.

### 4.3 Partitioned table

As there is no order among partitions in a partitioned table in DolphinDB, pandas.RangeIndex does not apply in orca. Therefore, if an orca DataFrame represents a partitioned table in DolphinDB, the following operations are not supported:
- Access rows through 'iloc'.
- Assign a Series or DataFrame of a different partition scheme to it.

### 4.4 Distributed calls

Some of DolphinDB's built-in functions currently do not support distributed calls, such as `median`,` quantile` and `mad`.

### 4.5 Null value

Null values in DolphinDB are represented by the minimum value of each data type. Null values in pandas are represented by nan, which is a floating point. Orca processes Null values in the same as DolphinDB. 

When data is dowloaded from DolphinDB server to pandas, numeric columns with Null values are converted into floating point types and Null values are converted into nan. 

As Null values in a string column in pandas are still nan, a string vector with Null values in pandas contains both string and floating point types. As [a column in a DolphinDB table cannot have mixed data types](# Data type), to upload a string vector that contains Null values from pandas to DolphinDB, we need to first change nan to empty string "":
```python
df = pd.DataFrame({"str_col": ["hello", "world", np.nan]})
odf = orca.DataFrame(df.fillna({"str_col": ""}))
```

### 4.6 Column-wise operations

As a columnar database, DolphinDB provides better support for row-wise operations than column-wise operations. In pandas, row-wise calculations can be conducted by specifying axis = 0 or axis = 'index' in functions, and column-wise calculations can be conducted by specifying axis = 1 or axis = 'columns' in functions. Orca functions in most cases only support row-wise calculations (axis = 0 or axis = 'index'). Only a few functions, such as `sum`,` mean`, `max`,` min`, `var`,` std`, etc., support column-wise calculations. For many aggregate functions such as average or sum, row-wise calcualtions (calculation over a column) have far better performance than column-wise calculations (calculation over a row). 

An orca DataFrame does not support function `transpose`, as a column in the transposed DataFrame may contain mixed types of data.

### 4.7 Python callables cannot be used as arguments

As DolphinDB Python API currently cannot parse Python functions, functions such as `DataFrame.apply` or ` DataFrame.agg` cannot accept a Python callable object as a parameter.

Orca provides a way to get around this: use a Python string that represents DolphinDB script as the parameter. The DolphinDB script can be DolphinDB's built-in functions, user-defined functions or conditional expressions. For more details, please refer to the section [Higher-Order Functions](# Higher-Order Functions).

## 5 Best Practices

### 5.1 Avoid unnecessary use of `to_pandas` and `from_pandas` 

Orca communicates with the server via DolphinDB Python API. Data storage, query and calculation all happen on the server. Orca is just a client that provides a pandas-like interface. The bottleneck of the system is often network communication. 

When function `to_pandas` is used to convert an orca object into a pandas object, the server will transfer the entire DolphinDB object to the client. Such operations should generally be avoided if not necessary. 

The following operations call `to_pandas` implicitly and should be used only if necessary:

- Print an orca DataFrame or Series representing a non-partitioned table
- Call `to_numpy` or use 'values'
- Call `Series.unique`, `orca.qcut` or other functions that return numpy.ndarray
- Call functions such as `plot` to draw graphs
- Export orca objects in third party format

Similarly, `from_pandas` uploads local pandas object to the DolphinDB server. When the 'data' parameter of `orca.DataFrame` or `orca.Series` is not an orca object, a pandas object is created and then uploaded to the DolphinDB server. 

### 5.2 Lazy evaluation

Orca uses a lazy evaluation strategy. The following 2 types of operations are not immediately evaluated on the server side. Instead, they are converted into intermediate expressions and are not executed until triggered with function `compute`. 

Please note that lazy evaluation strategy is adopted only when the calculation involves one orca object. Calculations involving multiple orca objects don't use lazy evaluation strategy. 

#### 5.2.1 Elementary arithmetic calculations, logical calculations and arithmetic functions that are not aggregate functions


```python
>>> df = orca.DataFrame({"a": [1, 2, 3], "b": [10, 10, 30], "c": [10, -5, 0]})
>>> x = df["a"] + df["b"]
>>> x    # not calculated yet
<orca.core.operator.ArithExpression object at 0x0000027FA5527B70>

>>> x.compute()    # trigger the calculation
0    11
1    12
2    33
dtype: int64
```
```
>>> y = df.c.abs()
>>> y    # not calculated yet
<orca.core.operator.ArithExpression object at 0x0000015A49C0D13>

>>> y.compute()
0    10
1    5
2    0
dtype: int64
```
```python
>>> c = df.cumsum()
>>> c
<dolphindb.orca.core.operator.ArithExpression at 0x2b2b487dcf8>

>>> c.compute()    
   a   b   c
0  1  10  10
1  3  20   5
2  6  50   5
```

```python
>>> c = df.transform("sqrt")
>>> c
<dolphindb.orca.core.operator.ArithExpression at 0x2b2b484d048>

>>> c.compute()    
          a         b         c
0  1.000000  3.162278  3.162278
1  1.414214  3.162278       NaN
2  1.732051  5.477226  0.000000
```
Please note that aggregate functions do not use lazy evaluation strategy.

#### 5.2.2 Conditional filtering

```python
>>> d = df[df["a"] > 2]
>>> d
<orca.core.frame.DataFrame object with a WHERE clause>

>>> d.compute()    
   a   b
2  3  30
```
```python
>>> d = df[df.a.isin([2, 3])]
>>> d
<'dolphindb.orca.core.frame.DataFrame' object with a WHERE clause>

>>> d.compute()    
   a   b  c
1  2  10 -5
2  3  30  0
```

### 5.3 Avoid applying NumPy functions to orca objects

We should avoid applying NumPy functions to orca objects. Use orca functions or [DolphinDB NumPy functions](https://github.com/dolphindb/orca/blob/master/tutorial_en/dolphindb_numpy.md) with orca objects instead. 

NumPy functions are often used to process a DataFrame or Series in pandas. For example:
```python
>>> ps = pd.Series([1,2,3])
>>> np.log(ps)
0 0.000000
1 0.693147
2 1.098612
dtype: float64
```

As a NumPy function does not recognize orca objects, it treats an orca object as a general array-like object and iterates over it. This will cause a lot of unnecessary network overhead, and the result is not an orca object. In some cases, exceptions may be thrown.

Orca provides some commonly used arithmetic functions, such as `log`,` exp`, etc. The script above can be rewritten in orca as follows:

```python
>>> os = orca.Series([1,2,3])
>>> os.log()
<orca.core.operator.ArithExpression object at 0x000001FE099585C0>
```

`os.log()` adopts the [lazy evaluation](#Lazy evaluation) strategy:
```
>>> os = orca.Series ([1,2,3])
>>> tmp = os.log ()
>>> tmp + = os * 2
>>> tmp = tmp.exp ()
>>> tmp
<orca.core.operator.ArithExpression object at 0x000001FE0C374E10>
>>> tmp.compute ()
0 7.389056
1 109.196300
2 1210.286380
dtype: float64
```

### 5.4 Restrictions on modifying DataFrames

The data type of an orca DataFrame column cannot be modified. 

A table that is not an in-memory table (such as a DFS table) has the following restrictions:
- Cannot add new columns
- Cannot be modified with 'update' statement

A partitioned table has the following restriction:
- Cannot assign a vector to a column with 'update' statement

Possible reasons of failures in modifying an orca object:
- The updated data types are incompatible with the original data type. For example, we cannot assign a string vector to an integer column. 
- If we create an orca DataFrame representing a partitioned table without adding an idex, orca cannot automatically add a default index, as a new column cannot be added to a partitioned table. A warning message will be generated. 
- When we add or update a column to an orca DataFrame representing a partitioned table, the column can only be the result of calculations based on data of the object itself, instead of a Python or NumPy array or an orca Series representing an in-memory table. 

If we add columns or modify data in an orca object representing a DolphinDB table that is not an in-memory table, this table will be loaded into memory. If the size of the table is very large, it may cause out-of-memory problem. Therefore, we should avoid modifying such orca objects.

For example, to calculate the product of two columns for each group in a DFS table, the following orca script adds a new column to the orca object df. This will load the DFS table as an in-memory table. There may be memory or performance issues when DFS table is very large:

```python
df = orca.load_table("dfs://orca", "tb")
df["total"] = df["price"] * df["amount"] # Will load the DFS table as an in-memory segmented table!
total_group_by_symbol = df.groupby(["date", "symbol"])["total"].sum()
```

We can take the following steps to avoid creating a new column so that large amounts of data won't be loaded into memory unnecessarily:
- 1. Set the grouping fields 'date' and 'symbol' as index with `set_index`, and set parameter 'level' accordingly in `groupby`
- 2. Set parameter 'lazy' of `groupby` to True for lazy evaluation. 
```python
df = orca.load_table("dfs://orca", "tb")
df.set_index(["date", "symbol"], inplace=True)
total = df["price"] * df["amount"]     # The DFS table is not loaded into memory
total_group_by_symbol = total.groupby(level=[0,1], lazy=True).sum()
```
Some orca functions do not support parameter 'inplace', as it may involve modifying a partitioned table.

### 5.5 Higher-order functions

Pandas functions such as `DataFrame.apply` and `groupby.filter` can accept a Python callable object as a parameter. Orca essentially uses the Python API to parse the user's program into DolphinDB script. Therefore, orca does not support parsing Python callables. With a Python callable, these functions will attempt to convert an orca object to a pandas object, call the corresponding pandas interface, and then convert the result back to an orca object. This will not only incur additional network communication, but also return a new orca object, which makes calculations inefficient compared with the case where all calculations are conducted on the same DataFrame.

As an alternative, for these interfaces, orca can accept a string and pass this string to DolphinDB for execution. This string can be a DolphinDB built-in function (or a partial application of a built-in function), a user-defined function in DolphinDB, or a conditional expression in DolphinDB, etc. 

#### 5.5.1 Grouped weighted average

pandas:
```python
wavg = lambda df: (df["prc"] * df["vol"]).sum() / df["vol"].sum()
df.groupby("symbol").apply(wavg)
```

orca:
```python
df.groupby("symbol")["prc"].apply("wavg{,vol}")
```

The orca script applies a partial application `wavg{, vol}` to column 'prc' after grouping. It is equivalent to the following DolphinDB script:
```SQL
select wavg{,vol}(prc) from df group by symbol
```

or

```SQL
select wavg(prc,vol) from df group by symbol
```

#### 5.5.2 Filter by criteria after grouping

pandas:
```python
df.groupby("symbol").filter(lambda x: len(x) > 1000)
```

orca:
```python
df.groupby("symbol").filter("size(*) > 1000")
```

The orca script above is equivalent to the following DolphinDB script:
```SQL
select * from df context by symbol having size(*) > 10000
```

#### 5.5.3 Apply an arithmetic function to all elements of a Series

pandas:

```python
s.apply(lambda x: x + 1)
```

orca:

```python
s.apply("(x->x+1)")
```

pandas:

```python
s.apply(np.log)
```

orca:

```python
s.apply("log")
```

Orca provides frequently used arithmetic functions, such as `log`, `exp`, `floor`, `ceil`, trigonometric functions, inverse trigonometric functions, etc. 

#### 5.5.4 Implement DolphinDB 'context by' statements

Function `groupby` together with functions such as `shift`, `cumsum`, `bfill` in pandas can implement DolphinDB's [context by clauses](https://www.dolphindb.com/help/contextby.html) with functions such as `move`, `cumsum`, `bfill`. 

pandas:
```
df.groupby("symbol")["prc"].shift(1)
```
orca:
```
df.groupby("symbol")["prc"].shift(1).compute()
```

Please note the different syntax of lambda function in pandas and orca:

pandas:
```python
df.groupby("symbol")["prc"].transform(lambda x: x - x.mean())
```

orca：
```python
df.groupby("symbol")["prc"].transform("(x->x - x.mean())").compute()
```

### 5.6 Use comma (,) instead of ampersand (&) in filtering

Orca extends pandas' conditional filtering to support commas in filtering statements. In "where" clauses, commas indicate execution order: the next condition will only be tested after the previous condition passes. If can have better performance than using ampersand (&).  

pandas:
```python
df[(df.x > 0) & (df.y < 0)]
```

orca:
```python
df[(df.x > 0), (df.y < 0)]
```

### 5.7 How to optimize certain conditional filtering queries

In certain conditional queries, the filtering conditions on the same orca object are identical on both sides of the equal sign. They refer to the same Python object, which means function `id` on them produces the same result. 

For example:
```python
df[df.x > 0] = df[df.x > 0] + 1
```

The filtering conditions looks identical on both sides of the equal sign in the script above. In Python, however, each time 'df.x > 0' is called, a new object is generated. In this example, an unnecessary object is generated. For better performance, we can use the following script, where the filtering condition is assigned to an intermediate variable. Orca executes an update statement with the intermediate variable. 

```python
df_x_gt_0 = (df.x > 0)
df[df_x_gt_0] = df[df_x_gt_0] + 1
```

## 6 How to implement functionalities not supported by orca

This tutorial explains the differences between orca and pandas together with orca's limitations. If these restrictions cannot be avoided (for examples, an orca function may not support certain parameters; `apply` a complex user-defined function that calls third-party library functions not supported in DolphinDB), then we can convert an orca DataFrame/Series into a pandas DataFrame/Series with function `to_pandas`. After executing in pandas, the results can be converted back into orca objects.

For example, orca does not currently support setting method="average" and na_option="keep" in function `rank`: 
```python
>>> df.rank(method='average', na_option='keep')
ValueError: method must be 'min'
```

If these parameter values are required, we can use the following script:
```
>>> pdf = df.to_pandas()
>>> rank = pdf.rank(method='average', na_option='keep')
>>> rank = orca.DataFrame(rank)
```

Although the script above can get the work done, it causes additional network communication cost. 
