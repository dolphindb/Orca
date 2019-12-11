# Orca入门指南

本文将详细介绍Orca的安装方法、基本操作，以及Orca相对pandas的差异，用户在使用Orca编程时需要注意的细节，以便用户能充分利用DolphinDB的优势，写出高效的Orca代码。

 - [1 安装](#1-安装)    
 - [2 快速入门](#2-快速入门)
 - [3 Orca的架构](#3-orca的架构)
 - [4 Orca的功能限制](#4-orca的功能限制)
 - [5 最佳实践](#5-最佳实践)
 - [6 如果Orca目前无法解决我的问题，我该怎么做？](#6-如果orca目前无法解决我的问题我该怎么做)

## 1 安装

Orca支持Linux和Windows系统，要求Python版本为3.6及以上，pandas版本为0.25.1及以上。

orca项目已经集成到[DolphinDB Python API](https://github.com/dolphindb/Tutorials_CN/blob/master/python_api.md)中。通过pip工具安装DolphinDB Python API，就可以使用orca。

```
pip install dolphindb
```

Orca是基于DolphinDB Python API开发的，因此，你需要有一个DolphinDB服务器，并通过connect函数连接到这个服务器，然后运行Orca：

```python
>>> import dolphindb.orca as orca
>>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)
```

如果你已经有现成的pandas程序，可以将pandas的import替换为：

```python
# import pandas as pd
import dolphindb.orca as pd

pd.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)
```

## 2 快速入门

通过传入一列值创建一个Orca Series对象。Orca会自动为它添加一个默认索引：

```python
>>> s = orca.Series([1, 3, 5, np.nan, 6, 8])
>>> s

0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64
```

通过传入一个字典创建与Orca DataFrame对象。字典中的每个元素必须是能转化为类似Series的对象：

```python
>>> df = orca.DataFrame(
...     {"a": [1, 2, 3, 4, 5, 6],
...      "b": [100, 200, 300, 400, 500, 600],
...      "c": ["one", "two", "three", "four", "five", "six"]},
...      index=[10, 20, 30, 40, 50, 60])
>>> df
    a    b      c
10  1  100    one
20  2  200    two
30  3  300  three
40  4  400   four
50  5  500   five
60  6  600    six
```

也可以直接传入一个pandas DataFrame以创建Orca DataFrame：

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

现在df就是一个orca DataFrame了：

```python
>>> type(df)
<class 'orca.core.frame.DataFrame'>
```

直接打印一个Orca对象时，服务端通常会把对应的整个DolphinDB数据传送到本地，这样做可能会造成不必要的网络开销。用户可以通过`head`函数查看一个Orca对象的顶部数行：

```python
>>> df.head()
                   A         B         C         D
2013-01-01  0.758590 -0.180460 -0.066231  0.259408
2013-01-02  1.165941  0.961164 -0.716258  0.143499
2013-01-03  0.441121 -0.232495 -0.275688  0.516371
2013-01-04  0.281048 -0.782518 -0.683993 -1.474788
2013-01-05 -0.959676  0.860089  0.374714 -0.535574
```

通过`index`, `columns`查看数据的索引、列名：

```python
>>> df.index
DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')

>>> df.columns
Index(['A', 'B', 'C', 'D'], dtype='object')
```

通过`to_pandas`把一个Orca DataFrame转换成pandas DataFrame：

```python
>>> pdf1 = df.to_pandas()
>>> type(pdf1)
<class 'pandas.core.frame.DataFrame'>
```

通过`read_csv`加载一个CSV文件，要求CSV文件位于DolphinDB服务端，所给的路径是它在服务端的路径：

```python
>>> df = orca.read_csv("/home/DolphinDB/Orca/databases/USPrices.csv")
```

## 3 Orca的架构

Orca的顶层是pandas API，底层是DolphinDB数据库，通过DolphinDB Python API实现Orca客户端与DolphinDB服务端的通信。Orca的基本工作原理是，在客户端通过Python生成DolphinDB脚本，将脚本通过DolphinDB Python API发送到DolphinDB服务端解析执行。Orca的DataFrame中只存储对应的DolphinDB的表的元数据，真正的存储和计算都是在服务端。

### Orca如何储存数据

Orca对象在DolphinDB中以一个DolphinDB表的形式储存。无论是Orca DataFrame还是Orca Series，它们的底层存储都是DolphinDB表，数据列和索引列存储在同一个表中。一个Orca DataFrame所表示的DolphinDB表包含若干数据列，以及若干索引列。而一个Orca Series所表示的DolphinDB表包含一列数据列，以及若干索引列。这使得索引对齐、表内各列计算、分组聚合等操作都能较容易地实现。

Orca的DataFrame中只存储对应的DolphinDB的表的元数据，包括表名、数据的列名、索引的列名等。如果尝试访问一个DataFrame的列，返回Series时并不会创建一个新的表。返回的Series和原有的DataFrame使用同一个表，只是Orca对象所记录的元数据产生了变化。

## 4 Orca的功能限制

由于Orca的架构，Orca的接口有部分限制：

### 列的数据类型

DolphinDB的表的每一个列必须指定一种数据类型。DolphinDB的ANY类型不能作为列的数据类型。因此，Orca的每一个列不能包括混合的数据类型。此外，列中的数据也不允许是一个DolphinDB不支持的Python对象，例如Python内置的list, dict，或标准库中的datetime等对象。

某些为这些DolphinDB不支持的类型而设计的函数，例如`DataFrame.explode`，在Orca中就没有实际意义。

### 列名的限制

DolphinDB的表中的列名必须是合法的DolphinDB变量名，即，仅包含字母、数字或下划线，且以字母开头，且不是DolphinDB的保留字，比如if。

DolphinDB不允许重复的列名。因此Orca的列名不能重复。

以大写字母加下划线`ORCA_`开头的列名是Orca的列名保留字，Orca会在内部将某些特殊的列（比如index）以这种形式命名。用户应该避免使用这类字符串作为Orca的列名，否则可能会出现预期之外的行为。

### 分区表没有严格顺序关系

如果DataFrame对应的DolphinDB表是一个分区表，数据存储并非连续，所以就没有RangeIndex的概念。DolphinDB分区表的各分区之间没有严格顺序关系。因此，如果一个DataFrame表示的是一个DolphinDB分区表，这些操作无法完成：

- 对分区表通过iloc访问相应的行
- 将一个不同分区类型的Series或DataFrame赋值给一个DataFrame

### 部分函数仅不支持分布式调用

DolphinDB的某些内置函数目前暂不支持分布式的版本，例如`median`, `quantile`, `mad`。

### 空值机制不同

DolphinDB的数值空值是用每个数据类型的最小值表示。而pandas的空值是用浮点数的nan表示。Orca的空值机制和DolphinDB保持一致，仅当发生网络传输（下载）时，会将DolphinDB包含空值的数值列转化成浮点数类型，将其中的空值转化为nan。

对于字符串类型，pandas的空值依然是nan，这就导致，pandas在储存包含空值的字符串时，实际上是使用字符串和浮点数混合类型。而[混合类型的列在DolphinDB中是不允许的](#列的数据类型)。DolphinDB用空字符串表示字符串类型的空值。用户如果想要上传一个包含空值的字符串，应该对字符串列进行预处理，填充空值：

```python
df = pd.DataFrame({"str_col": ["hello", "world", np.nan]})
odf = orca.DataFrame(df)    # Error
odf = orca.DataFrame(df.fillna({"str_col": ""}))    # Correct way to upload a string column with NULL values
```

### 轴（axis）的限制

DolphinDB作为列式存储的数据库，对逐行（row-wise）操作的支持要好于逐列（column-wise）操作。许多操作，例如求和、求平均值等聚合运算，跨行的聚合（求每一列的函数值）的性能要高于跨列的聚合（求每一行的函数值），大多函数都支持跨行计算，但仅有少量函数，例如`sum`, `mean`, `max`, `min`, `var`, `std`等，支持跨列计算。在pandas中，在函数的参数中指定axis=0或axis='index'就能完成跨行的计算，而指定axis=1或axis='columns'能完成跨列的计算。而Orca函数常常仅支持axis=0或axis='index'。

Orca的DataFrame也不支持`transpose`（转置）操作。因为转置后的DataFrame中的一列就可能包含混合类型的数据。

### 不接受Python可调用对象作为参数

DolphinDB Python API目前无法解析Python函数，因此，例如`DataFrame.apply`, `DataFrame.agg`等函数无法接受一个Python可调用对象作为参数。

对于这个限制，Orca提供了一个备选方案：传入一个DolphinDB字符串，它可以是DolphinDB的内置函数、自定义函数或条件表达式等。详细内容请参考[高阶函数](#高阶函数)一节。

## 5 最佳实践

### 减少`to_pandas`和`from_pandas`的调用

orca使用DolphinDB Python API与服务端通信。实际的数据储存、查询和计算都发生在服务端，orca仅仅是一个提供了类似pandas接口的客户端。因此，系统的瓶颈常常在网络通信上。用户在编写高性能的orca程序时，需要关注如何优化程序，以减少网络通信量。

调用`to_pandas`函数将orca对象转化为pandas对象时，服务端会把整个DolphinDB对象传输到客户端。如果没有必要，一般应该减少这样的转换。此外，以下操作会隐式调用`to_pandas`，因此也需要注意：

- 打印一个表示非分区表的Orca DataFrame或Series
- 调用`to_numpy`或访问`values`
- 调用`Series.unique`, `orca.qcut`等返回numpy.ndarray的函数
- 调用`plot`相关函数画图
- 将Orca对象导出为第三方格式的数据

类似地，`from_pandas`会将本地的pandas对象上传到DolphinDB服务端。当`orca.DataFrame`和`orca.Series`的data参数为非Orca对象时，也会先在本地创建一个pandas对象，然后上传到DolphinDB服务端。在编写Orca代码时，应该考虑减少来回的网络通信。

### Orca并非总是立刻求值

Orca采用了惰性求值策略，某些操作不会立刻在服务端计算，而是转化成一个中间表达式，直到真正需要时才发生计算。需要触发计算时，用户应调用`compute`函数。例如，对同一个DataFrame中的列进行四则运算，不会立刻触发计算：

```python
>>> df = orca.DataFrame({"a": [1, 2, 3], "b": [10, 10, 30]})
>>> c = df["a"] + df["b"]
>>> c    # not calculated yet
<orca.core.operator.ArithExpression object at 0x0000027FA5527B70>

>>> c.compute()    # trigger the calculation
0    11
1    12
2    33
dtype: int64
```

又如，条件过滤查询不会立刻触发计算：

```python
>>> d = df[df["a"] > 2]
>>> d
<orca.core.frame.DataFrame object with a WHERE clause>

>>> d.compute()    # trigger the calculation
   a   b
2  3  30
```

分组后使用`cumsum`等函数聚合，或调用`transform`，也不会立刻返回结果：

```python
>>> c = df.groupby("b").cumsum()
>>> c
<orca.core.operator.DataFrameContextByExpression object at 0x0000017C010692B0>

>>> c.compute()    # trigger the calculation
   a
0  1
1  3
2  3

>>> c = df.groupby("b").transform("count")
>>> c
<orca.core.operator.DataFrameContextByExpression object at 0x0000012C414FE128>

>>> c.compute()    # trigger the calculation
   a
0  2
1  2
2  1
```

#### 操作同一个DataFrame里的列以提高性能

如果操作的是同一个DataFrame里的列，Orca可以将这些操作优化为单个DolphinDB SQL表达式。这样的操作会有较高性能。例如：

- 逐元素计算：`df.x + df.y`, `df * df`, `df.x.abs()`
- 过滤行的操作：`df[df.x > 0]`
- isin操作：`df[df.x.isin([1, 2, 3])]`
- 时间类型/字符串访问器：`df.date.dt.month`
- 用同样长度的计算结果赋值：`df["ret"] = df["ret"].abs()`

当DataFrame是经过过滤的结果时，如果过滤的条件完全相同（在Python中是同一个对象，即调用`id`函数获得的值相同），也能做到这样的优化。

以下脚本可以优化：

```python
df[df.x > 0] = df[df.x > 0] + 1
```

上述脚本中，等号两边的过滤条件虽然看似相同，但在Python中实际产生了两个不同的对象。在DolphinDB引擎中会先执行一个select语句，再执行一个update语句。如果将这个过滤条件赋值给一个中间变量，Orca就可以将上述代码优化为单个DolphinDB的update语句：

```python
df_x_gt_0 = df.x > 0
df[df_x_gt_0] = df[df_x_gt_0] + 1
```

### 避免用numpy函数处理Orca对象

pandas中经常用numpy函数处理一个DataFrame或Series，例如：

```python
>>> ps = pd.Series([1,2,3])
>>> np.log(ps)
0    0.000000
1    0.693147
2    1.098612
dtype: float64
```

应该避免在Orca中使用这种写法。因为numpy函数不识别Orca对象，会将Orca对象当成一个一般的类似数组的对象，对它进行遍历计算。这样会带来大量不必要的网络开销，返回的类型也并非一个Orca对象。在某些情况下，还可能抛出难以理解的异常。用户应该避免调用numpy函数处理Orca对象。

Orca对一些常用的计算函数，例如`log`, `exp`等进行了扩展。对于以上pandas脚本，最佳的Orca改写如下：

```python
>>> os = orca.Series([1,2,3])
>>> os.log()
<orca.core.operator.ArithExpression object at 0x000001FE099585C0>
```

此时，`os.log()`采取了[惰性求值](#orca并非总是立刻求值)策略，返回结果为一个中间表达式，用户可以继续对这个表达式进行四则运算、比较运算、调用数值计算函数，直到真正需要获得计算结果时，才调用`compute`进行计算：

```
>>> os = orca.Series([1,2,3])
>>> tmp = os.log()
>>> tmp += os * 2
>>> tmp = tmp.exp()
>>> tmp
<orca.core.operator.ArithExpression object at 0x000001FE0C374E10>
>>> tmp.compute()
0       7.389056
1     109.196300
2    1210.286380
dtype: float64
```

#### 操作数的顺序

Orca的DataFrame和Series支持与numpy的数值进行四则运算或比较运算，但是有一些限制：numpy的数值必须出现在运算符的右侧：

```python
>>> p = np.exp(1)
>>> type(p)
<class 'numpy.float64'>

>>> s = orca.Series([1,2,3])
>>> p * s         # Could cause potential problems!
>>> s * p         # correct expression

>>> p / s         # Could cause potential problems!
>>> 1 / s * p     # correct expression
```

这是因为，numpy的数值类型重写了四则运算方法。当numpy的数值出现在运算符左侧时，会调用numpy的运算方法。而numpy的实现没有对Orca对象进行特殊处理。这可能会造成潜在的问题。如果numpy的数值出现在运算符右侧，则调用的是Orca的运算方法，能正确识别numpy数值类型。

如果操作数是Python的内置类型，就没有这个限制。

### 修改表数据的限制

在DolphinDB中，一个表的列的数据类型无法修改。

此外，一个非内存表（例如DFS表）有这些限制：

- 无法添加新的列
- 无法通过update语句修改其中的数据

而一个分区表有这些限制：

- 不同分区的数据之间没有严格的顺序关系
- 无法通过update语句将一个向量赋值给一个列

因此，当用户尝试对一个Orca对象进行修改时，操作可能会失败。Orca对象的修改有以下规则：

- 更新的数据类型不兼容，例如将一个字符串赋值给一个整数列时，会抛出异常
- 为一个表示非内存表的orca对象添加列，或修改其中的数据时，会将这个表复制为内存表中，并给出一个警告
- 自动为一个表示分区表的orca对象添加默认索引时，并不会真正添加一个列，此时会给出一个警告
- 为一个表示分区表的orca对象设置或添加一个列时，如果这个列是一个Python或numpy数组，或一个表示内存表的orca Series时，会抛出异常

当尝试给表示非内存表的orca对象添加列，或修改其中数据时，数据会复制为内存表，然后再进行修改。当处理海量数据时，可能导致内存不足。因此应该尽量避免对这类orca对象的修改操作。

Orca部分函数不支持inplace参数。因为inplace涉及到修改数据本身。

例如，以下orca脚本尝试为df添加一个列，会将DFS表复制为内存表，在数据量较大时可能会有性能问题：

```python
df = orca.load_table("dfs://orca", "tb")
df["total"] = df["price"] * df["amount"]     # Will copy the DFS table as an in-memory segmented table!
total_group_by_symbol = df.groupby(["date", "symbol"])["total"].sum()
```

以上脚本可以优化，不设置新的列，以避免大量数据复制。本例采用的优化方法是将分组字段date和symbol通过`set_index`设置为索引，并通过指定`groupby`的level参数，按索引字段进行分组聚合，指定`groupby`的lazy参数为True，不立刻对total进行计算。这样做，能避免添加一个新的列：

```python
df = orca.load_table("dfs://orca", "tb")
df.set_index(["date", "symbol"], inplace=True)
total = df["price"] * df["amount"]     # The DFS table is not copied
total_group_by_symbol = total.groupby(level=[0,1], lazy=True).sum()
```

### 高阶函数

pandas的许多接口，例如`DataFrame.apply`, `GroupBy.filter`等，都允许接受一个Python的可调用对象作为参数。Orca本质上是通过Python API，将用户的程序解析为DolphinDB的脚本进行调用。因此，Orca目前不支持解析Python的可调用对象。如果用户传入一个或多个可调用对象，这些函数会尝试将Orca对象转换为pandas对象，调用pandas的对应接口，然后将结果转换回Orca对象。这样做不仅带来额外的网络通信，也会返回一个新的DataFrame，使得部分计算无法达到在同一个DataFrame上操作时那样的高性能。

作为替代方案，对于这些接口，Orca可以接受一个字符串，将这个字符串传入DolphinDB进行计算。这个字符串可以是一个DolphinDB的内置函数（或内置函数的部分应用），一个DolphinDB的自定义函数，或者一个DolphinDB条件表达式，等等。这个替代方案为Orca带来了灵活性，用户可以按自己的需要，编写一段DolphinDB的脚本片段，然后，像pandas调用用户自定义函数一样，利用DolphinDB计算引擎执行这些脚本。

以下是将pandas接受可调用对象作为参数的代码改写为Orca代码的例子：

#### 求分组加权平均数

pandas:

```python
wavg = lambda df: (df["prc"] * df["vol"]).sum() / df["vol"].sum()
df.groupby("symbol").apply(wavg)
```

Orca:

```python
df.groupby("symbol")["prc"].apply("wavg{,vol}")
```

Orca脚本通过apply函数，对group by之后的prc列调用了一个DolphinDB的部分应用`wavg{,vol}`，转化为DolphinDB的脚本，等价于：

```SQL
select wavg{,vol}(prc) from df group by symbol
```

将这个部分应用展开，等价于：

```SQL
select wavg(prc,vol) from df group by symbol
```

#### 分组后按条件过滤

pandas:

```python
df.groupby("symbol").filter(lambda x: len(x) > 1000)
```

Orca:

```python
df.groupby("symbol").filter("size(*) > 1000")
```

上述例子的Orca脚本中，filter函数接受的字符串是一个过滤的条件表达式，转化为DolphinDB的脚本，等价于：

```SQL
select * from df context by symbol having size(*) > 10000
```

即，filter的字符串出现在了SQL的having语句中。

#### 对整个Series应用一个运算函数

pandas:

```python
s.apply(lambda x: x + 1)
```

Orca:

```python
s.apply("(x->x+1)")
```

pandas:

```python
s.apply(np.log)
```

Orca:

```python
s.apply("log")
```

常用的计算函数，比如`log`, `exp`, `floor`, `ceil`, 三角函数，反三角函数等，Orca已经集成。例如，求对数，通过`s.log()`即可实现。

### 过滤时用逗号(,)代替&符号

DolphinDB的where表达式中，逗号表示执行顺序，并且效率更高，只有在前一个条件通过后才会继续验证下一个条件。Orca对pandas的条件过滤进行了扩展，支持在过滤语句中用逗号：

pandas:

```python
df[(df.x > 0) & (df.y < 0)]
```

Orca:

```python
df[(df.x > 0), (df.y < 0)]
```

使用传统的&符号，会在最后生成DolphinDB脚本时将where表达式中的&符号转换为DolphinDB的`and`函数。而使用逗号，会在where表达式中的对应位置使用逗号，以达到更高的效率。

### 如何实现DolphinDB的context by语句

DolphinDB支持[context by语句](https://www.dolphindb.cn/cn/help/contextby.html)，支持在分组内处理数据。在Orca中，这个功能可以通过`groupby`后调用`transform`实现。而`transform`通常需要用户提供一个DolphinDB自定义函数字符串。Orca对`transform`进行了扩展。对一个中间表达式调用`groupby`，并指定扩展参数lazy=True，然后不给定参数调用`transform`，则Orca会对调用`groupby`的表达式进行context by的计算。例如：

pandas:

```python
df.groupby("date")["prc"].transform(lambda x: x.shift(5))
```

Orca的改写:

```python
df.groupby("date")["id"].transform("shift{,5}")
```

Orca的扩展用法:

```python
df.shift(5).groupby("date", lazy=True)["id"].transform()
```

这是Orca的一个特别的用法，它充分利用了惰性求值的优势。在上述代码中，`df.shift(5)`并没有发生真正的计算，而只是生成了一个中间表达式（通过`type(df.shift(5))`会发现它是一个ArithExpression，而不是DataFrame）。如果指定了`groupyby`的扩展参数lazy=True，`groupby`函数就不会对表达式计算后的结果进行分组。

在[动量交易策略](#)教程中，我们就充分利用了这个扩展功能，来实现DolphinDB的context by。


## 6 如果Orca目前无法解决我的问题，我该怎么做？

本文解释了诸多Orca与pandas的差异，以及Orca的一些限制。如果你无法规避这些限制（比如，Orca的函数不支持某个参数，或者，`apply`一个复杂的自定义函数，其中包括了第三方库函数调用，DolphinDB中没有这些功能），那么，你可以将Orca的DataFrame/Series通过`to_pandas`函数转化为pandas的DataFrame/Series，通过pandas执行计算后，将计算结果转换回Orca对象。

比如，Orca目前不支持`rank`函数的method="average"和na_option="keep"参数，如果你必须使用这些参数，你可以这么做：

```python
>>> df.rank(method='average', na_option='keep')
ValueError: method must be 'min'

>>> pdf = df.to_pandas()
>>> rank = pdf.rank(method='average', na_option='keep')
>>> rank = orca.DataFrame(rank)
```

这样做可以解决你的问题，但它带来了额外的网络通信，同时，新的DataFrame的底层存储的表不再是原先的DataFrame所表示的表，因此无法执行[针对同一个DataFrame操作的一些优化](#操作同一个dataframe里的列以提高性能)。

Orca目前还处于开发阶段，我们今后会为DolphinDB添加更丰富的功能。届时，Orca的接口、支持的参数也会更完善。