# orca入门指南

 - [1 安装](#1-安装)    
 - [2 快速入门](#2-快速入门)
 - [3 架构](#3-架构)
 - [4 功能限制](#4-功能限制)
 - [5 最佳实践](#5-最佳实践)
 - [6 如何实现orca目前不支持的功能](#6-如何实现orca目前不支持的功能)

## 1 安装

orca支持Linux和Windows系统，要求Python版本为3.6及以上，pandas版本为0.25.1及以上。

orca项目集成于[DolphinDB Python API](https://github.com/dolphindb/Tutorials_CN/blob/master/python_api.md)中。通过pip工具安装DolphinDB Python API，就可使用orca。

```
pip install dolphindb
```

orca是基于DolphinDB Python API开发的，因此，你需要打开一个DolphinDB服务器，并通过`connect(host, port, username, password)`函数连接到这个服务器，然后运行orca：

```python
import dolphindb.orca as orca
orca.connect("localhost", 8848, "admin", "123456")
```

若要在orca中运行已有的pandas程序，可将
```python
import pandas as pd
```
改为以下脚本：
```python
import dolphindb.orca as pd
pd.connect("localhost", 8848, "admin", "123456")
```

## 2 快速入门

通过传入一列值创建一个orca Series对象。orca会自动为它添加一个默认索引：
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

可通过传入一个字典创建orca DataFrame对象。字典中的每个元素必须是Series或者可转化为Series的对象：
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

亦可直接传入一个pandas DataFrame以创建orca DataFrame：
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

验证df是一个orca DataFrame：
```
>>> type(df)
<class 'dolphindb.orca.core.frame.DataFrame'>
```

若要查看一个orca对象的起始数行，可使用`head`函数。对一个较大的orca对象，使用`print`命令会使服务端把该对象的所有数据传送到本地，耗时较长，请谨慎使用。

```python
>>> df.head()
                   A         B         C         D
2013-01-01  0.758590 -0.180460 -0.066231  0.259408
2013-01-02  1.165941  0.961164 -0.716258  0.143499
2013-01-03  0.441121 -0.232495 -0.275688  0.516371
2013-01-04  0.281048 -0.782518 -0.683993 -1.474788
2013-01-05 -0.959676  0.860089  0.374714 -0.535574
```

index 属性查看数据的索引：
```python
>>> df.index
DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')
```

columns 属性查看列名：
```
>>> df.columns
Index(['A', 'B', 'C', 'D'], dtype='object')
```

使用`to_pandas`把一个orca DataFrame转换成pandas DataFrame：
```
>>> pdf1 = df.to_pandas()
>>> type(pdf1)
<class 'pandas.core.frame.DataFrame'>
```

`read_csv`用来加载一个位于DolphinDB服务端的CSV文件:
```python
>>> df = orca.read_csv("/home/DolphinDB/Orca/databases/USPrices.csv")
```

## 3 架构

orca的顶层是DolphinDB pandas API，底层是DolphinDB database，通过DolphinDB Python API实现orca客户端与DolphinDB服务端的通信。orca的基本工作原理是，在客户端通过pandas API生成DolphinDB脚本，将脚本通过DolphinDB Python API发送到DolphinDB服务端解析执行。

### orca如何储存数据

orca中的Series与DataFrame在DolphinDB server端均以一个DolphinDB数据表的形式储存，数据列和索引列存储在同一个表中。一个orca Series所对应的DolphinDB数据表包含一个数据列，以及若干索引列。一个orca DataFrame所对应的DolphinDB数据表包含若干数据列，以及若干索引列。这使得索引对齐、表内各列计算、分组聚合等操作都能比较高效地实现。

orca的DataFrame在Python客户端只存储对应的DolphinDB的数据表的元数据，包括表名、数据的列名、索引的列名等，数据的存储和计算都发生在DolphinDB服务端。访问一个DataFrame的某列会返回一个Series。该Series与其所属的的DataFrame对应同一个DolphinDB数据表，只是这两个orca对象所记录的元数据有所不同。

## 4 功能限制

由于orca的架构设计，orca的接口有以下限制：

### 4.1 数据类型

由于DolphinDB的数据表的每一个列必须指定一种数据类型，而且DolphinDB的ANY vector 不能作为数据表的一列，所以orca的DataFrame的一列中不能包括混合的数据类型。此外，一列中的元素仅可为Python内置的int, float, string等标量类型，不可为Python内置的list, dict等对象。某些为这些DolphinDB不支持的类型而设计的函数，例如`DataFrame.explode`，在orca中不支持。

### 4.2 列名

DolphinDB的数据表中的列名必须是合法的DolphinDB变量名，即仅包含字母、数字或下划线，且以字母开头，且不是DolphinDB的保留字，比如if。

DolphinDB不允许重复的列名。因此一个orca DataFrame中的列名不能重复。

以大写字母加下划线 ORCA_ 开头的列名是orca的列名保留字，orca会在内部将某些特殊的列（比如index）以这种形式命名。用户应该避免使用这类字符串作为orca的列名，否则结果可能不符预期。

### 4.3 分区表各分区之间没有顺序关系

由于DolphinDB分区表的各分区之间没有顺序关系，pandas.RangeIndex不适用。因此，如果一个orca DataFrame表示的是一个DolphinDB分区表，下列操作无法完成：

- 对分区表通过iloc访问相应的行
- 将一个不同分区类型的Series或DataFrame赋值给一个DataFrame

### 4.4 部分函数暂不支持分布式调用

DolphinDB的某些内置函数目前暂不支持分布式的版本，例如`median`, `quantile`, `mad`。

### 4.5 空值机制

DolphinDB中的空值是用每个数据类型的最小值表示，而pandas的空值是用浮点数的NaN表示。orca的空值机制与DolphinDB保持一致。

当数据从DolphinDB下载到orca时，系统会自动将包含空值的数值列转化成浮点数类型，并将其中的空值转化为NaN。

以一个简单例子解释。在DolphinDB中生成数据表pt：
```
n=100
dates=take(2018.01.01..2020.01.01, n)
syms =take(`AAPL``XOM, n)
price=rand(1.5 2.6 NULL 3.8 NULL 4.5, n)
volume = rand(100 200 NULL, n)
t=table(dates as date, syms as sym, price, volume)
dbPath="dfs://stocks"
if(existsDatabase(dbPath)){
    dropDatabase(dbPath)
}
db=database(dbPath, RANGE,  2018.01.01 2019.01.01 2020.01.01)
pt=db.createPartitionedTable(t, `pt, `date).append!(t)
```
然后将数据表pt下载到orca：
```python
>>> df=orca.read_table("dfs://stocks","pt") 
>>> df.head()
        date   sym  price  volume
0 2018-01-01  AAPL    NaN     NaN
1 2018-01-02          3.8     NaN
2 2018-01-03   XOM    NaN   200.0
3 2018-01-04  AAPL    3.8   100.0
4 2018-01-05          NaN   200.0
```
可见，因含有空值，volume列被自动转化为浮点类型。请注意，DolphinDB数据表中sym列（字符串列）中的空值现在被转化为空字符串""，而非NaN。若用户需要区分空字符串与NaN，则需进行如下操作：



由于pandas字符串类型的空值依然是NaN，所以pandas中包含空值的字符串列为字符串和浮点数混合类型。由于[DolphinDB中不允许混合类型的列](#数据类型)，所以从pandas上传到DolphinDB一个包含空值的字符串列之前，必须将其中的NaN改为Python的空字符串("")：
```python
df = pd.DataFrame({"str_col": ["hello", "world", np.nan]})
odf = orca.DataFrame(df.fillna({"str_col": ""}))
```

### 4.6 跨列计算

DolphinDB作为列式存储的数据库，对跨行（row-wise）计算的支持要好于跨列（column-wise）计算。pandas中的函数的参数中指定axis=0或axis='index'为跨行的计算，指定axis=1或axis='columns'为跨列的计算。orca函数大多数仅支持跨行计算(axis=0或axis='index')。仅有少量函数，例如`sum`, `mean`, `max`, `min`, `var`, `std`等，支持跨列计算。对求和、求平均值等聚合运算，跨行的聚合（求一列的函数值）的性能要高于跨列的聚合（求一行的函数值）。

orca的DataFrame也不支持`transpose`（转置）操作，因为转置后的DataFrame中的一列可能包含混合类型的数据。

### 4.7 不接受Python可调用对象作为参数

DolphinDB Python API目前无法解析Python函数，因此，pandas中与高阶函数有关的API，例如`DataFrame.apply`, 无法接受一个Python可调用对象(callable)作为参数。

针对这个限制，orca目前的解决方案为：传入一个用Python字符串表示的DolphinDB脚本，它可以是DolphinDB的内置函数、自定义函数或条件表达式等。详细内容请参考[高阶函数](#高阶函数)一节。

## 5 最佳实践

### 5.1 减少`to_pandas`和`from_pandas`的调用

orca使用DolphinDB Python API与服务端通信。数据储存、查询和计算都发生在服务端，orca仅仅是一个提供了类似pandas接口的客户端。因此，系统的瓶颈常常在网络通信上。用户在编写高性能的orca程序时，需要关注如何优化程序，以减少网络通信量。

调用`to_pandas`函数将orca对象转化为pandas对象时，服务端会把整个DolphinDB对象传输到客户端。如果没有必要，一般应该减少这样的转换。

此外，以下操作会隐式调用`to_pandas`，因此也需要注意：

- 打印一个表示非分区表的orca DataFrame或Series
- 调用`to_numpy`或`values`
- 调用`Series.unique`或`orca.qcut`等返回numpy.ndarray的函数
- 调用`plot`相关函数画图
- 将orca对象导出为第三方格式的数据

类似地，`from_pandas`会将本地的pandas对象上传到DolphinDB服务端。当`orca.DataFrame`和`orca.Series`的data参数为非orca对象时，也会先在本地创建一个pandas对象，然后上传到DolphinDB服务端。在编写orca代码时，应该考虑减少来回的网络通信。

### 5.2 惰性求值

orca采用了惰性求值策略，本章所列举的操作，若涉及一个orca对象，不会立刻发生计算，而是转化成一个中间表达式。需要触发计算时，调用`compute`函数。

请注意，涉及多个不同orca对象的运算，不会采用惰性求值策略。

#### 5.2.1 四则运算、逻辑运算或使用非聚合函数

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
```python
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
请注意，所有聚合函数都不采用惰性求值策略。

#### 5.2.2 条件过滤查询

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

### 5.3 避免使用NumPy函数处理orca对象

应避免使用NumPy函数处理orca对象。对orca对象，尽量使用orca内置函数或[DolphinDB NumPy函数](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/dolphindb_numpy.md)。

pandas中经常用NumPy函数处理一个DataFrame或Series，例如：

```python
>>> ps = pd.Series([1,2,3])
>>> np.log(ps)
0    0.000000
1    0.693147
2    1.098612
dtype: float64
```

若在orca中使用这种写法，由于NumPy函数不识别orca对象，会将orca对象当成一个一般的类似数组(array-like)的对象，对它进行遍历计算。这样会带来大量不必要的网络开销（networking overhead），返回的类型也并非一个orca对象。在某些情况下，还可能抛出难以理解的异常。

orca提供了一些常用的计算函数，例如`log`, `exp`等。对于以上pandas脚本，可用orca改写如下：

```python
>>> os = orca.Series([1,2,3])
>>> os.log()
<orca.core.operator.ArithExpression object at 0x000001FE099585C0>
```

`os.log()`采取了[惰性求值](#惰性求值)策略：
```python
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

### 5.4 修改数据表的限制

在orca中，一个表的列的数据类型无法修改。

修改一个非内存表（例如DFS表）有这些限制：
- 无法添加新的列
- 无法通过`update`函数修改其中的数据

修改一个分区表有这些限制：
- 无法通过`update`函数将一个向量赋值给一个列

当用户尝试对一个orca对象进行修改时，操作失败的可能的原因有以下几条：
- 更新的数据类型不兼容，例如不可将一个字符串赋值给一个整数列。
- 在创建一个表示分区表的orca DataFrame时，若没有对其添加索引，orca也无法自动为其添加默认索引，因为对分区表无法添加列。此时会给出一个警告。
- 为一个表示分区表的orca DataFrame更新或添加一个列时，这个列（新值）不允许是Python或NumPy数组，或一个表示内存表的orca Series，仅可为基于该对象内部数据进行计算得到的结果。

当尝试给表示非内存表的orca对象添加列，或修改其中数据时，数据会加载为内存表，然后再进行修改。当处理海量数据时，可能导致内存不足。因此应该尽量避免对这类orca对象的修改操作。

例如，为计算一个DFS数据表中每组内两列乘积之和，以下脚本为orca对象df添加一个新列total，为price与amount两列的乘积。这个操作会将DFS表加载为内存表。

```python
df = orca.read_table("dfs://orca", "tb")
df["total"] = df["price"] * df["amount"]     # Will load the DFS table as an in-memory segmented table!
total_group_by_symbol = df.groupby(["date", "symbol"])["total"].sum()
```

使用以下步骤，可以避免创建新的列，以避免不必要的数据复制。
- 1. 将分组字段date和symbol通过`set_index`设置为索引，然后通过指定`groupby`的level参数，按索引字段进行分组聚合。
- 2. 指定`groupby`的lazy参数为True，不立刻对total进行计算（乘法）。
```python
df = orca.read_table("dfs://orca", "tb")
df.set_index(["date", "symbol"], inplace=True)  # happens on the client side, not on the server side
total = df["price"] * df["amount"]     # The DFS table is not loaded into memory. Calculation has not happened yet. 
total_group_by_symbol = total.groupby(level=[0,1], lazy=True).sum()
```

此外，由于分布表无法被修改，所以orca的某些函数不支持inplace参数，因为这会导致修改输入数据。


### 5.5 高阶函数

pandas的许多接口，例如`DataFrame.apply`, `GroupBy.filter`等，都允许接受一个Python的可调用对象(callable)作为参数。orca通过Python API，将用户的程序解析为DolphinDB的脚本进行调用。因此，orca目前不支持解析Python的可调用对象。如果用户传入Python的可调用对象，这些函数会尝试将orca对象转换为pandas对象，调用pandas的对应接口，然后将结果转换回orca对象。这样做不仅导致额外的网络通信，也会返回一个新的orca对象，使得部分计算无法达到在同一个DataFrame上操作时那样的高性能。

作为替代方案，对于这些接口，orca可以接受一个字符串，将这个字符串传入DolphinDB以执行。这个字符串可以是一个DolphinDB的内置函数（或内置函数的部分应用），一个DolphinDB的自定义函数，或者一个DolphinDB条件表达式，等等。

以下是将pandas接受可调用对象作为参数的代码改写为orca代码的例子：

#### 5.5.1 求分组加权平均数

pandas:
```python
wavg = lambda df: (df["prc"] * df["vol"]).sum() / df["vol"].sum()
df.groupby("symbol").apply(wavg)
```

orca:
```python
df.groupby("symbol")["prc"].apply("wavg{,vol}")
```

orca脚本通过`apply`函数，对group by之后的prc列调用了一个DolphinDB的部分应用`wavg{,vol}`。等价于以下DolphinDB脚本：
```SQL
select wavg{,vol}(prc) from df group by symbol
```

更易于理解的版本：
```SQL
select wavg(prc,vol) from df group by symbol
```

#### 5.5.2 分组后按条件过滤

pandas:
```python
df.groupby("symbol").filter(lambda x: len(x) > 1000)
```

orca:
```python
df.groupby("symbol").filter("count(*) > 1000")
```

上述例子的orca脚本中，`filter`函数接受的字符串是一个过滤的条件表达式。转化为DolphinDB的脚本，等价于：
```SQL
select * from df context by symbol having count(*) > 10000
```

#### 5.5.3 对整个Series应用同一个运算函数

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

orca提供常用的计算函数包括`log`, `exp`, `floor`, `ceil`, 三角函数，反三角函数等。

#### 5.5.4 实现DolphinDB的context by语句

pandas中，`groupby`后调用`shift`, `cumsum`, `bfill`等函数，可以实现DolphinDB中的[context by语句](https://www.dolphindb.cn/cn/help/contextby.html)与`move`, `cumsum`, `bfill`等函数组合使用的功能。orca的写法和pandas相同：

pandas:
```
df.groupby("symbol")["prc"].shift(1)
```
orca:
```
df.groupby("symbol")["prc"].shift(1).compute()
```

请注意lambda函数在pandas与orca中的不同写法：

pandas:
```python
df.groupby("symbol")["prc"].transform(lambda x: x - x.mean())
```
orca中可改写为：
```python
df.groupby("symbol")["prc"].transform("(x->x - x.mean())").compute()
```

### 5.6 过滤时用逗号(,)代替&符号

orca对pandas的条件过滤进行了扩展，支持在过滤语句中用逗号。逗号表示执行顺序，并且效率比使用 & 更高，只有在前一个条件通过后才会继续验证下一个条件。

pandas:
```python
df[(df.x > 0) & (df.y < 0)]
```

orca:
```python
df[(df.x > 0), (df.y < 0)]
```

#### 5.7 某些过滤条件的优化写法

当运算涉及到条件过滤时，有时等号双方针对同一个orca对象的过滤条件**完全**相同（在Python中是同一个对象，即调用`id`函数获得的值相同）。例如：

```python
df[df.x > 0] = df[df.x > 0] + 1
```

上述脚本中，等号两边的过滤条件虽然看似相同，但在Python中，每一个`df.x > 0`的调用都产生了新的对象，这会导致orca中产生一个不必要的对象，影响性能。可使用以下脚本进行优化，将这个过滤条件赋值给一个中间变量。orca会执行一个update语句。

```python
df_x_gt_0 = (df.x > 0)
df[df_x_gt_0] = df[df_x_gt_0] + 1
```

## 6 如何实现orca尚不支持的功能

本文解释了诸多orca与pandas的差异，以及orca的一些限制。如果无法规避这些限制（例如，orca的函数不支持某些参数，或者，`apply`一个复杂的自定义函数，其中包括了第三方库函数调用而DolphinDB中尚未支持这些功能），可使用`to_pandas`函数将orca的DataFrame/Series转化为pandas的DataFrame/Series，在pandas中运行后，将计算结果转换回orca对象。

例如，orca目前不支持`rank`函数的method="average"和na_option="keep"参数：
```python
>>> df.rank(method='average', na_option='keep')
ValueError: method must be 'min'
```

如果必须使用这些参数，可以使用以下脚本：
```
>>> pdf = df.to_pandas()
>>> rank = pdf.rank(method='average', na_option='keep')
>>> rank = orca.DataFrame(rank)
```

以上脚本可以解决问题，但导致了额外的网络通信。

orca目前还处于开发阶段，我们今后会为DolphinDB添加更丰富的功能。届时，orca的接口、支持的参数也会更完善。