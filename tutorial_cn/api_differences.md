# orca与pandas的差异

本文介绍orca与pandas在以下几个方面的差异：

- [1 数据类型](#1-数据类型)
- [2 通用函数](#2-通用函数)
- [3 Input/output](#3-inputoutput)
- [4 Series/DataFrame](#4-seriesdataframe)
  - [4.1 Series和DataFrame的创建与修改](#41-series和dataframe的创建与修改)
  - [4.2 Series和DataFrame的四则运算](#42-series和dataframe的四则运算)
  - [4.3 Series和DataFrame的属性和方法](#43-series和dataframe的属性和方法)
- [5 Index objects](#5-index-objcts)
- [6 groupby](#6-groupby)
- [7 Resampling](#7-resampling)
- [8 orca分区表](#8-orca分区表)

### 1 数据类型

#### 1.1 整数类型的差异

DolphinDB支持CHAR, SHORT, INT, LONG等不同字节数的整数类型，整数字面量默认解析为32位INT类型，而pandas的整数字面量默认解析为64位。

#### 1.2 字符串类型的差异

- 字符串类型的标量

DolphinDB支持STRING和SYMBOL两种字符串类型，STRING类型可以进行max, min等比较运算，但SYMBOL不允许。pandas中的字符串类型的底层存储是np.object。pandas也提供了经过优化的category类型，使用整数作为底层存储，对于取值范围有限的数据，能减少内存占用。DolphinDB的SYMBOL类型能实现类似的功能。但pandas允许将任何数据类型的数据转换为category类型，而DolphinDB只允许将字符串类型转换为SYMBOL类型。

- 字符串类型的序列（Series）
  
DolphinDB要求数据表的一列中元素的数据类型必须相同，而pandas允许一个Series中的数据有不同数据类型。DolphinDB中字符串类型的NULL值是一个空字符串，而pandas中字符串类型的空值是np.NaN。

#### 1.3 日期和时间类型的差异

- 不同精度的时间单位

  DolphinDB支持多种不同单位的日期和时间类型，包括DATE, MINUTE, SECOND, TIME, NANOTIME等，而[pandas的所有日期和时间数据](https://pandas.pydata.org/pandas-docs/version/0.25.3/user_guide/timeseries.html#dateoffset-objects)均为np.datetime64\[ns\]类型，通过freq参数来区分不同精度的时间单位。

- 带有时区的时间类型暂不支持

  如下例所示，若一个pandas的Series带有时区，转化为orca的Series之后时区信息将会丢失。

  ```python
  >>> ps = pd.Series([pd.Timestamp('2016-01-01', tz='US/Eastern') for _ in range(3)])
  >>> os = orca.Series(ps)
  >>> ps
  0   2016-01-01 00:00:00-05:00
  1   2016-01-01 00:00:00-05:00
  2   2016-01-01 00:00:00-05:00
  dtype: datetime64[ns, US/Eastern]

  >>> os
  0   2016-01-01 05:00:00
  1   2016-01-01 05:00:00
  2   2016-01-01 05:00:00
  dtype: datetime64[ns]
  ```
  
#### 1.4 category类型

orca暂不支持category类型的数据。将一个category类型的pandas Series转换为orca的Series之后，数据类型变为object。

```python
>>> ps = pd.Series(pd.Categorical(list('baabc'), categories=list('abc'), ordered=True))
>>> os = orca.Series(ps)
>>> ps
0    b
1    a
2    a
3    b
4    c
dtype: category
Categories (3, object): [a < b < c]

>>> os
0    b
1    a
2    a
3    b
4    c
dtype: object
```

### 2 通用函数

orca提供以下通用函数：

  |函数|描述|
  |:---|:---|
  |connect|将会话连接到DolphinDB服务器|
  |merge|连接两个DataFrame|
  |concat|按照columns对齐连接两个DataFrame|
  |date_range|创建时间序列|
  |to_datetime|将字符串转换成时间类型|
  |isna|判断是否是空值|
  |isnull|判断是否是空值|
  |notna|判断是否不是空值|
  |notnull|判断是否不是空值|
  |save_table|保存orca的内存表|
  |read_shared_table|读取共享表|

下面对`connect`函数和`save_table`函数加以说明：

- `connect`函数

  使用orca之前，必须调用`connect`函数建立一个与DolphinDB服务器的连接。
  ```python
  import dolphindb.orca as orca
  orca.connect("localhost", 8848, "admin", "123456")     # orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)
  ```

- `save_table`函数

  该函数的用法请参见[orca教程：数据写入](https://github.com/dolphindb/orca/blob/master/tutorial_cn/saving_data.md)。

### 3 Input/output函数

orca提供以下input/output函数：

  |函数|描述|
  |:---|:---|
  |read_csv|导入数据|
  |read_table|读取DolphinDB的磁盘表、磁盘分区表和分布式表|
  |read_shared_table|读取DolphinDB的共享表|
  
- `read_csv`函数

  - engine参数
    
    pandas的`read_csv`函数的engine参数的取值可以是'c'或者'python'，表示使用哪一种引擎进行导入。
      
    而orca的`read_csv`函数的engine参数的取值可以是'c', 'python', 或 'dolphindb'，默认值为'dolphindb'。取值为'dolphindb'时，`read_csv`函数会在DolphinDB服务器目录下寻找要导入的数据文件；取值为'python'或'c'时，`read_csv`函数会在python客户端的目录下寻找要导入的数据文件。

    > 请注意，当engine参数设置为'python'或者'c'时，orca的`read_csv`函数调用了pandas的`read_csv`函数进行导入。下面列出的差异均是在engine参数设置为'dolphindb'时的差异。

    当engine参数设置为'dolphindb'时，orca的`read_csv`函数的语法如下：
    ```python
    read_csv(path, sep=',', delimiter=None, names=None,  index_col=None,engine='dolphindb', usecols=None, squeeze=False, prefix=None, dtype=None, partitioned=True, db_handle=None, table_name=None, partition_columns=None, *args, **kwargs)
    ```
    
  - dtype参数

    在pandas中，`read_csv`函数支持dtype参数，该参数接收一个字典，键是列名，值是Python原生类型（bool, int, float, str）或NumPy的dtype（np.bool, np.int8, np.float32, 等等）。
      
    例如：
    ```python
    dfcsv = pd.read_csv("path_to/allTypesOfColumns.csv", dtype={"tbool": np.bool, "tchar": np.int8, "tshort": np.int16, "tint": np.int32, "tlong": np.int64, "tfloat": np.float32, "tdouble": np.float64})
    ```
      
    与pandas不同的是，orca的`read_csv`函数的dtype参数还支持以字符串的方式指定DolphinDB的提供的所有[数据类型](https://www.dolphindb.cn/cn/help/DataType.html)，包括所有时间类型和字符串类型。
      
    例如：
    ```python
    dfcsv = orca.read_csv("path_to/allTypesOfColumns.csv", dtype={"tstring":'STRING', "tsymbol": 'SYMBOL', "date": 'DATE', "second": 'SECOND'， "tint": np.int32})
    ```
    
  - sep与delimiter参数
    
    pandas的这两个参数支持对正则表达式的解析，而orca目前尚不支持这一点。
    
  - partitioned参数 

    bool类型，该参数为True时，表示允许分区方式将数据导入(调用DolphinDB的[`ploadText`函数](https://www.dolphindb.cn/cn/help/ploadText.html))；当该参数为False时，表示强制以非分区的方式导入数据(调用DolphinDB的[`loadText`函数](https://www.dolphindb.cn/cn/help/loadText.html))。

    > 请注意：orca的分区表与orca的内存表相比，在操作时也存在许多差异，具体见[orca分区表](#8-orca分区表)。若数据量不大，且对orca与pandas的一致性要求较高，则尽量不要将数据以分区的方式导入。若数据量很大，对性能要求也很高，则建议采用分区方式导入数据。
      
  - db_handle, table_name以及partition_columns参数

    orca的`read_csv`还支持db_handle, table_name和partition_columns这3个参数，用于在导入数据时通过指定数据库，数据表，分区列等相关信息，将数据导入到DolphinDB的分区表。关于这几个参数的具体用法与示例请参见[orca分区表](#8-orca分区表)。

- `read_table`函数

  在pandas中，`read_table`函数用于导入文本文件。在orca中，`read_table`函数仅用于导入一个[DolphinDB的分区表](https://github.com/dolphindb/Tutorials_CN/blob/master/database.md)。关于该函数的具体用法与示例请参见[orca分区表](#8-orca分区表)。

- `read_shared_table`函数

  用于读取共享表（内存表或流表），返回一个orca的DataFrame。关于该函数的具体用法与示例请参见[orca教程：数据加载](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/data_import.md)。

### 4 Series与DataFrame相关操作函数

#### 4.1 Series/DataFrame的创建与修改

- Series/DataFrame的创建

  pandas中定义一个Series时，可以不设置name参数，或者使用数字作为name。这种做法，在orca中相当于在DolphinDB server端新建一个只含有一列的表，且不设置列名或使用数字作为列名。由于DolphinDB中，表的列名不允许为空值且不可全部为数字，因此这些情况下，系统会为该Series自动生成一个用户不可见的列名作为name。orca会抛出WARNING信息，例如，orca中创建一个name为0的series时，会抛出WARNING：

  ```python
  >>> a = orca.Series([1, 2, 3, 4], name='0')
  C:\ProgramData\Anaconda3\lib\site-packages\dolphindb\orca\core\common.py:33: NotDolphinDBIdentifierWarning: The DataFrame contains an invalid column name for DolphinDB. It will be converted to an automatically generated column name.
  "generated column name.", NotDolphinDBIdentifierWarning)

  >>> a
  0    1
  1    2
  2    3
  3    4
  Name: 0, dtype: int64
  ```
- 向Series或DataFrame中追加数据
    
  pandas允许通过访问index中不存在的值而增加新的行，但是orca暂不支持这种操作。

  ```python
  >>> ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
  >>> os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

  >>> ps['e']=1
  >>> ps
  a    10
  b     1
  c    19
  d    -5
  e     1
  dtype: int64

  >>> os['e']=1
  >>> os 
  a    10
  b     1
  c    19
  d    -5
  dtype: int64
  ```

#### 4.2 Series和DataFrame的四则运算

- 惰性求值

  多数只涉及一个orca对象的运算，会采用惰性求值策略，不会立刻发生计算，而是转化成一个中间表达式。需要触发计算时，调用`compute`函数。涉及多个不同orca对象的运算，不会采用惰性求值策略。
  
  关于惰性求值的更多细节，请参考[orca入门指南](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/user_guide.md)中的5.2小节。

- Null值

  在Python与pandas中，任何数与Null值比较，都返回False。而orca则将Null值视为每个数据类型中的最小值。
 
  下例中，分别对pandas和orca的Series进行条件过滤。orca中，将NaN<1的结果视为True。

  ```python
  >>> ps = pd.Series([1,np.nan,0])
  >>> os = orca.Series([1,np.nan,0])
  >>> ps[ps<1]
  2    0.0
  dtype: float64

  >>> os[os<1].compute()
  1    NaN
  2    0.0
  dtype: float64
  ```

- 空字符串

  pandas的字符串类型中，NaN值与空字符串不同。orca的字符串类型中，空字符串被视为NaN值。

  ```python
  >>> ps = pd.Series(["","s","a"])
  >>> os = orca.Series(["","s","a"])

  >>> ps.hasnans
  False

  >>> os.hasnans
  True
  ```

- 零作为分母

  pandas中，非零数除以零的结果为同符号的无穷大；零除以零的结果为NaN。orca中，任何数除以零的结果均为NaN。

  ```python
  >>> ps=pd.Series([1,0,2])
  >>> os=orca.Series([1,0,2])
  >>> ps.div(0)
  0    inf
  1    NaN
  2    inf
  dtype: float64

  >>> os.div(0).compute()
  0   NaN
  1   NaN
  2   NaN
  dtype: float64
  ```

#### 4.3 Series和DataFrame的属性和方法

本小节根据pandas官方提供的[Series](https://pandas.pydata.org/pandas-docs/stable/reference/series.html)和[DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#dataframe)的文档，依次介绍orca与pandas的不同之处。用户也可从中了解orca中Series和DataFrame支持的属性。

#### 4.3.1 属性函数

除了pandas已经取缔的属性之外，orca的Series和DataFrame唯一不支持的属性是memory_usage。

#### 4.3.2 转化函数

转化函数中目前仅支持`Series.to_numpy`这一个函数。

#### 4.3.3 索引与迭代相关函数

以下函数可用于orca中的Series与DataFrame对象：

  |函数|描述|
  |:---|:---|
  |head|返回前n个值|
  |tail|返回最后n个值|
  |loc|通过index访问|
  |iloc|通过下标访问|
  |where|填充不符合过滤条件的值|
  |mask|填充符合过滤条件的值|

orca中的DataFrame对象还支持以下函数：

  |函数|描述|
  |:---|:---|
  |items|遍历DataFrame|
  |iteritems|遍历DataFrame|
  |lookup|根据标签查询数据|
  |get|访问某一列|

下面对`loc`与`iloc`做具体说明。

- 通过`loc`访问Series和DataFrame

  如下所示，orca暂不支持使用`loc`访问带有DatetimeIndex的Series和DataFrame。
  ```python
  >>> pdd = pd.DataFrame({'id': [1, 2, 2, 3, 3], 
                         'sym': ['s', 'a', 's', 'a', 's'], 
                      'values': [np.nan, 2, 2, np.nan, 2]},
                      index=pd.date_range('20190101', '20190105', 5))
  >>> odd = orca.DataFrame(pdd)
  >>> pdd
              id sym  values
  2019-01-01   1   s     NaN
  2019-01-02   2   a     2.0
  2019-01-03   2   s     2.0
  2019-01-04   3   a     NaN
  2019-01-05   3   s     2.0
    
  >>> pdd.loc["20190103":"20190105"]
              id sym  values
  2019-01-03   2   s     2.0
  2019-01-04   3   a     NaN
  2019-01-05   3   s     2.0
    
  >>> odd.loc["20190103":"20190105"]
  RuntimeError
  ```
  
  当DataFrame的index中有重复的值时，pandas不支持以重复的index值作为范围的边界，而orca则将重复的index值第一次出现的位置作为范围的边界。

  ```python
  >>> pdf = pd.DataFrame([[1, 2, 1], [4, 5, 5], [7, 8, 7], [1, 5, 8], [7, 5, 1]], index=[7, 8, 2, 8, 9], columns=['max_speed', 'shield', 'size'])
  >>> odf = orca.DataFrame([[1, 2, 1], [4, 5, 5], [7, 8, 7], [1, 5, 8], [7, 5, 1]], index=[7, 8, 2, 8, 9], columns=['max_speed', 'shield', 'size'])
  >>> pdf
    max_speed  shield  size
  7          1       2     1
  8          4       5     5
  2          7       8     7
  8          1       5     8
  9          7       5     1
  
  >>> pdf.loc[8:]
  KeyError: 'Cannot get left slice bound for non-unique label: 8'
  
  >>> pdf.loc[:8]
  KeyError: 'Cannot get right slice bound for non-unique label: 8'

  >>> odf.loc[8:]
    max_speed  shield  size
  8          4       5     5
  2          7       8     7
  8          1       5     8
  9          7       5     1
  
  >>> odf.loc[:8]
      max_speed  shield  size
  7          1       2     1
  8          4       5     5
  ```
  
- 修改Series和DataFrame中列的数据类型

  在pandas中，使用`loc`和`iloc`更改Series或DataFrame中某个值的数据类型，就会更改该值所在之列的类型。也可使用`astype`函数更改整列的类型。而orca中不允许修改列的类型。

  ```python
  >>> ps = pd.Series([10, 20, 30])
  >>> ps
  0    10
  1    20
  2    30
  dtype: int64

  >>> os = orca.Series([10, 20, 30])
  >>> os
  0    10
  1    20
  2    30
  dtype: int64

  >>> ps.loc[0]=100.5
  >>> ps
  0    100.5
  1     20.0
  2     30.0
  dtype: float64

  >>> os.loc[0]=100.5
  >>> os
  0    101
  1     20
  2     30
  dtype: int64
  ```

  上例试图将Series的一个元素的值由int64改为float64类型。在pandas中，整个Series的数据类型都被改为float64。在orca中，由于Series的数据类型无法更改，所以系统强制将该元素的新值转换为int64类型。通过`loc`或`iloc`对Series和DataFrame进行类似的修改操作都会有类似的差异。

- 使用`loc`向DataFrame新增一行或者一列

  pandas支持使用`loc`访问不存在的index值或者columns，以新增行或者列。而orca暂不支持。

  ```python
  >>> ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
  >>> os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
  >>> ps
  a    10
  b     1
  c    19
  d    -5
  dtype: int64

  >>> ps.loc['e']=1
  >>> ps
  a    10
  b     1
  c    19
  d    -5
  e     1
  dtype: int64

  >>> os
  a    10
  b     1
  c    19
  d    -5
  dtype: int64

  >>> os['e']=1
  >>> os
  a    10
  b     1
  c    19
  d    -5
  dtype: int64
  ```

- `loc`和`iloc`暂不支持MultiIndex
 
#### 4.3.4 二元运算函数 (binary operator functions)

除了[combine](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.combine.html#pandas.DataFrame.combine)和[combine_first](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.combine_first.html#pandas.DataFrame.combine_first)函数之外，orca支持pandas提供的所有[二元函数](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#binary-operator-functions)。但是，orca的DataFrame或者Series在进行四则运算时，除了本文第2.2小节所提及的差异之外，还存在其他一些差异。

- 二元运算函数的axis参数

  orca的DataFrame与Series进行二元运算时，不支持aixs参数取值为1或'columns'，只能取值为0或'index'。
    
  pandas中，可使用`add`函数将一个Series与一个DataFrame的每行相加：
  ```python
  >>> pdf = pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
  >>> pdf.add(pd.Series([1, 2], index=["angles","degrees"]))
            angles  degrees
  circle         1      362
  triangle       4      182
  rectangle      5      362
  ```

  亦可将一个Series与一个DataFrame的每列相加：
  ```python
  >>> pdf.add(pd.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle']), axis='index')
            angles  degrees
  circle         1      361
  triangle       5      182
  rectangle      7      363
  ```

  orca仅支持将Series与DataFrame的每列相加(axis=0或'index'):
  ```python
  >>> odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
  >>> odf.add(orca.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle']), axis='index')
            angles  degrees
  circle         1      361
  triangle       5      182
  rectangle      7      363
  ```

  请注意，在orca中，只有涉及一个DataFrame与一个Series的二元运算不支持axis='columns'，其他情况支持axis='columns'，例如一个DataFrame与一个list相加：
  ```python
  >>> odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
  >>> (odf + [1, 2]).compute()
            angles  degrees
  circle         1      362
  triangle       4      182
  rectangle      5      362
  ```
  请注意，上例只涉及一个orca对象odf，所以采用了惰性求值。之前的例子中涉及了两个orca对象，所以没有采用惰性求值。


- 除数是负数

  pandas中负数在除法中作为除数是有意义的；在orca中，如果负数在除法中作为除数，返回的结果将是NaN。
    
  如下例所示，在求余运算中，除数中负数在结果中的对应位置值为NaN。
  ```python
  >>> pd.Series([1, 2, 12, 10]) % [10, 1, 19, -4]
  a     1
  b     0
  c    12
  d    -2
  dtype: int64

  >>> (orca.Series([1, 2, 12, 10]) % [10, 1, 19, -4]).compute()
  a     1.0
  b     0.0
  c    12.0
  d     NaN
  dtype: float64
  ```
  请注意，此时由于NaN为float类型，而DolphinDB中同一列的数据类型必须一致，所以orca自动将该列的类型转换为float。若这里的-4改为正整数，则结果的类型为int。

- 求余运算

  pandas支持对浮点数的求余运算，而orca暂不支持。
  ```python   
  >>> pd.Series([5.5, 10, -4.5, 2.5, np.nan]) % pd.Series([2.5, -4.5, 2.5, np.nan, 3])
  0    0.5
  1   -3.5
  2    0.5
  3    NaN
  4    NaN
  dtype: float64

  >>> orca.Series([5.5, 10, -4.5, 2.5, np.nan]) % orca.Series([2.5, -4.5, 2.5, np.nan, 3])
  RuntimeError: <Server Exception> in run: Arguments for mod (%) cannot be string or floating number.
  ```

#### 4.3.5 应用函数、分组运算与窗口函数

以下函数可用于orca中的Series/DataFrame对象：

  |函数|描述|
  |:---|:---|
  |apply|应用函数|
  |agg|应用聚合函数|
  |aggregate|应用聚合函数|
  |groupby|分组运算|
  |rolling|滑动窗口|
  |ewm|指数加成滑动|

- `apply`, `agg`与`aggregate`函数

  orca的这三个函数目前仅支持字符串或者一个字典，不支持lambda函数。

  例如，pandas中的以下代码    
  ```python
  >>> ps=pd.Series([1, 2, 12, 10])
  >>> ps.apply(lambda x: x + 1)
  ```
    
  在orca中，应改写为：
  ```python
  >>> os=orca.Series([1, 2, 12, 10])
  >>> os.apply("(x->x+1)").compute()
  ```

  关于这三个函数的更多细节，请参见[orca使用教程高阶函数部分](https://github.com/dolphindb/orca/blob/master/tutorial_cn/user_guide.md#%E9%AB%98%E9%98%B6%E5%87%BD%E6%95%B0)。

- `groupby`函数
  
  orca的`groupby`函数的语法如下：
  ```python
  DataFrame.groupby(self, by=None, level=None, as_index=True, sort=True, squeeze=False, ascending=True, **kwargs)
  ```
  更多细节请参见[第6节](#6-groupby)。

- `rolling`函数

  orca的`rolling`函数支持window和on参数。若遇到空值，pandas在对应位置返回NaN，而orca忽略空值，仍旧进行计算。
  ```python
  >>> pdf = pd.DataFrame({'id': np.arange(1, 6, 1), 'B': [0, 1, 2, np.nan, 4]})
  >>> pdf
     id    B
  0   1  0.0
  1   2  1.0
  2   3  2.0
  3   4  NaN
  4   5  4.0

  >>> pdf.rolling(2, on="id").sum()
     id    B
  # output
  0   1  NaN
  1   2  1.0
  2   3  3.0
  3   4  NaN
  4   5  NaN

  >>> odf = orca.DataFrame({'id': np.arange(1, 6, 1), 'B': [0, 1, 2, np.nan, 4]})
  >>> odf.rolling(2, on="id").sum()
     id    B
  0   1  NaN
  1   2  1.0
  2   3  3.0
  3   4  2.0
  4   5  4.0
  ```

  不指定on参数时，则默认按照index进行滚动窗口计算。 
  ```python
  >>> otime = orca.to_datetime(['20130101 09:00:00','20130101 09:00:02','20130101 09:00:03','20130101 09:00:05','20130101 09:00:06'])
  >>> odf = orca.DataFrame({'A': ["a", "c", "w", "f", "f"], 'B': [0, 1, 2, np.nan, 4]}, index=orca.Index(data=otime, name='time'))
  >>> odf
                       A    B
  2013-01-01 09:00:00  a  0.0
  2013-01-01 09:00:02  c  1.0
  2013-01-01 09:00:03  w  2.0
  2013-01-01 09:00:05  f  NaN
  2013-01-01 09:00:06  f  4.0

  >>> odf.rolling('2s').sum()
                         B
  2013-01-01 09:00:00  0.0
  2013-01-01 09:00:02  1.0
  2013-01-01 09:00:03  3.0
  2013-01-01 09:00:05  0.0
  2013-01-01 09:00:06  4.0
    ```

- `ewm`函数

  目前orca的`ewm`函数可调用以下函数：

    |函数|描述|
    |:---|:---|
    |mean|平均值|
    |std|标准差|
    |var|方差|

#### 4.3.6 计算函数

以下计算函数可用于orca中的Series/DataFrame对象：

  |函数|描述|
  |:---|:---|
  |abs|绝对值|
  |all|判断是否为空|
  |any|判断是否为空|
  |clip|介于阈值之间的值|
  |clip_lower|大于下界的值|
  |clip_upper|小于上界的值|
  |corr|相关性|
  |count|非空元素的个数|
  |cov|协方差|
  |cummax|累计最大值|
  |cummin|累计最小值|
  |cumprod|累乘|
  |cumsum|累加|
  |kurt|倾斜度|
  |kurtosis|峰度|
  |mad|平均绝对利差|
  |max|最大值|
  |mean|平均值|
  |median|中位数|
  |min|最小值|
  |mode|众数|
  |pct_change|百分比变化率|
  |prod|乘积|
  |product|乘积|
  |quantile|分位数|
  |rank|排名|
  |round|规整|
  |sem|无偏标准差|
  |skew|无偏斜|
  |std|标准差|
  |sum|求和|
  |var|方差|
  |nunique|非重复值的个数|

orca中的Series对象还支持以下函数：

  |函数|描述|
  |:---|:---|
  |between|返回介于阈值之间的值|
  |unique|返回不重复的值|
  |is_unique|判断是否有重复的值|
  |is_monotonic|判断是否单调递增|
  |is_monotonic_increasing|判断是否单调递增|
  |is_monotonic_decreasing|判断是否单调递减|

- `cummax`, `cummin`, `cumprod`和`cumsum`函数

  pandas中这些函数在遇到NaN值时会返回NaN，orca会在计算时略过NaN值。

  ```python
  >>> ps = pd.Series([1,2, np.nan, 4])
  >>> ps.cumsum()
    0    1.0
    1    3.0
    2    NaN
    3    7.0
    dtype: float64
    
  >>> os = orca.Series([1,2, np.nan, 4])
  >>> os.cumsum().compute()
    0    1.0
    1    3.0
    2    3.0
    3    7.0
    dtype: float64
  ```

- `quantile`函数

  当interpolation="nearest"，quantile又恰好在两个元素的中点时，pandas返回较大值，而orca返回较小值。
  ```python
  >>> x=[0, 1, 2, 3]
  >>> pd.Series(x).quantile(0.5,interpolation="nearest")
  2

  >>> orca.Series(x).quantile(0.5,interpolation="nearest")
  1.0
  ```
  此外，如上例所示，orca中`quantile`函数的结果一律为float类型。

- `rank`函数

  与pandas相比，orca的`rank`函数新增了两个参数：rank_from_zero和group_num。语法如下：
  ```python
  >>> rank(self, axis=0, method='min', numeric_o·nly=None, na_option='top', ascending=True, pct=False, rank_from_zero=False, group_num=None)
  ```
  - rank_from_zero参数
    当rank_from_zero取值为True时，排名从0起始，否则与pandas一致，排名从1起始。
  - group_num参数
    表示排序分组数，参考DolphinDB文档中的[`rank`](https://www.dolphindb.cn/cn/help/rank.html)函数。

  pandas的`rank`函数对重复的元素的排名处理有很多选项，默认选项为method='average'；而orca中对重复的元素的排名处理是固定的，只支持method='min'这一选项。
  
  pandas的`rank`函数的na_option参数可取值为'keep', 'top'与'bottom'，默认情况下NaN值返回NaN，不参与排名。而orca的中的`rank`函数的na_option参数仅可取值为'top'，NaN值视为最小值参与排名。

  ```python
  >>> x = [0.1, 1.3, 2.7, np.nan, np.nan, 1.3]
  >>> ps = pd.Series(x)
  >>> os = orca.Series(x)
  
  >>> ps.rank()
  0    1.0
  1    2.5
  2    4.0
  3    NaN
  4    NaN
  5    2.5
  dtype: float64

  >>> os.rank().compute()
  0    3
  1    4
  2    6
  3    1
  4    1
  5    4
  dtype: int32
  ```

- `sum`函数

  在pandas中，对字符串调用`sum`函数会将多个字符串拼接在一起；orca中不能对字符串调用`sum`函数。

- `Series.between`函数

  pandas中的`between`是一个三元运算符，上下边界都支持向量类型。orca的`between`函数仅支持标量作为参数，且不支持inclusive参数。

  ```python
  >>> ps=pd.Series([1,2,3,4])
  >>> ps.between([1,2,2,5],[10,10,3,6])
    0     True
    1     True
    2     True
    3    False
    dtype: bool
  
  >>> os=orca.Series([1,2,3,4])
  >>> os.between([1,2,2,5],[10,10,3,6])
  NotImplementedError
  >>> os.between(2,4).compute()
    0    False
    1     True
    2     True
    3     True
    dtype: bool
  ```

#### 4.3.7 Reindexing/selection/label manipulation

以下函数可用于orca中的Series/DataFrame对象：

  |函数|描述|
  |:---|:---|
  |drop_duplicates|删除重复的值|
  |duplicated|判断是否重复|
  |first|返回第一个值|
  |head|返回前n个值|
  |idxmax|返回index的最大值|
  |idxmin|返回index的最小值|
  |last|返回最后一个值|
  |rename|重命名|
  |tail|返回最后n个值|

orca中的DataFrame对象还支持以下函数：

  |函数|描述|
  |:---|:---|
  |drop|删除某列|
  |reindex|重置index|
  |reset_index|重置index|
  |set_index|设置index|

#### 4.3.8 Reshaping, sorting, transposing

以下函数可用于orca中的DataFrame对象：

  |函数|描述|
  |:---|:---|
  |droplevel|删除DataFrame的指定索引/列级别|
  |pivot|根据给定的行和列索引重构DataFrame|
  |pivot_table|从源数据构造透视表|
  |reorder_levels|根据给定顺序改变多级索引的先后顺序|
  |sort_values|按照指定的列进行排序|
  |stack|将数据的行索引变成列索引|
  |melt|`pivot`函数的逆操作|
  |transpose|交换行索引与列索引，相当于进行转置|

在使用以上函数时，orca与pandas存在以下差异：

- `pivot`函数

  orca的`pivot`函数的values参数只能为单列, 不允许是多个列。
  ```python
  >>> pdf = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'], 
  'bar': ['A', 'B', 'C', 'A', 'B', 'C'], 'baz': [1, 2, 3, 4, 5, 6], 'zoo': [2, 4, 5, 6, 4, 7]})
  >>> pdf.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
      baz       zoo      
  bar   A  B  C   A  B  C
  foo                    
  one   1  2  3   2  4  5
  two   4  5  6   6  4  7

  >>> odf = orca.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'], 
  'bar': ['A', 'B', 'C', 'A', 'B', 'C'], 'baz': [1, 2, 3, 4, 5, 6], 'zoo': [2, 4, 5, 6, 4, 7]})
  >>> odf.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
  TypeError: values must be a string
  ```

  orca中不能通过下标的方式指定values。
  ```python
  >>> pdf.pivot(index='foo', columns='bar')['baz']
  bar  A  B  C
  foo         
  one  1  2  3
  two  4  5  6

  >>> odf.pivot(index='foo', columns='bar')['baz']
  RuntimeError: <Server Exception> in run: Syntax Error: [line #1] Invalid expression: from ORCA_QGQ5EMGl9xE0Hzs9 pivot by foo as foo , bar as bar ) ) 
  ```

- `pivot_table`函数

  orca目前仅支持的参数有：values, index, columns和aggfunc。这些参数存在以下限制：

  - values参数：只能为单个字符串, 不允许是多个列，
  - index与columns参数：可以是单个字符串，或DataFrame的一个列，或DataFrame的一个或多个列的表达式，不允许是多个列。
  - aggfunc参数：只能是字符串，不允许是numpy的函数或者自定义函数。

  pandas支持的以下操作orca目前暂不支持：
  ```python
  >>> pdf = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                      "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                      "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                      "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                      "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
  ```
  1. 对某一列进行多种聚合运算：
  ```python
  >>> pdf.pivot_table(values='E', index='C',aggfunc={'E': ["min", "max", "mean"]})
            max  mean  min
  C                   
  large      9        6      4
  small      9        6      2
  ```

  2. values和index为多个列：
  ```python
  >>> pdf.pivot_table(values=['D', 'E'], index=['A', 'C'], aggfunc=np.mean)
                    D          E
  A   C                       
  bar large  5.500000   7.500000
      small  5.500000   8.500000
  foo large  2.000000   4.500000
      small  2.333333   4.333333
  ```

- `stack`函数

  pandas的`stack`函数返回的结果与orca略有差异。
  
  ```python
  >>> pdf = pd.DataFrame([[0, 1], [2, 3]], index=['cat', 'dog'], columns=['weight', 'height'])
  >>> odf = orca.DataFrame([[0, 1], [2, 3]], index=['cat', 'dog'], columns=['weight', 'height'])
  >>> odf
      weight   height
  cat      0       1
  dog      2       3

  >>> pdf.stack()
  cat   weight      0
        height      1
  dog   weight      2
        height      3
  dtype: int64
  
  >>> odf.stack()
  cat   weight     0
  dog   weight     2
  cat   height     1
  dog   height     3
  dtype: int64
  ```

  orca的`stack`函数不支持multi level columns。
  ```python
  >>> pdf = pd.DataFrame([[1, 2], [2, 4]], 
        index=['cat', 'dog'], 
        columns=pd.MultiIndex.from_tuples([('weight', 'kg'), ('weight', 'pounds')]))
  >>> odf = orca.DataFrame(pdf)
  >>> odf
      weight       
          kg pounds
  cat      1      2
  dog      2      4

  >>> pdf.stack()
                weight
  cat kg           1
      pounds       2
  dog kg           2
      pounds       4

  >>> odf.stack()
  ValueError: multi-level columns are not supported. 
  ```

- `melt`函数

  orca中`melt`函数的value_vars的类型必须是数值类型，而pandas支持更多的类型。
  ```python
  >>> pdf = pd.DataFrame({'A': {0:'a', 1:'b', 2:'c'}, 'B': {0:1, 1:3, 2:5},  'C': {0:2, 1:4, 2:6}})
  >>> odf = orca.DataFrame(pdf)

  >>> pdf.melt(id_vars='B',value_vars='A')
     B variable value
  0  1        A     a
  1  3        A     b
  2  5        A     c

  >>> odf.melt(id_vars='B',value_vars='A')
  TypeError: The value column to melt must be numeric type
  ```

  orca的`melt`函数暂不支持Multi level columns。
  ```python
  >>> pdf = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'}, 'B': {0: 1, 1: 3, 2: 5}, 'C': {0: 2, 1: 4, 2: 6}}) 
  >>> pdf.columns = [list('ABC'), list('DEF')]
  >>> pdf.melt(id_vars=[('A', 'D')], value_vars=[('B', 'E')])
    (A, D) variable_0 variable_1  value
  0      a          B          E      1
  1      b          B          E      3
  2      c          B          E      5

  >>> odf = orca.DataFrame(pdf)
  >>> odf.melt(id_vars=[('A', 'D')], value_vars=[('B', 'E')])
  KeyError: [('A', 'D')]
  ```

  orca.melt id_vars 和 value_vars可指定重复列。pandas不支持。
  ```python
  >>> pdf = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'}, 'B': {0: 1, 1: 3, 2: 5},  'C': {0: 2, 1: 4, 2: 6}})
  >>> odf = orca.DataFrame(pdf)
  >>> odf.melt(id_vars='B',value_vars='B')
    B variable  value
  0  1        B      1
  1  3        B      3
  2  5        B      5

  >>> pdf.melt(id_vars='B', value_vars='B')
  Exception: Data must be 1-dimensional
  ```

- `sort_values`函数

orca中`sort_values`函数目前仅支持ascending参数。在排序中，orca将NaN值视为最小值处理。
  ```python
  >>> ps=pd.Series([10, 1, 19, np.nan, -5])
  >>> os=orca.Series([10, 1, 19, np.nan, -5])
  >>> ps.sort_values()
  3    -5.0
  1     1.0
  0    10.0
  2    19.0
  4     NaN
  dtype: float64

  >>> os.sort_values()
  4     NaN
  3    -5.0
  1     1.0
  0    10.0
  2    19.0
  dtype: float64
  ```

- `transpose`函数

orca的`transpose`函数不允许对包含字符串的DataFrame或者各列数据类型不同的DataFrame使用。

DataFrame中含有字符串类型：
```python
>>> pdf = pd.DataFrame({'name': ['Alice', 'Bob'], 'grade': ['A', 'B'], 'Gender': ['female', 'male']})
>>> pdf.transpose()
             0     1
name     Alice   Bob
grade        A     B
Gender  female  male

>>> odf = orca.DataFrame(pdf)
>>> odf.transpose()
# raise error
```

DataFrame中不包含字符串类型，但是各个列数据类型不相同：
```python
>>> pdf = pd.DataFrame({'id': np.array([1,2],dtype=np.int32), 'val': np.array([10,20],dtype=np.int64)})
>>> pdf.transpose()
          0      1
id        1      2
val      10     20

>>> odf = orca.DataFrame(pdf)
>>> odf.transpose()
# raise error
```

#### 4.3.9 序列化、IO等函数

orca支持pandas的所有[序列化相关函数](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#serialization-io-conversion)。除此之外，orca还提供`to_pandas`函数，将一个orca对象转化为pandas的对象。

orca的`to_csv`函数与pandas有所不同，具体请参见[orca教程：数据写入](https://github.com/dolphindb/orca/blob/master/tutorial_cn/saving_data.md#1-%E5%B0%86%E6%95%B0%E6%8D%AE%E5%AF%BC%E5%87%BA%E5%88%B0%E7%A3%81%E7%9B%98)。

### 5 Index对象

orca目前支持的Index类型有Index, Int64Index, DatetimeIndex和MultiIndex。

#### 5.1 Index的属性

orca的Index对象具有以下属性：

  |属性|描述|
  |:---|:---|
  |values|返回取值|
  |is_monotonic|判断是否单调|
  |is_monotonic_increasing|判断是否单调递增|
  |is_monotonic_decreasing|判断是否单调递减|
  |is_unique|判断是否有重复的值|
  |hasnans|判断是否有空值|
  |dtype|返回数据类型|
  |shape|返回形状|
  |name|返回名字|
  |nbytes|返回字节数|
  |ndim|返回维度|
  |size|返回大小|
  |T|返回转置|

#### 5.2 相关函数

orca的Index对象支持以下函数：

  |函数|描述|
  |:---|:---|
  |max|最大值|
  |min|最小值|

### 6 groupby

orca的`groupby`函数目前仅支持by参数，且只能用于DataFrame。

以下函数可用于orca.DataFrameGroupBy对象：

  |函数|描述|
  |:---|:---|
  |all|判断是否为空|
  |any|判断是否为空|
  |bfill|向后填充|
  |count|非空元素的个数|
  |cumcount|累计非空元素的个数|
  |cummax|累计最大值|
  |cummin|累计最小值|
  |cumprod|累乘|
  |cumsum|累加|
  |ffill|向前填充|
  |first|返回第一个元素|
  |last|返回最后一个元素|
  |mad|平均绝对利差|
  |max|最大值|
  |mean|平均值|
  |median|中位数|
  |min|最小值|
  |ohlc|忽略空值求和|
  |pct_change|百分比变化率|
  |resample|重采样|
  |size|元素个数|
  |sem|无偏标准差|
  |skew|无偏斜|
  |std|标准差|
  |sum|求和|
  |var|方差|

### 7 重采样

orca支持`resample`函数，该函数目前支持的参数如下：

  |参数|说明|
  |:----|:----|
  |rule|DateOffset，可以是字符串或者是dateoffset对象|
  |on|时间列，采用该列进行重采样|
  |level|字符串或整数，对于MultiIndex，采用level指定的列进行重采样|

orca支持的DateOffset如下：

  |Date Offset|Frequency String|
  |:---|:---|
  |BDay or BusinessDay|'B'|
  |WeekOfMonth|'WOM'|
  |LastWeekOfMonth|'LWOM'|
  |MonthEnd|'M'|
  |MonthBegin|'MS'|
  |BMonthEnd or BusinessMonthEnd|'BM'|
  |BMonthBegin or BusinessMonthBegin|'BMS'|
  |SemiMonthEnd|'SM'|
  |SemiMonthBegin|'SMS'|	
  |QuarterEnd|'Q'|
  |QuarterBegin|'QS'|
  |BQuarterEnd|'BQ'|
  |BQuarterBegin|'BQS'|
  |FY5253Quarter|'REQ'|
  |YearEnd|'A'|
  |YearBegin|'AS' or 'BYS'|
  |BYearEnd|'BA'|
  |BYearBegin|'BAS'|
  |FY5253|'RE'|
  |Day|'D'|
  |Hour|'H'|
  |Minute|'T' or 'min'|
  |Second|'S'|
  |Milli|'L' or 'ms'|	
  |Micro|'U' or 'us'|	
  |Nano|'N'|

### 8 orca分区表

#### 8.1 orca分区表

pandas作为全内存计算的分析工具，无法解决数据量过大时的内存不足，计算效率低下等问题。DolphinDB是一个分布式时序数据库，内置了丰富的计算和分析功能。它可将TB级的海量数据存储在[DolphinDB分区表](https://github.com/dolphindb/Tutorials_CN/blob/master/database.md)中，充分利用CPU，进行高性能分析计算。
  
orca中可使用`read_csv`函数将数据导入DolphinDB的分区数据库，使用`read_table`函数来加载DolphinDB数据表。关于这两个函数的具体介绍请见[orca教程：数据加载](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/data_import.md)。

#### 8.2 orca分区表与内存表的使用差异

- 对表的计算

  pandas和orca的内存表调用`groupby`函数后支持继续调用`all`, `any`和`median`函数，orca的分区表则不支持。
    
- 对表的访问

  - 不能通过`iloc`和`loc`访问orca的分区表。
 
  - pandas和orca的内存表支持对非整数类型的index重复选择，而orca的分区表不支持：
      
    ```python
    >>> ps = pd.Series([0,1,2,3,4], index=['a','b','c','d','e'])
    >>> os = orca.Series([0, 1, 2, 3, 4], index=['a', 'b', 'c', 'd', 'e'])
    >>> ps[['a','b','a']]
    a    0
    b    1
    a    0
    dtype: int64

    >>> os[['a','b','a']]
    a    0
    b    1
    a    0
    dtype: int64
    ```
      
- 对表结构、表数据的修改 
    
  orca分区表不支持对表结构、表数据的修改操作。

  - 带有inplace参数的函数用于orca的分区表时，不可设置inplace=True。
    
    例如`rename`函数：
    ```python
    >>> df=orca.read_table("dfs://demoDB", "tb1")
    >>> df.rename(columns={"value": "values"}, inplace=True)
    ValueError: A segmented table is not allowed to be renamed inplace
      ```

欢迎在使用orca的同时，通过[GitHub issues](https://github.com/dolphindb/orca/issues)给我们反馈。orca将在我们的共同努力之下不断完善。











