# Orca与pandas的差异

由于DolphinDB是一款相对成熟的高性能分布式时序数据库，其底层对一些方法的处理机制已经成型，这就决定了Orca在某些细节方面会与pandas存在差异。为了方便用户更快地了解和掌握Orca，本文按照以下几个模块来系统地介绍Orca与pandas存在的差异。

- [1 数据类型的差异](#1-数据类型的差异)
- [2 通用函数的差异](#2-通用计算函数的差异)
- [3 Input/output的差异](#3-inputoutput的差异)
- [4 Series、DataFrame的差异](#4-seriesdataframe的差异)
  - [4.1 Series和DataFrame的创建与修改](#41-series和dataframe的创建与修改)
  - [4.2 Series和DataFrame的四则运算](#42-series和dataframe的四则运算)
  - [4.3 Sereis和DataFrame的属性和方法](#43-sereis和dataframe的属性和方法)
- [5 Index Objects的差异](#5-index-objcts的差异)
- [6 GroupBy的差异](#6-groupby的差异)
- [7 Resampling的差异](#7-resampling的差异)
- [8 Orca分区表的特殊差异](#8-Orca分区表的特殊差异)

### 1 数据类型的差异

  - DolphinDB支持CHAR, SHORT, INT, LONG等不同字节数的整数类型，整数字面量默认解析为32位INT类型，而pandas的整数字面量默认解析为64位。
  - DolphinDB支持STRING和SYMBOL两种字符串类型，STRING类型可以进行max, min等比较运算，但SYMBOL不允许。而pandas中的字符串类型的底层存储是np.object。pandas也提供了经过优化的category类型，使用整数作为底层存储，对于取值范围有限的数据，能减少内存占用。DolphinDB的SYMBOL类型能实现类似的功能。但pandas允许将任何数据类型的数据转换为category类型，而DolphinDB只允许字符串类型。
  - DolphinDB支持多种不同单位的日期和时间类型，包括DATE, MINUTE, SECOND, TIME, NANOTIME等，而pandas底层的日期时间存储使用np.datetime64\[ns\]类型，以freq表示时间单位
  - DolphinDB要求表一列中元素的数据类型必须相同，而pandas允许一个Series的中的数据有不同数据类型。因此，DolphinDB中字符串类型的NULL值实际上是一个空字符串，而pandas中字符串类型的空值是np.NaN。

### 2 通用函数的差异

  Orca提供以下通用函数：

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

  下面对`connect`函数和`save_table`函数加以说明：

  - `connect`函数

    由于Orca是基于DolphinDB Python API开发的，为此，Orca提供`connect`函数来建立一个与DolphinDB服务器的连接，而且在开始使用Orca函数之前，必须首先调用该函数创建连接。

    ```python
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)
    ```

  - `save_table`函数

    该函数的具体意义及用法请参见[Orca写数据教程](https://2xdb.net/dolphindb/Orca/blob/master/%E5%86%99%E6%95%B0%E6%8D%AE%E6%95%99%E7%A8%8B.md)。

### 3 Input/output的差异

  Orca现在支持的Input/output函数有：`read_csv`和`read_table`。

  - `read_csv`函数

    下面详细介绍Orca的`read_csv`函数与pandas的`read_csv`函数的差异。

    - engine参数

      pandas的`read_csv`函数的engine参数的取值可以是‘c’或者‘python’，表示使用哪一种引擎进行导入。
      
      而Orca的`read_csv`函数的engine参数的取值可以是{‘c’, ‘python’, ‘dolphindb’}，且该参数默认取值为‘dolphindb’。当取值为‘dolphindb’时，`read_csv`函数会在DolphinDB服务器目录下寻找要导入的数据文件。当取值为‘python’或‘c’时，`read_csv`函数会在python客户端的目录下寻找要导入的数据文件。

    > 注意，当engine参数设置为‘python’或者‘c’时，Orca的`read_csv`函数相当于调用了pandas的`read_csv`函数进行导入。下面列出的差异均是在engine参数设置为‘dolphindb’的前提下的差异。

    当engine参数设置为‘dolphindb’时，Orca的`read_csv`函数目前支持的参数如下：

    ```Python
    read_csv(path, sep=',', delimiter=None, names=None,  index_col=None,engine='dolphindb', usecols=None, squeeze=False, prefix=None, dtype=None, partitioned=True, db_handle=None, table_name=None, partition_columns=None, *args, **kwargs):
    ```

    - dtype参数

      在pandas中，`read_csv`函数支持dtype参数，该参数接收一个字典，键是列名，值是Python原生类型（bool, int, float, str）或np的dtype（np.bool, np.int8, np.float32, etc.）
      
      例如：
      
      ```Python
      dfcsv = pd.read_csv("path_to/allTypesOfColumns.csv", dtype={"tbool": np.bool, "tchar": np.int8, "tshort": np.int16, "tint": np.int32, "tlong": np.int64, "tfloat": np.float32, "tdouble": np.float64})
      ```
      
      与pandas不同的是，Orca的`read_csv`函数的dtype参数还支持以字符串的方式指定DolphinDB的提供的所有[数据类型](https://www.dolphindb.cn/cn/help/DataType.html)，包括所有时间类型和字符串类型。
      
      例如：
      
      ```Python
      dfcsv = orca.read_csv("path_to/allTypesOfColumns.csv", dtype={"tstring":'STRING', "tsymbol": 'SYMBOL', "date": 'DATE', "second": 'SECOND'， "tint": np.int32})
      ```
    
    - sep和delimiter参数

      pandas的这两个参数支持对正则表达式的解析，而Orca的目前无法支持这一点。

    - partitioned参数 

      bool类型，该参数为True时，表示允许分区方式将数据导入(实际上是调用DolphinDB的[ploadText函数](https://www.dolphindb.cn/cn/help/ploadText.html))；当该参数为False时，表示强制以非分区的方式导入数据(实际上是调用DolphinDB的[loadText函数](https://www.dolphindb.cn/cn/help/loadText.html))。

      > 注意：Orca的分区表与Orca的内存表相比，在操作时也存在许多差异，具体见[Orca分区表的特殊差异](#8-Orca分区表的特殊差异)。若您的数据量不是很大，且在使用Orca时对Orca与pandas的一致性要求更高，请尽量不要将数据以分区的方式导入。若您数据量极大，对性能要求极高，则建议您采用分区方式导入数据。

    - db_handle, table_name以及partition_columns参数

      Orca的`read_csv`还支持db_handle, table_name和partition_columns这3个参数，这些参数用于在导入数据的时通过指定DolphinDB的数据库和表等相关信息，将数据导入到DolphinDB的分区表，关于这几个参数的具体用法与示例请参见[Orca写数据教程](https://2xdb.net/dolphindb/Orca/blob/master/%E5%86%99%E6%95%B0%E6%8D%AE%E6%95%99%E7%A8%8B.md)。

  - `read_table`函数

    在pandas中，`read_table`函数用于导入一个表格形式的文件。在Orca中，`read_table`函数用于导入一个[DolphinDB的分区表](https://github.com/dolphindb/Tutorials_CN/blob/master/database.md)。
    
    例如，假设DolphinDB Server上已有数据库和表如下：
    
    ```
    tdata=table(1..5 as id, rand(2.0,5) as value)
    db=database('dfs://testRead',VALUE,1..5)
    tb=db.createPartitionedTable(tdata,`tb1,`id)
    tb.append!(tdata)
    ```
    
    其中，数据库名称为"dfs://testRead"，创建的分区表名为"tb1"。我们可以通过`read_table`函数将一个分区表加载到Python应用程序中，存放在一个Orca的DataFrame对象里。
    
    ```Python
    import orca
    orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    odfs=orca.read_table("dfs://testRead","tb1")
    ```

### 4 Series、DataFrame的差异

  本节主要介绍Orca的Series、DataFrame与pandas的Series、DataFrame的差异。

#### 4.1 Series和DataFrame的创建与修改

  - Series或DataFrame的创建

    pandas允许在定义一个Series时不设置name参数，或者使用数字作为name，这在Orca中的实现相当于在DolphinDB server端新建一个只含有一列的表，而表的列名则不允许为空值且不能使用数字。因此，在创建Orca的Series而不指定名字时，系统会默认为该Series自动生成一个名字，当然，用户不会感知到自动生成的名字，只是会看到Orca抛出的WARNING信息。例如，创建一个名字为0的series，Orca会抛出WARNING：

    ```Python
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> a = orca.Series([1, 2, 3, 4], name='0')
    # raise warning:
    /Orca/Orca/core/common.py:36: NotDolphinDBIdentifierWarning: The DataFrame contains an invalid column name for DolphinDB. Will convert to an automatically generated column name.
      "generated column name.", NotDolphinDBIdentifierWarning)
    >>> a
    # output
    0    1
    1    2
    2    3
    3    4
    Name: 0, dtype: int64
    ```

  - Series或DataFrame的修改

    pandas与Orca的强制转换机制存在差异。若将一个精度更高的数据类型的值赋值给一个精度更低的Series，在pandas中最终得到的Series取值为向下取整，在Orca中则为四舍五入取整。

    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> ps=pd.Series([10,20,30], index=[0,2,3])
    >>> os=orca.Series([10,20,30], index=[0,2,3])
    >>> os
    # output
    Out[26]: 
    0    10
    2    20
    3    30
    dtype: int64

    # set
    >>> os[0]=100.5
    >>> os
    # output 
    0    101
    2     20
    3     30
    dtype: int64

    >>> ps[0]=100.5
    >>> ps
    # output 
    0    100
    2     20
    3     30
    dtype: int64
    ```

  - 向Series或DataFrame的追加数据
    
    pandas允许通过直接访问一个不存在的index去增加新的行，但是Orca暂不支持这么做。

    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
    >>> os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
    >>> ps
    # output
    a    10
    b     1
    c    19
    d    -5
    dtype: int64

    >>> ps['e']=1
    >>> ps
    # output
    a    10
    b     1
    c    19
    d    -5
    e     1
    dtype: int64

    >>> os
    # output
    a    10
    b     1
    c    19
    d    -5
    dtype: int64

    >>> os['e']=1
    >>> os 
    # output
    # still not changed 
    a    10
    b     1
    c    19
    d    -5
    dtype: int64
    ```

#### 4.2 Series和DataFrame的四则运算

  - 空值的处理

    在pandas中，任何数与空值比较，返回都是False，这其实是Python中NaN比较的规则，而Orca则将空值视为该类型的最小值。

    下例中，分别对pandas和Orca的Series进行条件过滤，可以看出Orca将NaN值视为符合过滤条件的值输出。

    ```Python
    >>> import pandas as pd
    >>> import numpy as np
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> ps = pd.Series([1,np.nan,0])
    >>> os = orca.Series([1,np.nan,0])
    >>> ps[ps<1]
    # output
    2    0.0
    dtype: float64

    >>> os[os<1].compute()
    # output
    1    NaN
    2    0.0
    dtype: float64
    ```

  - 空字符串的处理

    pandas的字符串会区分NaN值和空字符串，Orca的空字符串就是NaN。

    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> ps = pd.Series(["","s","a"])
    >>> os = orca.Series(["","s","a"])
    >>> ps.hasnans
    # output
    False

    >>> os.hasnans
    # output
    True
    ```

  - 零的处理

    pandas非零数除以零得到同符号的无穷大；零除以零得到NaN。Orca任何数除以零得到NULL。

    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> ps=pd.Series([1,0,2])
    >>> os=orca.Series([1,0,2])

    >>>ps.div(0)
    # output
    0    inf
    1    NaN
    2    inf
    dtype: float64

    >>> os.rdiv(0).compute()
    # output
    0   NaN
    1   NaN
    2   NaN
    dtype: float64
    ```

#### 4.3 Sereis和DataFrame的属性和方法

  本小节根据pandas官方提供的[Series](https://pandas.pydata.org/pandas-docs/stable/reference/series.html)和[DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#dataframe)的文档，依次介绍Orca与pandas的不同之处。

#### 4.3.1 Attributes and underlying data

  除了pandas已经取缔的属性之外，Orca的Series和DataFrame唯一没有支持的属性就是`memory_usage`这一属性。

#### 4.3.2 Conversion

  由于Orca的优势在于对批量数据读写与计算，因此目前在Conversion方面的功能并不完善，现在仅支持`Series.to_numpy`这一功能。

#### 4.3.2 Indexing, iteration

  以下函数可用于orca.DataFrame对象和orca.Series对象：

  |函数|描述|
  |:---|:---|
  |head|返回前n个值|
  |tail|返回最后n个值|
  |loc|通过index访问|
  |iloc|通过下标访问|
  |where|用NaN填充不符合过滤条件的值|
  |mask|用NaN填充符合过滤条件的值|

  orca.DataFrame对象还以下函数：

  |函数|描述|
  |:---|:---|
  |items|遍历DataFrame|
  |iteritems|遍历DataFrame|
  |lookup|根据标签查询数据|
  |get|访问某一列|

  下面对`loc`，`iloc`做特殊说明。

  - 通过`loc`访问Series和DataFrame

    如下所示，Orca暂不支持通过loc去访问带有DatetimeIndex的Series和DataFrame。
    
    ```Python
    >>> import pandas as pd
    >>> import numpy as np
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> pdd = pd.DataFrame(
              {'id': [1, 2, 2, 3, 3], 'sym': ['s', 'a', 's', 'a', 's'], 'values': [np.nan, 2, 2, np.nan, 2]},
              index=pd.date_range('20190101', '20190105', 5))
    >>> odd = orca.DataFrame(pdd)
    >>> pdd
    # output
                id sym  values
    2019-01-01   1   s     NaN
    2019-01-02   2   a     2.0
    2019-01-03   2   s     2.0
    2019-01-04   3   a     NaN
    2019-01-05   3   s     2.0
    
    >>> pdd.loc["20190103":"20190105"]
    # output
                id sym  values
    2019-01-03   2   s     2.0
    2019-01-04   3   a     NaN
    2019-01-05   3   s     2.0
    
    >>> odd.loc["20190103":"20190105"]
    # raise error
    ```

    当DataFrame的表中有重复的index时，padas不支持以重复的index值为slice的下界，而Orca则以第一个出现的重复值为slice的下界输出结果。

    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> pdf = pd.DataFrame([[1, 2, 1], [4, 5, 5], [7, 8, 7], [1, 5, 8], [7, 5, 1]], index=[7, 8, 2, 8, 9],
                    columns=['max_speed', 'shield', 'size'])
    >>> odf = orca.DataFrame([[1, 2, 1], [4, 5, 5], [7, 8, 7], [1, 5, 8], [7, 5, 1]], index=[7, 8, 2, 8, 9],
                        columns=['max_speed', 'shield', 'size'])
    >>> pdf
    # output
      max_speed  shield  size
    7          1       2     1
    8          4       5     5
    2          7       8     7
    8          1       5     8
    9          7       5     1
    >>> pdf.loc[8:]
    # raise error

    >>> odf
    # output
      max_speed  shield  size
    7          1       2     1
    8          4       5     5
    2          7       8     7
    8          1       5     8
    9          7       5     1
    
    >>> odf.loc[8:]
    # output 
      max_speed  shield  size
    8          4       5     5
    2          7       8     7
    8          1       5     8
    9          7       5     1
    ```

  - 通过`loc`与`iloc`修改Series和DataFrame中值的类型

    pandas可以通过`loc`和`iloc`更改DataFrame中一个列（Series）的类型。更改其中一个值的类型会导致整列类型变更，也可以直接通过调用`astype`函数更改整列的类型。而Orca不允许修改列的类型。

    下例中，pandas通过`loc`去修改Series的值，返回的结果是整列Series变成了np.float64类型，而Orca返回的结果仍然是np.int64类型。通过`iloc`去访问Series和DataFrame并进行类似的修改操作同样会有类似的差异。

    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> ps = pd.Series([10, 20, 30], index=[0, 2, 3])
    >>> ps
    # output
    0    10
    2    20
    3    30
    dtype: int64

    >>> os = orca.Series([10, 20, 30], index=[0, 2, 3])
    >>> os
    # output
    0    10
    2    20
    3    30
    dtype: int64

    >>>ps.loc[0]=100.5
    >>>ps
    # output
    0    100.5
    2     20.0
    3     30.0
    dtype: float64

    >>> os.loc[0]=100.5
    >>> os
    # output
    0    101
    2     20
    3     30
    dtype: int64
    ```

  - 通过`loc`与`iloc`修改Series和DataFrame的值

    Orca不支持：当index有重复的列，通过一个DataFrame以index对齐的原则去修改另一个DataFrame的值
    
    ```Python
    >>> import pandas as pd
    >>> import numpy as np
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> pdf = pd.DataFrame([[1, 2, 1], [4, 5, 5], [7, 8, 7], [1, 5, 8], [7, 5, 1]], index=[7, 8, 9, 8, 9],  columns=['max_speed', 'shield', 'size'])
    >>> pdf
    # output
      max_speed  shield  size
    7          1       2     1
    8          4       5     5
    9          7       8     7
    8          1       5     8
    9          7       5     1

    >>> pdf.loc[7:] = pd.DataFrame([[1, 1, 1], [5, 5, 5], [7, 7, 7], [8, 8, 8], [6, 6, 6]], index=[7, 8, 9, 8, 9], columns=['max_speed', 'shield', 'size'])
    >>> pdf
    # output
      max_speed  shield  size
    7          1       1     1
    8          5       5     5
    9          7       7     7
    8          8       8     8
    9          6       6     6

    >>> odf = orca.DataFrame([[1, 2, 1], [4, 5, 5], [7, 8, 7], [1, 5, 8], [7, 5, 1]], index=[7, 8, 9, 8, 9], columns=['max_speed', 'shield', 'size'])
    >>> odf
    # output
        max_speed  shield  size
    7          1       2     1
    8          4       5     5
    9          7       8     7
    8          1       5     8
    9          7       5     1

    >>> odf.loc[7:] = orca.DataFrame([[1, 1, 1], [5, 5, 5], [7, 7, 7], [8, 8, 8], [6, 6, 6]], index=[7, 8, 9, 8, 9], columns=['max_speed', 'shield', 'size'])
    # raise error
    ```

  - 通过`loc`向DataFrame新增一行或者一列

    pandas支持直接通过loc访问不存在的index或者columns来新增行或者列，而Orca暂不支持。

    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> pdf = pd.DataFrame([[1, 2, 1], [4, 5, 5], [7, 8, 7], [1, 5, 8], [7, 5, 1]], index=[7, 8, 2, 8, 9],
                    columns=['max_speed', 'shield', 'size'])
    >>> odf = orca.DataFrame([[1, 2, 1], [4, 5, 5], [7, 8, 7], [1, 5, 8], [7, 5, 1]], index=[7, 8, 2, 8, 9],
                      columns=['max_speed', 'shield', 'size'])
    >>> pdf
    # output
    a    10
    b     1
    c    19
    d    -5
    dtype: int64

    >>> ps.loc['e']=1
    >>> ps
    # output
    a    10
    b     1
    c    19
    d    -5
    e     1
    dtype: int64

    >>> os
    # output
    a    10
    b     1
    c    19
    d    -5
    dtype: int64

    >>> os['e']=1
    >>> os 
    # output
    # still not changed 
    a    10
    b     1
    c    19
    d    -5
    dtype: int64
    ```

  - `loc`和`iloc`暂不支持对MultiIndex的访问

#### 4.3.3 Binary operator functions

  除了[combine](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.combine.html#pandas.DataFrame.combine)和[combine_first](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.combine_first.html#pandas.DataFrame.combine_first)函数之外，Orca支持pandas提供的所有[二元函数](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#binary-operator-functions)。但是，Orca的DataFrame或者Series在进行四则运算时，除了本文第2.2小节所提及的差异之外，在四则运算进行的方式上也存在一定差异。

  - 二元运算函数的axis参数

    pandas提供的二元运算函数中都有一个axis参数。需要说明的是，Orca的DataFrame和Orca的Series进行二元运算时，不支持aixs参数取值为'columns'或0

    pandas中，对DataFrame和Series进行加法运算，有如下几种形式：

    ```Python
    >>> import pandas as pd

    >>> pdf = pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
    >>> pdf + pd.Series([1, 2], index=["angles", "degrees"])
    # output
              angles  degrees
    circle          1      362
    triangle        4      182
    rectangle       5      362

    >>> pdf.add(pd.Series([1, 2], index=["angles","degrees"]))
    # output
              angles  degrees
    circle          1      362
    triangle        4      182
    rectangle       5      362

    >>> pdf.add(pd.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle']), axis='index')
    # output
                angles  degrees
    circle           1      361
    triangle         5      182
    rectangle        7      363
    ```

    上例中，直接通过”+“号将DataFrame和Series进行的相加以及不指定axis参数的相加默认按照axis=0的规则进行相加。Orca不支持这种情况，仅支持axis='index'或1的情况。因此在Orca中，一个DataFrame可以这样与一个Series进行四则运算：

    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
    >>> odf.add(orca.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle']), axis='index')
              angles  degrees
    circle          1      361
    triangle        5      182
    rectangle       7      363
    ```

    > 注意，在Orca中，只有上述情况是不支持axis='columns'的，若将一个DataFrame与一个list相加，axis='columns'的情况还是支持的：  
    
    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
    >>> (odf + [1, 2]).compute()
    # output
              angles  degrees
    circle         1      362
    triangle       4      182
    rectangle      5      362
    ```

  - 除数是负数

    在Orca中，如果负数在除法中作为除数，返回的结果将是NaN。

    如下例所示，在求余运算中，除数中出现了负数，Orca返回的结果对应位置值为NaN。

    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)
    
    >>> pd.Series([1, 2, 12, 10], index=['a', 'b', 'c', 'd']) % [10, 1, 19, -4]
    # output
    a     1
    b     0
    c    12
    d    -2
    dtype: int64

    >>> (orca.Series([1, 2, 12, 10], index=['a', 'b', 'c', 'd']) % [10, 1, 19, -4]).compute()
    # output
    a     1.0
    b     0.0
    c    12.0
    d     NaN
    dtype: float64
    ```

  - 求余运算

    pandas支持对浮点数的求余运算，而Orca暂不支持。
    
    ```Python
    >>> import pandas as pd
    >>> import numpy as np
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)
    
    >>> pd.Series([5.5, 10, -4.5, 2.5, np.nan]) % pd.Series([2.5, -4.5, 2.5, np.nan, 3])
    # output
    0    0.5
    1   -3.5
    2    0.5
    3    NaN
    4    NaN
    dtype: float64

    >>> orca.Series([5.5, 10, -4.5, 2.5, np.nan]) % orca.Series([2.5, -4.5, 2.5, np.nan, 3])
    # raise error
    ```

#### 4.3.4 Function application, GroupBy & window

  以下函数可用于orca.DataFrame对象和orca.Series对象：

  |函数|描述|
  |:---|:---|
  |apply|应用多个函数|
  |agg|应用多个聚合函数|
  |aggregate|应用多个聚合函数|
  |groupby|分组运算|
  |rolling|滑动窗口|
  |ewm|指数加成滑动|

  下面介绍一下Orca与pandas仍存在的差异。

  - `apply`，`agg`，`aggregate`函数

    Orca的这三个函数目前仅支持字符串或者一个字典，不支持lambda函数。

    ```Python
    >>> import pandas as pd

    >>> ps=pd.Series([1, 2, 12, 10], index=['a', 'b', 'c', 'd'])
    >>> ps.apply(lambda x: x + 1)
    ```
    
    上面的脚本在Orca中，则需要这样实现：
    
    ```Python
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> os=orca.Series([1, 2, 12, 10], index=['a', 'b', 'c', 'd'])
    >>> os.apply("(x->x+1)")
    ```

    关于这三个函数的具体限制，请参见[Orca使用教程高阶函数部分](https://2xdb.net/dolphindb/Orca/blob/master/Orca%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B.md#%E9%AB%98%E9%98%B6%E5%87%BD%E6%95%B0)。

  - `groupby`函数
  
    Orca的groupby函数目前支持的参数如下：

    ```Python
    DataFrame.groupby(self, by=None, level=None, as_index=True, sort=True, squeeze=False, ascending=True, **kwargs)
    ```
    具体差异请参见文章[第6节](#6-groupby的差异)。

  - `rolling`函数

    Orca的rolling函数目前支持window和on参数。在进行rolling时，若遇到空值，pandas在对应位置返回NaN，而Orca返回上一次计算的结果。

    ```Python
    >>> import pandas as pd
    >>> import numpy as np
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> pdf = pd.DataFrame({'id': np.arange(1, 6, 1), 'B': [0, 1, 2, np.nan, 4]})
    >>> pdf
    # output
      id    B
    0   1  0.0
    1   2  1.0
    2   3  2.0
    3   4  NaN
    4   5  4.0

    >>> pdf.rolling(2, on="id").sum()
    # output  
      id    B
    0   1  NaN
    1   2  1.0
    2   3  3.0
    3   4  NaN
    4   5  NaN

    >>> odf = orca.DataFrame({'id': np.arange(1, 6, 1), 'B': [0, 1, 2, np.nan, 4]})
    >>> odf
    # output 
      id    B
    0   1  0.0
    1   2  1.0
    2   3  2.0
    3   4  NaN
    4   5  4.0

    >>> odf.rolling(2, on="id").sum()
    # output
      id    B
    0   1  NaN
    1   2  1.0
    2   3  3.0
    3   4  2.0
    4   5  4.0
    ```

    不指定on参数时，则默认按照index进行rolling。

    ```Python
    >>> import pandas as pd
    >>> import numpy as np
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> otime = orca.to_datetime(['20130101 09:00:00','20130101 09:00:02','20130101 09:00:03','20130101 09:00:05','20130101 09:00:06'])
    >>> odf = orca.DataFrame({'A': ["a", "c", "w", "f", "f"], 'B': [0, 1, 2, np.nan, 4]}, index=orca.Index(data=otime, name='time'))
    >>> odf
    # output
                        A    B
    2013-01-01 09:00:00  a  0.0
    2013-01-01 09:00:02  c  1.0
    2013-01-01 09:00:03  w  2.0
    2013-01-01 09:00:05  f  NaN
    2013-01-01 09:00:06  f  4.0

    >>> odf.rolling('2s').sum()
    # output
                          B
    2013-01-01 09:00:00  0.0
    2013-01-01 09:00:02  1.0
    2013-01-01 09:00:03  3.0
    2013-01-01 09:00:05  0.0
    2013-01-01 09:00:06  4.0
    ```

  - `ewm`函数

    目前Orca的`ewm`函数可调用以下函数：

    |函数|描述|
    |:---|:---|
    |mean|平均值|
    |std|标准差|
    |var|方差|

#### 4.3.5 Computations/descriptive stats的差异

  以下函数可用于orca.DataFrame对象和orca.Series对象：

  |函数|描述|
  |:---|:---|
  |abs|绝对值|
  |all|判断是否为空|
  |any|判断是否为空|
  |clip|返回介于阈值之间的值|
  |clip_lower|返回大于下界的值|
  |clip_upper|返回小于上界的值|
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
  |mode|返回出现最多的行|
  |pct_change|百分比变化率|
  |prod|返回乘积|
  |product|返回乘积|
  |quantile|分位数|
  |rank|排名|
  |round|规整|
  |sem|无偏标准差|
  |skew|无偏斜|
  |std|标准差|
  |sum|求和|
  |var|方差|
  |nunique|返回非重复值的个数|

  orca.Series对象还具备以下函数：

  |函数|描述|
  |:---|:---|
  |between|返回介于阈值之间的值|
  |unique|返回不重复的值|
  |is_unique|判断是否有重复的值|
  |is_monotonic|判断是否单调|
  |is_monotonic_increasing|判断是否单调递增|
  |is_monotonic_decreasing|判断是否单调递减|

  在Orca提供的函数中，有以下差异。

  - `cummax`, `cummin`, `cumprod`和`cumsum`函数

    这几个函数在遇到NaN值时会返回NaN，Orca会在NaN值的位置返回前一个计算结果。

    ```Python
    >>> import pandas as pd
    >>> import numpy as np
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> pdf = pd.DataFrame([[1, np.nan, 3], [4, np.nan, 6], [7, np.nan, 9], [np.nan, np.nan, 1]], columns=['A', 'B', 'C'])
    >>> odf = orca.DataFrame([[1, np.nan, 3], [4, np.nan, 6], [7, np.nan, 9], [np.nan, np.nan, 1]], columns=['A', 'B', 'C'])

    >>> pdf.cumsum()
    # output
          A   B     C
    0   1.0 NaN   3.0
    1   5.0 NaN   9.0
    2  12.0 NaN  18.0
    3   NaN NaN  19.0

    >>> odf.cumsum().compute()
    # output
          A   B   C
    0   1.0 NaN   3
    1   5.0 NaN   9
    2  12.0 NaN  18
    3  12.0 NaN  19
    ```

  - `rank`函数

    Orca的`rank`函数支持的参数如下：

    ```Python
    rank(self, ascending=True, rank_from_zero=False, group_num=None)
    ```

    与pandas相比，该函数新增了两个参数：rank_from_zero和group_num。

    - rank_from_zero参数

      当rank_from_zero取值为True时，最小的排名为0，否则最小的排名为1，和pandas一致。

    - group_num参数

      表示排名的分组数，参考DolphinDB文档中的[`rank`函数](https://www.dolphindb.cn/cn/help/rank.html)。

    在计算时，pandas的`rank`函数会把重复的排名取平均值，而Orca中两个重复的元素具有相同的排名。在遇到NaN时，pandas返回NaN，而Orca将NaN值视为最小值。

    ```Python
    >>> import pandas as pd
    >>> import numpy as np
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> ps = pd.Series([0.1, 1.3, 2.7, np.nan, np.nan, 1.3])
    >>> os = orca.Series([0.1, 1.3, 2.7, np.nan, np.nan, 1.3])
    >>> os 
    # output
    0    0.1
    1    1.3
    2    2.7
    3    NaN
    4    NaN
    5    1.3
    dtype: float64

    >>> ps.rank()
    # output
    0    1.0
    1    2.5
    2    4.0
    3    NaN
    4    NaN
    5    2.5
    dtype: float64

    >>> os.os.rank().compute()
    # output
    0    3
    1    4
    2    6
    3    1
    4    1
    5    4
    dtype: int32
    ```

  - `sum`函数

    在Pandas中，对字符串调用`sum`函数会将多个字符串拼接在一起，Orca则不能对字符串调用`sum`函数。

  - `Series.between`函数

    pandas中的`between`是一个三元运算符，上下边界都支持向量类型。Orca的`between`函数仅支持标量作为参数，且不支持inclusive参数。

    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> ps=pd.Series([10, 1, 19, -5])
    >>> os=orca.Series([10, 1, 19, -5])
    >>> ps.between([1,2,4,5],[15,10,4,5])
    # output
    0     True
    1    False
    2    False
    3    False
    dtype: bool
    >>> os.between([1,2,4,5],[15,10,4,5])
    # raise error
    ```

#### 4.3.6 Reindexing/selection/label manipulation的差异

  以下函数可用于orca.DataFrame对象和orca.Series对象：

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

  orca.DataFrame对象还具有以下函数：

  |函数|描述|
  |:---|:---|
  |drop|删除某列|
  |reindex|重置index|
  |reset_index|重置index|
  |set_index|设置index|

#### 4.3.7 Reshaping, sorting

Orca目前仅支持`sort_values`函数，且该函数仅支持ascending参数提供排序的功能，将NaN值视为最小值处理。

  - `sort_values`函数

  Orca的`sort_values`函数仅支持
  ```Python    
  >>> import pandas as pd
  >>> import numpy as np
  >>> import orca
  >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

  >>> ps=pd.Series([10, 1, 19, -5, np.nan])
  >>> os=orca.Series([10, 1, 19, -5, np.nan])
  >>> ps.sort_values()
  # output
  3    -5.0
  1     1.0
  0    10.0
  2    19.0
  4     NaN
  dtype: float64

  >>> os.sort_values()
  # output
  4     NaN
  3    -5.0
  1     1.0
  0    10.0
  2    19.0
  dtype: float64
  ```

#### 4.3.8 Serialization / IO / conversion

  Orca支持pandas所支持的所有[序列化相关函数](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#serialization-io-conversion)，并提供一个`to_pandas`函数，该函数将一个Orca对象转化为pandas的对象。

### 5 Index Objcts的差异

  Orca目前支持的Index类型有Index,Int64Index,DatetimeIndex和MultiIndex，下面介绍Index对象所支持的属性和方法。

#### 5.1 Index的属性

  Orca的Index对象具有以下属性：

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

#### 5.2 Modifying and computations

  Orca的Index对象支持以下函数：

  |函数|描述|
  |:---|:---|
  |max|最大值|
  |min|最小值|

### 6 GroupBy的差异

  Orca的`groupby`函数目前仅支持by参数，且只能对DataFrame进行groupby。

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

### 7 Resampling的差异

  Orca的`resample`函数目前仅支持rule和on两个参数。其中freq可以指定以下Date Offset：

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

### 8 Orca分区表的特殊差异

#### 8.1 Orca的分区表

  pandas作为全内存计算的分析工具，无法解决当数据量过大时带来的内存不足，计算效率低下等问题。
  
  DolphinDB是一个分布式时序数据库，并且内置了丰富的计算和分析功能。它可以将TB级的海量数据存储在多台物理机器上，充分利用CPU，对海量数据进行高性能分析计算。
  
  Orca作为基于DolphinDB开发的分布式pandas接口，其最大的优势就是在语法和pandas保持一致的前提下很好地解决了pandas的瓶颈：大数据场景下的性能问题。而这一问题的解决，则依赖于[DolphinDB分区表](https://github.com/dolphindb/Tutorials_CN/blob/master/database.md)。在Orca中，我们也引入Orca分区表的概念。

  在下列情况下，数据将以分区表的形式处理：

  - `read_csv`函数

    在本文[第3节](#3-inputoutput的差异)中曾提及，指定partitioned参数为True会将数据以分区的方式导入。

  - 通过`read_table`函数指定数据库和表导入分区表

#### 8.2 Orca分区表的特殊差异

  在DolphinDB中，分区表与内存表存在着一些差异，在Orca中，分区表的操作也存在这诸多限制。

  - `all`，`any`和`median`函数
  
  pandas和Orca的内存表进行group by之后支持调用`all`,`any`和`median`函数，Orca的分区表则不支持。

  - 对非整数类型的index重复选择

    pandas和Orca的内存表支持以下操作，而Orca的分区表不支持：
    
    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> ps = pd.Series([0,1,2,3,4], index=['a','b','c','d','e'])
    >>> os = orca.Series([0, 1, 2, 3, 4], index=['a', 'b', 'c', 'd', 'e'])
    >>> ps[['a','b','a']]
    # output
    a    0
    b    1
    a    0
    dtype: int64
    >>> os[['a','b','a']]
    # output
    a    0
    b    1
    a    0
    dtype: int64
    ```

  - 以DataFrame的index为基准对齐设置某一列的值
    
    pandas和Orca的内存表可以这样设置一列，而Orca的分区表不支持：
    
    ```Python
    >>> import pandas as pd
    >>> import orca
    >>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)

    >>> df = pd.DataFrame({"a": [1,2,3]}, index=[0,1,2])
    >>> df
    # output
    a
    0  1
    1  2
    2  3

    >>> ps = pd.Series([10,20,30], index=[0,2,3])
    >>> os = orca.Series([10,20,30], index=[0,2,3])
    >>> os
    # output
    0    10
    2    20
    3    30
    dtype: int64

    >>> df["b"] = s
    >>> df
    # output
    a     b
    0  1  10.0
    1  2   NaN
    2  3  20.0
    ```

  以上列出的差异仅供参考，也欢迎Orca的使用着和贡献着提出你的意见和建议，Orca将在我们的共同努力之下不断完善。