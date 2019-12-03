# Orca对分布式表的操作

DolphinDB是一个分布式时序数据库，并且内置了丰富的计算和分析功能。它可以将TB级的海量数据存储在多台物理机器上，充分利用CPU，对海量数据进行高性能分析计算。通过Orca，我们可以在python环境中使用与pandas语法相同的脚本对DolphinDB分布式数据库中的数据进行复杂高效的计算。本教程主要介绍Orca对DolphinDB分布式表的操作。

 - [1 读取分布式表](#1-读取分布式表)    
 - [2 查询和计算](#2-查询和计算)
     - [2.1 取前n条记录](#21-取前n条记录)        
     - [2.2 排序](#22-排序)        
     - [2.3 按照条件查询](#23-按照条件查询)        
     - [2.4 groupby分组查询](#24-groupby分组查询)        
     - [2.5 resample重采样](#25-resample重采样)        
     - [2.6 rolling移动窗口](#26-rolling移动窗口)        
     - [2.7 数据连接](#27-数据连接)
 - [3 把Orca dataframe追加到dfs表](#3-把dataframe追加到dfs表)
 - [4 小结](#4-小结)


本示例使用的是DolphinDB单机模式。首先，创建本教程的示例数据库dfs://orca_stock 。创建数据库的DolphinDB脚本如下所示：

```
login("admin","123456")
if(existsDatabase("dfs://orca_stock")){
	dropDatabase("dfs://orca_stock")
}
dates=2019.01.01..2019.01.31
syms="A"+string(1..30)
sym_range=cutPoints(syms,3)
db1=database("",VALUE,dates)
db2=database("",RANGE,sym_range)
db=database("dfs://orca_stock",COMPO,[db1,db2])
n=10000000
datetimes=2019.01.01T00:00:00..2019.01.31T23:59:59
t=table(rand(datetimes,n) as trade_time,rand(syms,n) as sym,rand(1000,n) as qty,rand(500.0,n) as price)
trades=db.createPartitionedTable(t,`trades,`trade_time`sym).append!(t)

n=200000
datetimes=2019.01.01T00:00:00..2019.01.02T23:59:59
syms="A"+string(1..30)
t2=table(rand(datetimes,n) as trade_time,rand(syms,n) as sym,rand(500.0,n) as bid,rand(500.0,n) as offer)
quotes=db.createPartitionedTable(t2,`quotes,`trade_time`sym).append!(t2)

syms="A"+string(1..30)
t3=table(syms as sym,rand(0 1,30) as type)
infos=db.createTable(t3,`infos).append!(t3)
```

>注意：需要在DolphinDB客户端或通过DolphinDB Python API创建分布式表，不能直接在Orca创建分布式表。

在Orca中通过`connect`函数连接到DolphinDB服务器：

```python
>>> import dolphindb.orca
>>> orca.connect("localhost",8848,"admin","123456")
```

用户需要根据实际情况修改IP地址和端口号。

## 1 读取分布式表

Orca通过`read_table`函数读取分布式表，返回的结果是Orca DataFrame。例如：读取示例数据库dfs://orca_stock 中的表trades：

```python
>>> trades = orca.read_table('dfs://orca_stock','trades')
>>> type(trades)
orca.core.frame.DataFrame
```

查看trades的列名：

```python
>>> trades.columns
Index(['trade_time', 'sym', 'qty', 'price'], dtype='object')
```

查看trades各列的数据类型：

```python
>>> trades.dtypes
trade_time    datetime64[s]
sym                  object
qty                   int32
price               float64
dtype: object
```

查看trades的行数：

```python
>>> len(trades)
10000000
```

DolphinDB分布式表对应的Orca DataFrame只存储元数据，包括表名、数据的列名等信息。由于分布式表不是连续存储，各个分区之间没有严格的顺序关系，因此分布式表对应的DataFrame没有RangeIndex的概念。如果需要设置index，可以使用`set_index`函数。例如，把trades中的trade_time设置为index：

```python
>>> trades.set_index('trade_time')
```

如果要将index列转换为数据列，可以用`reset_index`函数。

```python
>>> trades.reset_index()
```

## 2 查询和计算

Orca采用惰性求值，某些计算不会立即在服务端计算，而是转换为一个中间表达式，直到真正需要时才发生计算。如果用户需要立即触发计算，可以调用`compute`函数。

> 注意，示例数据库dfs://orca_stock 中的数据是随机生成的，因此用户的运行结果会与本章中的结果有所差异。

### 2.1 取前n条记录

`head`函数可以查询前n条记录，默认取前5条。例如，取trades的前5条记录：

```python
>>> trades.head()
           trade_time  sym  qty       price
0 2019-01-01 18:04:33  A16  855  482.526769
1 2019-01-01 13:57:38  A12  244   61.675293
2 2019-01-01 23:58:15  A10   36  297.623295
3 2019-01-01 23:02:43  A16  426  109.041012
4 2019-01-01 04:33:53   A1  472   75.778951
```

### 2.2 排序

`sort_values`方法可以根据某列排序。例如，trades按照price降序排序，取前5条记录：

```python
>>> trades.sort_values(by='price', ascending=False).head()
           trade_time  sym  qty       price
0 2019-01-03 12:56:09  A22  861  499.999998
1 2019-01-18 17:25:21  A19   95  499.999963
2 2019-01-30 02:18:48  A30  114  499.999949
3 2019-01-23 08:31:56   A3  926  499.999926
4 2019-01-20 03:36:53   A3  719  499.999892
```

按照多列排序：

```python
>>> trades.sort_values(by=['qty','trade_time'], ascending=False).head()
           trade_time  sym  qty       price
0 2019-01-31 23:58:50  A24  999  359.887697
1 2019-01-31 23:57:26   A3  999  420.156175
2 2019-01-31 23:56:34   A2  999  455.228435
3 2019-01-31 23:52:58   A6  999  210.819227
4 2019-01-31 23:45:17  A14  999  310.813216
```

### 2.3 按照条件查询

Orca支持按照单个或多个条件多虑查询。例如，

查询trades中2019年1月2日的数据：

```python
>>> tmp = trades[trades.trade_time.dt.date == "2019.01.01"]
>>> tmp.head()
           trade_time sym  qty       price
0 2019-01-01 00:32:21  A2  139  383.971293
1 2019-01-01 21:19:09  A2  263  100.932553
2 2019-01-01 18:50:48  A2  890  335.614454
3 2019-01-01 23:29:16  A2  858  469.223992
4 2019-01-01 09:58:51  A2  883  235.753424
```

查询trades中2019年1月30日，股票代码为A2的数据：

```python
>>> tmp = trades[(trades.trade_time.dt.date == '2019.01.30') & (trades.sym == 'A2')]
>>> tmp.head()
           trade_time sym  qty       price
0 2019-01-30 04:41:56  A2  880  428.552654
1 2019-01-30 14:13:53  A2  512  488.826978
2 2019-01-30 14:31:28  A2  536  478.578219
3 2019-01-30 04:09:41  A2  709  255.435903
4 2019-01-30 13:18:50  A2  355  404.782260
```

### 2.4 groupby分组查询

`groupby`函数用于分组聚合。以下函数都可以用于groupby对象：

|函数|描述|
|:---|:----|
|count|返回非NULL元素的个数|
|sum|求和|
|mean|均值|
|min|最小值|
|max|最大值|
|mode|众数|
|abs|绝对值|
|prod|乘积|
|std|标准差|
|var|方差|
|sem|平均值的标准误差|
|skew|倾斜度|
|kurtosis|峰度|
|cumsum|累积求和|
|cumprod|累积乘积|
|cummax|累积最大值|
|cummin|累积最小值|

计算trades中每天的记录数：

```python
>>> trades.groupby(trades.trade_time.dt.date)['sym'].count()
trade_time
2019-01-01    322573
2019-01-02    322662
2019-01-03    323116
2019-01-04    322436
2019-01-05    322156
2019-01-06    324191
2019-01-07    321879
2019-01-08    323319
2019-01-09    322262
2019-01-10    322585
2019-01-11    322986
2019-01-12    322839
2019-01-13    322302
2019-01-14    322032
2019-01-15    322409
2019-01-16    321810
2019-01-17    321566
2019-01-18    323651
2019-01-19    323463
2019-01-20    322675
2019-01-21    322845
2019-01-22    322931
2019-01-23    322598
2019-01-24    322404
2019-01-25    322454
2019-01-26    321760
2019-01-27    321955
2019-01-28    322013
2019-01-29    322745
2019-01-30    322193
2019-01-31    323190
dtype: int64
```

计算trades中每天每只股票的记录数：

```python
>>> trades.groupby([trades.trade_time.dt.date,'sym'])['price'].count()
trade_time  sym
2019-01-01  A1     10638
            A10    10747
            A11    10709
            A12    10715
            A13    10914
                   ...  
2019-01-31  A5     10717
            A6     10934
            A7     10963
            A8     10907
            A9     10815
Length: 930, dtype: int64
```

Orca支持通过agg一次应用多个聚合函数。和pandas不同，Orca在agg中使用字符串来表示要调用的聚合函数。例如，对计算trades中每天价格的最大值、最小值和均值：

```python
>>> trades.groupby(trades.trade_time.dt.date)['price'].agg(["min","max","avg"])
               price                        
                 min         max         avg
trade_time                                  
2019-01-01  0.003263  499.999073  249.913612
2019-01-02  0.000468  499.999533  249.956874
2019-01-03  0.000054  499.999998  249.927257
2019-01-04  0.000252  499.999762  249.982737
2019-01-05  0.001907  499.999704  250.097487
2019-01-06  0.000318  499.999824  249.991605
2019-01-07  0.003196  499.999548  249.560505
2019-01-08  0.000216  499.996703  250.024405
2019-01-09  0.002635  499.998985  249.966446
2019-01-10  0.000725  499.996717  249.663324
2019-01-11  0.003140  499.998267  250.243786
2019-01-12  0.000105  499.998453  250.077061
2019-01-13  0.004297  499.999139  250.097489
2019-01-14  0.003510  499.999452  249.775830
2019-01-15  0.002501  499.999638  250.021218
2019-01-16  0.000451  499.998059  250.044059
2019-01-17  0.002359  499.998462  249.808932
2019-01-18  0.000104  499.999963  249.918651
2019-01-19  0.000999  499.998000  249.899495
2019-01-20  0.000489  499.999892  249.606668
2019-01-21  0.000729  499.999774  249.839876
2019-01-22  0.000834  499.999331  249.632037
2019-01-23  0.001982  499.999926  249.955031
2019-01-24  0.000323  499.993956  249.557851
2019-01-25  0.000978  499.999716  249.722053
2019-01-26  0.002582  499.998753  249.897519
2019-01-27  0.000547  499.999809  250.404666
2019-01-28  0.002729  499.998545  249.622289
2019-01-29  0.000487  499.999598  249.950167
2019-01-30  0.000811  499.999949  250.182493
2019-01-31  0.000801  499.999292  249.317517
```

Orca groupby支持过滤功能。和pandas不同，Orca中的过滤条件用字符串形式的表达式来表示，而不是lambda函数。

例如，返回trades中每天每只股票均价大于200，并且记录数大于11000的记录:

```python
>>> trades.groupby([trades.trade_time.dt.date,'sym'])['price'].filter("avg(price) > 200 and count(price) > 11000")
0        499.171179
1        375.553059
2        119.240890
3        370.198534
4          5.876941
            ...    
88416     37.872317
88417    373.259785
88418    435.154484
88419    436.163806
88420    428.455914
Length: 88421, dtype: float64
```

### 2.5 resample重采样

Orca支持`resample`函数，可以对常规时间序列数据重新采样和频率转换。目前，resample函数的参数如下：

|参数|说明|
|:----|:----|
|rule|DateOffset，可以是字符串或者是dateoffset对象|
|on|时间列，采用该列进行重采样|
|level|字符串或整数，对于MultiIndex，采用level指定的列进行重采样|

Orca支持pandas的所有dateoffset。具体可查看[https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects)。

例如，对trades中的数据重新采样，每3分钟计算一次：

```python
>>> trades.resample('3T', on='trade_time')['qty'].sum()
trade_time
2019-01-01 00:00:00    321063
2019-01-01 00:03:00    354917
2019-01-01 00:06:00    329419
2019-01-01 00:09:00    340880
2019-01-01 00:12:00    356612
                        ...  
2019-01-31 23:45:00    322829
2019-01-31 23:48:00    344753
2019-01-31 23:51:00    330959
2019-01-31 23:54:00    336712
2019-01-31 23:57:00    328730
Length: 14880, dtype: int64
```

如果trades设置了trade_time为index，也可以用以下方法重新采样：

```python
>>> trades.resample('3T', level='trade_time')['qty'].sum()
```

如果要用dateoffset函数生成的对象来表示dateoffset，需要先导入pandas的dateoffset。按3分钟重新采样也可以使用以下写法：

```
>>> from pandas.tseries.offsets import *
>>> ofst = Minute(n=3)
>>> trades.resample(ofst,on='trade_time')['qty'].sum()
```

### 2.6 rolling移动窗口

Orca提供了`rolling`函数，可以在移动窗口中做计算。目前，`rolling`函数的参数如下：

|参数|说明|
|:----|:----|
|window|整型，表示窗口的长度|
|on|字符串，根据该列来计算窗口|

以下函数可用于orca.DataFrame.rolling对象：

|函数|描述|
|:---|:---|
|count|非NULL元素的个数|
|sum|求和|
|var|方差|
|std|标准差|
|min|最小值|
|max|最大值|
|corr|相关性|
|covar|协方差|
|skew|倾斜度|
|kurtosis|峰度|

对于分布式表对应的DataFrame，在滑动窗口中计算时，是以分区为单位单独计算的，因此每个分区的计算结果的前window-1个值为空。例如，trades中2019.01.01和2019.01.02的数据在长度为3的滑动窗口中求price的和：

```python
>>> tmp = trades[(trades.trade_time.dt.date == '2019.01.01') | (trades.trade_time.dt.date == '2019.01.02')]
>>> re = tmp.rolling(window=3)['price'].sum()
0                 NaN
1                 NaN
2          792.386603
3          601.826312
4          444.858366
             ...     
646057    1281.099161
646058    1287.816045
646059     963.262163
646060     865.797011
646061     719.050068
Name: price, Length: 646062, dtype: float64

```

### 2.7 数据连接

Orca提供了连接DataFrame的功能。分布式表对应的DataFrame，既可以连接普通内存表对应的DataFrame，也可以连接分布式表对应的DataFrame。两个分布式表对应的DataFrame连接时必须同时满足以下条件：
- 两个分布式表在同一个数据库中
- 连接列必须包含所有分区列

Orca提供了`merge`和`join`函数。

`merge`函数支持以下参数：

|参数|说明|
|:---|:----|
|right|Orca DataFrame或Series|
|how|字符串，表示连接的类型，可以是left、right、outer和inner，默认值是inner|
|on|字符串，表示连接列|
|left_on|字符串，表示左表的连接列|
|right_on|字符串，表示右表的连接列|
|left_index|左表的索引|
|right_index|右表的索引|
|suffixes|字符串，表示重复列的后缀|

`join`函数是`merge`函数的特例，它的参数及含义与`merge`基本相同，只是`join`默认为左外连接，即how='left'。

例如，对trades和quotes进行内连接：

```python
>>> quotes = orca.read_table('dfs://orca_stock','quotes')
>>> trades.merge(right=quotes, left_on=['trade_time','sym'], right_on=['trade_time','sym'], how='inner')
               trade_time  sym  qty       price         bid       offer
0     2019-01-01 02:36:34  A15  273  186.144261  317.458480  155.361661
1     2019-01-01 05:37:59  A13  185  420.397500  248.447426  115.722893
2     2019-01-01 00:59:43  A10  751   89.801687  193.925714  144.345473
3     2019-01-01 21:58:36  A16  175  251.753495  116.810807  439.178207
4     2019-01-01 10:53:54  A16  532   71.733640  240.927647  388.718680
...                   ...  ...  ...         ...         ...         ...
25035 2019-01-02 03:59:51   A3  220   50.004418  107.905522  167.375994
25036 2019-01-02 17:54:01   A3  202  195.189216  134.463906  142.443428
25037 2019-01-02 16:57:50   A9  627   68.661644  440.421876  110.801070
25038 2019-01-02 10:27:43  A28  414  487.337282  169.081363  261.171073
25039 2019-01-02 17:02:51   A3  661  243.960836   92.999404   26.747609

[25040 rows x 6 columns]

```

使用`join`函数对trades和quotes进行左外连接：

```python
>>> trades.set_index(['trade_time','sym'], inplace=True)
>>> quotes.set_index(['trade_time','sym'], inplace=True)
>>> trades.join(quotes)
                         qty       price  bid  offer
trade_time          sym                             
2019-01-01 18:04:25 A14  435  378.595626  NaN    NaN
2019-01-01 20:38:47 A13  701  275.039372  NaN    NaN
2019-01-01 02:43:03 A16  787  138.751605  NaN    NaN
2019-01-01 20:32:42 A14  989  188.035335  NaN    NaN
2019-01-01 16:59:16 A13  847  118.071427  NaN    NaN
...                      ...         ...  ...    ...
2019-01-31 17:21:27 A30    3   49.855063  NaN    NaN
2019-01-31 13:49:01 A6   273  245.966115  NaN    NaN
2019-01-31 16:42:29 A7   548  197.814548  NaN    NaN
2019-01-31 03:42:11 A5   563  263.999224  NaN    NaN
2019-01-31 20:48:57 A9   809  318.420522  NaN    NaN

[10000481 rows x 4 columns]
```

## 3 把dataframe追加到dfs表

Orca提供了`append`函数，可以将Orca DataFrame追加到dfs表中。

`append`函数具有以下参数：

|参数|说明|
|:---|:---|
|other|要追加的DataFrame|
|ignore_index|布尔值，是否忽略索引。默认为False|
|verify_integrity|布尔值。默认为False|
|sort|布尔值，表示是否排序。默认为None|
|inplace|布尔值，表示是否插入到dfs表。默认为False|

例如，往dataframe追加到trades对应的分布式表：

```python
>>> import pandas as pd
>>> odf=orca.DataFrame({'trade_time':pd.date_range('20190101 12:30',periods=5,freq='T'),
                   'sym':['A1','A2','A3','A4','A5'],
                   'qty':[100,200,300,400,500],
                   'price':[100.5,263.1,254.9,215.1,245.6]})
>>> trades.append(odf,inplace=True)
>>> len(trades)
10000005
```

Orca扩展了append函数，支持inplace参数，即允许就地添加数据。如果inplace为False，表现和pandas相同。分布式表中的内容会复制到内存中，此时trades对应的只是一个内存表，odf中的内容只追加到内存表，没有真正地追加到dfs表。

## 4 小结

对于分布式表，目前Orca还具有一些功能上的限制，例如分区表对应的DataFrame没有RangeIndex的概念、一些函数不支持在分布式表上使用以及修改表中数据的限制等。具体请参考[Orca快速入门指导](https://2xdb.net/dolphindb/orca/blob/master/user_guide.md)。

