# orca教程：数据加载

本文介绍在orca中加载数据的方法。

- [建立数据库连接](#1-建立数据库连接)
- [导入数据](#2-导入数据)
   - [`read_csv`函数](#21-read_csv函数)
   - [`read_table`函数](#22-read_table函数)
   - [`read_shared_table`函数](#23-read_shared_table函数)
   - [`from_pandas`函数](#24-from_pandas函数)
- [对其它格式文件的支持](#3-对其它格式文件的支持)

## 1 建立数据库连接

在orca中通过`connect`函数连接到DolphinDB服务器：
```python
>>> import dolphindb.orca as orca
>>> orca.connect("localhost", 8848, "admin", "123456")     # orca.connect(HOST, PORT, USERNAME, PASSWORD)
```

## 2 导入数据

下面的教程使用了一个数据文件：[quotes.csv](./data/quotes.csv)。

### 2.1 `read_csv`函数

`read_csv`函数engine参数的取值可以是'python'或'dolphindb'，默认值为'dolphindb'。当取值为'dolphindb'时，`read_csv`函数会在DolphinDB服务器目录下寻找要导入的数据文件。当取值为'python'时，`read_csv`函数会调用pandas的`read_csv`函数，在Python客户端的目录下寻找要导入的数据文件。

当engine参数设置为'dolphindb'时，orca的`read_csv`函数支持的参数如下：

|参数|说明|
|:--|:--|
|path|文件路径|
|sep|分隔符|
|delimiter|分隔符|
|names|指定列名|
|index_col|指定作为index的列|
|engine|进行导入的引擎|
|usecols|指定要导入的列|
|squeeze|当数据文件只有一行或者一列时，是否将DataFrame压缩成Series|
|prefix|给每列加上的前缀字符串|
|dtype|指定数据类型导入|
|partitioned|是否允许以分区的方式导入数据|
|db_handle|要导入的数据库路径|
|table_name|要导入的表名|
|partition_columns|进行分区的列名|

下面详细介绍orca与pandas用法有所不同的几个参数。

- dtype参数

   orca在导入csv的时候会自动识别要导入文件的数据类型，支持各种通用时间格式。用户也可以通过dtype参数来指定数据类型。

   orca的`read_csv`函数不仅支持指定NumPy的各种数据类型(np.bool, np.int8, np.float32, etc.)，还支持支持以字符串的方式指定DolphinDB的提供的所有[数据类型](https://www.dolphindb.cn/cn/help/DataType.html)。
      
   例如：
      
   ```python
   YOUR_DIR = "/dolphindb/database"
   df = orca.read_csv(YOUR_DIR+"/quotes.csv", dtype={"TIME":"NANOTIMESTAMP", "Exchange":"SYMBOL", "SYMBOL":"SYMBOL", "Bid_Price":np.float64, "Bid_Size":np.int32, "Offer_Price":np.float64, "Offer_Size":np.int32})
   ```

- partitioned参数 

   bool类型，默认为True。该参数设为True时，在数据规模达到一定程度时，会将数据导入为分区内存表，如果设为False，会将csv文件导入为不分区的普通内存表。

   > 请注意：orca的分区分区表与orca的内存表相比，有诸多不同之处，具体见[orca分区表的特殊差异](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/api_differences.md#82-orca%E5%88%86%E5%8C%BA%E8%A1%A8%E7%9A%84%E7%89%B9%E6%AE%8A%E5%B7%AE%E5%BC%82)。若您的数据量不是很大，且在使用orca时对orca与pandas的一致性要求更高，请尽量不要将数据以分区的方式导入。若您数据量极大，对性能要求极高，则建议您采用分区方式导入数据。

- db_handle, table_name以及partition_columns参数

   调用orca的`read_csv`函数时若指定db_handle, table_name和partition_columns这3个参数，相当于调用了DolphinDB的[`loadTextEx`](https://www.dolphindb.cn/cn/help/loadTextEx.html)函数，直接将数据导入DolphinDB的分区表。
   
   关于`read_csv`函数将数据导入分区表的详细介绍请参考[orca的分区表](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/api_differences.md#81-orca%E7%9A%84%E5%88%86%E5%8C%BA%E8%A1%A8)。

#### 2.1.1 导入到内存表 

- 导入为普通内存表

   若`read_csv`函数的partitioned参数设定为False，数据会导入为普通内存表（没有分区）。下面的例子中'YOUR_DIR'为数据文件存放的路径。
   ```python
   >>> YOUR_DIR = "dolphindb/database" 
   df = orca.read_csv(YOUR_DIR + "/quotes.csv", partitioned=False)
   ```
   
- 导入为分区内存表

   若`read_csv`函数的partitioned参数设定为True（默认值），会并行导入数据，产生分区内存表。并行导入的速度快，但是数据加载过程中的内存占用是普通内存表的两倍。
   ```python
   >>> df = orca.read_csv(YOUR_DIR + "/quotes.csv")
   ```

#### 2.1.2 导入到磁盘表

DolphinDB的分区表可以保存为本地磁盘分区表，也可以保存为分布式表(dfs table)。本地磁盘分区表与分布式表的区别在于分布式表的数据库路径以"dfs://"开头，而磁盘分区表的数据库路径是本地路径。

**示例**

首先创建一个本地磁盘分区数据库。下面的脚本中，'YOUR_DIR'为保存磁盘数据库的路径。
          
```python
>>> s = orca.default_session()
>>> YOUR_DIR = "/dolphindb/database" 
>>> create_onDiskPartitioned_database = """
dbPath="{YOUR_DIR}" + "/DB10"
login('admin', '123456')
if(existsDatabase(dbPath))
  dropDatabase(dbPath)
db=database(dbPath, RANGE, datehour(2017.01.01 00:00:00+(0..24)*3600))
""".format(YOUR_DIR=YOUR_DIR)
>>> s.run(create_onDiskPartitioned_database)
```

在Python客户端中调用orca的`read_csv`函数，指定数据库db_handle为磁盘分区数据库YOUR_DIR + "/DB1"，指定表名table_name为"quotes"和进行分区的列partition_columns为"time"，将数据导入到DolphinDB的本地磁盘分区表，并返回一个表示DolphinDB数据表的对象df，用于后续计算。

```python
>>> df = orca.read_csv(path=YOUR_DIR+"/quotes.csv", dtype={"TIME":"NANOTIMESTAMP", "Exchange":"SYMBOL", "SYMBOL":"SYMBOL", "Bid_Price":np.float64, "Bid_Size":np.int32, "Offer_Price":np.float64, "Offer_Size":np.int32}, db_handle=YOUR_DIR + "/DB10", table_name="quotes", partition_columns="time")
>>> df
<'dolphindb.orca.core.frame.DataFrame' object representing a column in a DolphinDB segmented table>
>>> df.head()

                        time Exchange Symbol  Bid_Price  Bid_Size  \
0 2017-01-01 04:40:11.686699        T   AAPL       0.00         0   
1 2017-01-01 06:42:50.247631        P   AAPL      26.70        10   
2 2017-01-01 07:00:12.194786        P   AAPL      26.75         5   
3 2017-01-01 07:15:03.578071        P   AAPL      26.70        10   
4 2017-01-01 07:59:39.606882        K   AAPL      26.90         1   
   Offer_Price  Offer_Size  
0        27.42           1  
1        27.47           1  
2        27.47           1  
3        27.47           1  
4         0.00           0  
   ```

   上述脚本中的`defalut_session`是之前通过`connect`函数创建的会话。在Python端，我们可以通过这个会话与DolphinDB服务端进行交互。

   > 请注意：在通过`read_csv`函数导入数据之前，需要确保在DolphinDB服务器上已经创建了指定的数据库。若表已存在则追加数据，若表不存在则创建表并导入数据。

#### 2.1.3 导入到分布式表
   
   调用`read_csv`函数时，若指定了db_handle参数为dfs数据库路径，则数据将直接导入到DolphinDB的dfs数据库中。

   **示例**  

   请注意只有启用enableDFS=1的集群环境或者DolphinDB单例模式才能使用分布式表。分布式数据库路径均以"dfs://"开头。
          
   ```python
   >>> s = orca.default_session()
   >>> YOUR_DIR = "/dolphindb/database" 
   >>> create_dfs_database = """
   dbPath="dfs://demoDB"
   login('admin', '123456')
   if(existsDatabase(dbPath))
      dropDatabase(dbPath)
   db=database(dbPath, RANGE, datehour(2017.01.01 00:00:00+(0..24)*3600))
   """
   >>> s.run(create_dfs_database)
   ```
   
   调用orca的`read_csv`函数，指定db_handle为分布式数据库"dfs://demoDB"，table_name为"quotes"，分区列partition_columns为"time"，将数据导入分布式表:

   ```python
   >>> df = orca.read_csv(path=YOUR_DIR + "/quotes.csv", dtype={"Exchange": "SYMBOL", "SYMBOL": "SYMBOL"}, db_handle="dfs://demoDB", table_name="quotes", partition_columns="time")
   >>> df
   <'dolphindb.orca.core.frame.DataFrame' object representing a column in a DolphinDB segmented table>
   >>> df.head()
   
                           time Exchange Symbol  Bid_Price  Bid_Size  \
   0 2017-01-01 04:40:11.686699        T   AAPL       0.00         0   
   1 2017-01-01 06:42:50.247631        P   AAPL      26.70        10   
   2 2017-01-01 07:00:12.194786        P   AAPL      26.75         5   
   3 2017-01-01 07:15:03.578071        P   AAPL      26.70        10   
   4 2017-01-01 07:59:39.606882        K   AAPL      26.90         1   
      Offer_Price  Offer_Size  
   0        27.42           1  
   1        27.47           1  
   2        27.47           1  
   3        27.47           1  
   4         0.00           0  
   ```

### 2.2 `read_table`函数

orca提供`read_table`函数，可用于加载磁盘表、磁盘分区表和分布式表。

`read_table`函数的参数如下:

|参数|用法|
|:--|:--|
|database|数据库名称|
|table|表名|
|partition|需要导入的分区,可选参数|

- 加载DolphinDB的磁盘表

   首先在DolphinDB服务端创建一个本地磁盘表：

   ```python
   >>> s = orca.default_session()
   >>> YOUR_DIR = "/dolphindb/database" 
   >>> create_onDisk_database="""
   saveTable("{YOUR_DIR}"+"/demoOnDiskDB", table(2017.01.01..2017.01.10 as date, rand(10.0,10) as prices), "quotes")
   """.format(YOUR_DIR=YOUR_DIR)
   >>> s.run(create_onDisk_database)
   ```

   通过`read_table`函数加载磁盘表：

   ```python
   >>> df = orca.read_table(YOUR_DIR + "/demoOnDiskDB", "quotes")
   >>> df.head()

         date    prices
   0 2017-01-01  8.065677
   1 2017-01-02  2.969041
   2 2017-01-03  3.688191
   3 2017-01-04  4.773723
   4 2017-01-05  5.567130
   ```

- 加载DolphinDB的磁盘分区表

   对于已经在DolphinDB上创建的数据表，可以通过`read_table`函数直接加载。例如，加载[2.1.2](#212-导入磁盘表)小节中创建的磁盘分区表：
   ```python
   >>> df = orca.read_table(YOUR_DIR + "/DB10", "quotes")
   ```

- 加载DolphinDB的分布式表

   加载[2.1.3](#213-导入分布式表)小节中创建的分布式表：
   ```python
   >>> df = orca.read_table("dfs://demoDB", "quotes")
   ```

### 2.3 `read_shared_table`函数

`read_shared_table`函数可读取一个DolphinDB的共享表（内存表或流表），返回一个orca的DataFrame。

在DolphinDB中将表共享：
```dolphindb
t = table(1..3 as id, take(`a`b, 3) as sym)
share t as sharedT
```

orca中读取共享表：
```python
>>> df = orca.read_shared_table("sharedT")
>>> df
   id sym
0   1   a
1   2   b
2   3   a
```

调用`append`函数，并指定参数inplace=True，向其中插入数据：
```python
>>> df.append(orca.DataFrame({"id": [4], "sym": ["b"]}), inplace=True)
>>> df
   id sym
0   1   a
1   2   b
2   3   a
3   4   b
```

在DolphinDB中访问共享表，可见数据已经成功插入：
```dolphindb
> sharedT;
id sym
-- ---
1  a
2  b
3  a
4  b
```

### 2.4 `from_pandas`函数

orca的`from_pandas`函数可将一个pandas的DataFrame转化为orca的DataFrame。
```python
>>> import pandas as pd
>>> import numpy as np

>>> pdf = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),  columns=['a', 'b', 'c'])
>>> odf = orca.from_pandas(pdf)
```

## 3 对其它格式文件的支持

对于其它数据格式的导入，orca也提供了与pandas类似的接口。这些方法包括：`read_pickle`, `read_fwf`, `read_msgpack`, `read_clipboard`, `read_excel`, `read_json`, `json_normalize`,`build_table_schema`, `read_html`, `read_hdf`, `read_feather`, `read_parquet`, `read_sas`, `read_sql_table`, `read_sql_query`, `read_sql`, `read_gbq`, `read_stata`。