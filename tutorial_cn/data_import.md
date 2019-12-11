# Orca数据加载教程

本文介绍在Orca中加载数据的方法。

- [建立数据库连接](#1-建立数据库连接)
- [导入数据](#2-导入数据)
   - [`read_csv`函数](#21-read_csv函数)
   - [`read_table`函数](#22-read_table)
   - [`from_pandas`函数](#23-from_pandas)
- [对其它格式文件的支持](#3-对其它格式文件的支持)

## 1 建立数据库连接

在Orca中通过`connect`函数连接到DolphinDB服务器：

```Python
>>> import dolphindb.orca as orca
>>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)
```

## 2 导入数据

下面的教程使用了一个数据文件：[quotes.csv](./data/quotes.csv)。

### 2.1 `read_csv`函数

Orca提供`read_csv`函数，用于导入数据集。需要说明的是，Orca的`read_csv`函数的engine参数的取值可以是{‘c’, ‘python’, ‘dolphindb’}，且该参数默认取值为‘dolphindb’。当取值为‘dolphindb’时，`read_csv`函数会在DolphinDB服务器目录下寻找要导入的数据文件。当取值为‘python’或‘c’时，`read_csv`函数会在python客户端的目录下寻找要导入的数据文件。

> 请注意，当engine参数设置为‘python’或者‘c’时，Orca的`read_csv`函数相当于调用了pandas的`read_csv`函数进行导入。本节是基于engine参数取值为‘dolphindb’的前提下对Orca的`read_csv`函数进行讲解。

当engine参数设置为‘dolphindb’时，Orca的`read_csv`函数目前支持的参数如下：

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

下面详细介绍Orca与pandas实现有所不同的几个参数。

- dtype参数

   Orca在导入csv的时候会自动识别要导入文件的数据类型，支持各种通用时间格式。用户也可以通过dtype参数来强制指定数据类型。

   需要说明的是，Orca的`read_csv`函数不仅支持指定各种numpy的数据类型（np.bool, np.int8, np.float32, etc.），还支持支持以字符串的方式指定DolphinDB的提供的所有[数据类型](https://www.dolphindb.cn/cn/help/DataType.html)，包括所有时间类型和字符串类型。
      
   例如：
      
   ```Python
   dfcsv = orca.read_csv("DATA_DIR/quotes.csv", dtype={"TIME": "NANOTIMESTAMP", "Exchange": "SYMBOL", "SYMBOL": "SYMBOL", "Bid_Price": np.float64, "Bid_Size": np.int32， "Offer_Price": np.float64, "Offer_Size": np.int32})
   ```

- partitioned参数 

   bool类型，默认为True。该参数为True时，在数据规模达到一定程度时，会将数据导入为分区内存表，如果设置为False，会直接将csv导入为未经分区的DolphinDB普通内存表。

   > 请注意：Orca的分区表与Orca的内存表相比，在操作时也存在许多差异，具体见[Orca分区表的特殊差异]https://github.com/dolphindb/Orca/blob/master/tutorial_cn/api_differences.md#82-orca%E5%88%86%E5%8C%BA%E8%A1%A8%E7%9A%84%E7%89%B9%E6%AE%8A%E5%B7%AE%E5%BC%82)。若您的数据量不是很大，且在使用Orca时对Orca与pandas的一致性要求更高，请尽量不要将数据以分区的方式导入。若您数据量极大，对性能要求极高，则建议您采用分区方式导入数据。

- db_handle, table_name以及partition_columns参数

   Orca的`read_csv`还支持db_handle, table_name和partition_columns这3个参数，这些参数用于在导入数据的时通过指定DolphinDB的数据库和表等相关信息，将数据导入到DolphinDB的分区表。

   DolphinDB支持通过多种方式[将数据导入DolphinDB数据库](https://github.com/dolphindb/Tutorials_CN/blob/master/import_data.md)，Orca在调用`read_csv`函数时指定db_handle, table_name以及partition_columns参数，本质上是调用DolphinDB的[loadTextEx](https://www.dolphindb.cn/cn/help/loadTextEx.html)函数，通过这种方式，我们可以直接将数据直接导入DolphinDB的分区表。

#### 2.1.1 导入到内存表 

- 导入为内存分区表

   直接调用`read_csv`函数，数据会并行导入。由于采用并行导入，导入速度快，但是对内存占用是普通表的两倍。下面的例子中'DATA_DIR'为数据文件存放的路径。

   ```Python
   >>> DATA_DIR = "dolphindb/database" # e.g. data_dir
   >>> df = orca.read_csv(DATA_DIR + "/quotes.csv")
   >>> df.head()
   # output

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

- 导入为普通内存表

   partitioned参数取值为False，导入为普通内存表。导入对内存要求低，但是计算速度略低于上面的导入方式：

   ```Python
   df = orca.read_csv(DATA_DIR + "/quotes.csv"， partitioned=False)
   ```

#### 2.1.2 导入到磁盘表

DolphinDB的分区表可以保存在本地磁盘，也可以保存在dfs上，磁盘分区表与分布式表的区别就在于分布式表的数据库路径以"dfs://"开头，而磁盘分区表的数据库路径是本地路径。

**示例**

我们在DolphinDB服务端创建一个磁盘分区表，下面的脚本中，'YOUR_DIR'为保存磁盘数据库的路径：
          
```dolphindb
dbPath=YOUR_DIR + "/demoOnDiskPartitionedDB"
login('admin', '123456')
if(existsDatabase(dbPath))
   dropDatabase(dbPath)
db=database(dbPath, RANGE, datehour(2017.01.01 00:00:00+(0..24)*3600))
```

> 请注意：以上两段脚本需要在DolphinDB服务端执行，在Python客户端中则可以通过DolphinDB Python API执行脚本。

在Python客户端中调用Orca的`read_csv`函数，指定数据库db_handle为磁盘分区数据库YOUR_DIR + "/demoOnDiskPartitionedDB"，指定表名table_name为"quotes"和进行分区的列partition_columns为"time"，将数据导入到DolphinDB的磁盘分区表，并返回一个表示DolphinDB数据表的对象给df，用于后续计算。

```Python
>>> df = orca.read_csv(path=DATA_DIR+"/quotes.csv", dtype={"Exchange": "SYMBOL", "SYMBOL": "SYMBOL"}, db_handle=YOUR_DIR + "/demoOnDiskPartitionedDB", table_name="quotes", partition_columns="time")
>>> df
# output
<'dolphindb.orca.core.frame.DataFrame' object representing a column in a DolphinDB segmented table>
>>> df.head()
# output
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

   将上述过程整合成的Python中可执行的脚本如下：
     
   ```Python
   >>> s = orca.default_session()
   >>> DATA_DIR = "/dolphindb/database" # e.g. data_dir
   >>> YOUR_DIR = "/dolphindb/database" # e.g. database_dir
   >>> create_onDiskPartitioned_database = """
   dbPath="{YOUR_DIR}" + "/demoOnDiskPartitionedDB"
   login('admin', '123456')
   if(existsDatabase(dbPath))
      dropDatabase(dbPath)
   db=database(dbPath, RANGE, datehour(2017.01.01 00:00:00+(0..24)*3600))
   """.format(YOUR_DIR=YOUR_DIR)
   >>> s.run(create_onDiskPartitioned_database)
   >>> df = orca.read_csv(path=DATA_DIR+"/quotes.csv", dtype={"Exchange": "SYMBOL", "SYMBOL": "SYMBOL"}, db_handle=YOUR_DIR + "/demoOnDiskPartitionedDB", table_name="quotes", partition_columns="time")
   ```      

   上述脚本中，我们使用的`defalut_session`实际上就是通过`orca.connect`函数创建的会话，在Python端，我们可以通过这个会话与DolphinDB服务端进行交互。关于更多功能，请参见[DolphinDB Python API](https://github.com/dolphindb/python3_api_experimental)。

   > 请注意：在通过`read_csv`函数指定数据库导入数据之前，需要确保在DolphinDB服务器上已经创建了对应的数据库。`read_csv`函数根据指定的数据库，表名和分区字段导入数据到DolphinDB数据库中，若表存在则追加数据，若表不存在则创建表并且导入数据。

#### 2.1.3 导入到分布式表
   
   `read_csv`函数若指定db_handle参数为dfs数据库路径，则数据将直接导入到DolphinDB的dfs数据库中。

   **示例**  

   请注意只有启用enableDFS=1的集群环境或者DolphinDB单例模式才能使用分布式表。   
     
   与磁盘分区表类似，首先需要在DolphinDB服务器上创建分布式表，只需要将数据库路径改为"dfs://"开头的字符串即可。
          
   ```dolphindb
   dbPath="dfs://demoDB"
   login('admin', '123456')
   if(existsDatabase(dbPath))
      dropDatabase(dbPath)
   db=database(dbPath, RANGE, datehour(2017.01.01 00:00:00+(0..24)*3600))
   ```

   在Python客户端中调用Orca的`read_csv`函数，指定数据库db_handle为分布式数据库"dfs://demoDB"，指定表名table_name为"quotes"和进行分区的列partition_columns为"time"，将数据导入到DolphinDB的分布式表。

   ```Python
   >>> df = orca.read_csv(path=DATA_DIR+"/quotes.csv", dtype={"Exchange": "SYMBOL", "SYMBOL": "SYMBOL"}, db_handle="dfs://demoDB", table_name="quotes", partition_columns="time")
   >>> df
   # output
   <'dolphindb.orca.core.frame.DataFrame' object representing a column in a DolphinDB segmented table>
   >>> df.head()
   # output
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

   将上述过程整合成的Python中可执行的脚本如下：
     
   ```Python
   >>> s = orca.default_session()
   >>> DATA_DIR = "/dolphindb/database" # e.g. data_dir
   >>> create_dfs_database = """
   dbPath="dfs://demoDB"
   login('admin', '123456')
   if(existsDatabase(dbPath))
      dropDatabase(dbPath)
   db=database(dbPath, RANGE, datehour(2017.01.01 00:00:00+(0..24)*3600))
   """
   >>> s.run(create_dfs_database)
   >>> df = orca.read_csv(path=DATA_DIR+"/quotes.csv", dtype={"Exchange": "SYMBOL", "SYMBOL": "SYMBOL"}, db_handle="dfs://demoDB", table_name="quotes", partition_columns="time")
   ```      

### 2.2 `read_table`函数

Orca提供`read_table`函数，通过该函数指定DolphinDB数据库和表名来加载DolphinDB数据表的数据，可以用于加载DolphinDB的磁盘表、磁盘分区表和分布式表。若您已在DolphinDB中创建了数据库和表，则可以直接在Orca中调用该函数加载存放在DolphinDB服务端中的数据，`read_table`函数支持的参数如下:

|参数|用法|
|:--|:--|
|database|数据库名称|
|table|表名|
|partition|需要导入的分区,可选参数|

- 加载DolphinDB的磁盘表

   `read_table`函数可以用于加载DolphinDB的磁盘表，首先在DolphinDB服务端创建一个本地磁盘表：

   ```Python
   >>> s = orca.default_session()
   >>> YOUR_DIR = "/dolphindb/database" # e.g. database_dir
   >>> create_onDisk_database="""
   saveTable("{YOUR_DIR}"+"/demoOnDiskDB", table(2017.01.01..2017.01.10 as date, rand(10.0,10) as prices), "quotes")
   """.format(YOUR_DIR=YOUR_DIR)
   >>> s.run(create_onDisk_database)
   ```

   通过`read_table`函数加载磁盘表：

   ```Python
   >>> df = orca.read_table(YOUR_DIR + "/demoOnDiskDB", "quotes")
   >>> df.head()
   # output
         date    prices
   0 2017-01-01  8.065677
   1 2017-01-02  2.969041
   2 2017-01-03  3.688191
   3 2017-01-04  4.773723
   4 2017-01-05  5.567130
   ```

   > 请注意: `read_table`函数要求所要导入的数据库和表在DolphinDB服务器上已经存在，若只存在数据库和没有创建表，则不能将数据成功导入到Python中。

- 加载DolphinDB的磁盘分区表

   对于已经在DolphinDB上创建的数据表，可以通过`read_table`函数直接加载。例如，加载[2.1.2](#212-导入磁盘表)小节中创建的磁盘分区表：

   ```Python
   >>> df = orca.read_table(YOUR_DIR + "/demoOnDiskPartitionedDB", "quotes")
   ```

- 加载DolphinDB的分布式表

   分布式表同样可以通过`read_table`函数加载。例如，加载[2.1.3](#213-导入分布式表)小节中创建的分布式表：

   ```Python
   >>> df = orca.read_table("dfs://demoDB", "quotes")
   ```

### 2.3 `from_pandas`函数

Orca提供`from_pandas`函数，该函数接受一个pandas的DataFrame作为参数，返回一个Orca的DataFrame，通过这个方式，Orca可以直接加载原先存放在pandas的DataFrame中的数据。

```Python
>>> import pandas as pd
>>> import numpy as np

>>> pdf = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),  columns=['a', 'b', 'c'])
>>> odf = orca.from_pandas(pdf)
```

## 3 对其它格式文件的支持

对于其它数据格式的导入，Orca也提供了与pandas类似的接口。这些方法包括：`read_pickle`, `read_fwf`, `read_msgpack`, `read_clipboard`, `read_excel`, `read_json`, `json_normalize`,`build_table_schema`, `read_html`, `read_hdf`, `read_feather`, `read_parquet`, `read_sas`, `read_sql_table`, `read_sql_query`, `read_sql`, `read_gbq`, `read_stata`。