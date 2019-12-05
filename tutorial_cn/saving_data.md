# Orca写数据教程

Orca项目在DolphinDB之上实现了pandas API，使用户能更高效地分析处理海量数据。在数据存储方面，与pandas相比，Orca具备以下显著优势：

- 更灵活的选择

   Orca不仅能像pandas一样在内存中进行计算，将DatFrame中的数据导出到磁盘，也能随时将DataFrame的数据以及计算结果追加到DolphinDB的数据表中，为后续的数据查询、分析提供参考。

- 更优异的性能

   当数据量非常大而又需要保存数据时，在pandas中可以将整个DataFrame的数据保存到磁盘，在下一次运行Python程序时，用户再重新将磁盘上的数据加载到内存，这一做法无疑需要在导入和导出操作上耗费大量时间。而Orca对数据的存储与计算过程均进行了优化，用户只需在程序结束前将数据写入到DolphinDB数据表，在下一次运行Python程序时，用户无须重新将整个表的数据加载到内存，也可以立刻进行分析和计算操作。
   
本文将介绍如何通过Orca保存数据：

- [1 将数据导出到磁盘](#1-将数据导出到磁盘)
- [2 将数据保存到DolphinDB数据表](#2-将数据保存到dolphindb数据表)
  - [2.1 保存数据到Orca内存表](#21-保存数据到orca内存表)
  - [2.2 保存数据到Orca磁盘表](#22-保存数据到orca磁盘表)
  - [2.3 保存数据到Orca分布式表](#23-保存数据到orca分布式表)
- [3 小结](#3-小结)

### 1 将数据导出到磁盘

Orca的Series和DataFrame均支持`to_csv`，`to_excel`等将数据导出为固定格式的文件保存到指定路径的方法。下面对`to_csv`进行特殊说明。

- `to_csv`函数

   pandas的`to_csv`函数的engine参数的取值可以是‘c’或者‘python’，表示使用哪一种引擎进行导入。
   
   Orca的`to_csv`函数的engine参数的取值可以是{‘c’, ‘python’, ‘dolphindb’}，且该参数默认取值为‘dolphindb’。当取值为‘dolphindb’时，`to_csv`函数会将数据导出到DolphinDB服务器目录下，且只支持sep和append两个参数；当取值为‘python’或‘c’时，`to_csv`函数会将数据导出到python客户端的目录下，并支持pandas所支持的所有参数。
   
   **示例**

   调用`to_csv`函数导出数据，并通过`read_csv`函数再将数据导入。以下的脚本中'YOUR_DIR'表示用户保存csv文件的路径。由于数据是随机生成的，每次执行生成的表数据都不相同，下面的输出结果仅供参考。

   ```Python
   >>> YOUR_DIR = "/dolphindb/database" # e.g. data_dir
   >>> odf = orca.DataFrame({"type": np.random.choice(list("abcde"),10), "value": np.random.sample(10)*100})
   >>> odf.to_csv(path_or_buf=YOUR_DIR + "/demo.csv")
   >>> df1 = orca.read_csv(path=YOUR_DIR + "/demo.csv")
   >>> df1
   # output
     type      value
   0    c  93.697510
   1    c  64.533273
   2    e  11.699053
   3    c  46.758312
   4    d   0.262836
   5    e  30.315109
   6    a  72.641846
   7    e  60.980473
   8    c  89.597063
   9    d  25.223624
   ```

### 2 将数据保存到DolphinDB数据表

使用Orca的一个重要场景是，用户从其他数据库系统或是第三方Web API中取得数据后存入DolphinDB数据表中。本节将介绍通过Orca将取到的数据上传并保存到DolphinDB的数据表中。

Orca数据表按存储方式分为三种:

- 内存表：数据仅保存在内存中，存取速度最快，但是节点关闭后数据就不存在了。
- 本地磁盘表：数据保存在本地磁盘上。可以从磁盘加载到内存。
- 分布式表：数据存储在DolphinDB服务端，并未加载到内存，客户端只是获得了数据库和表的信息，通过DolphinDB的分布式计算引擎，仍然可以像本地表一样做统一查询。

下面以例子的形式解释这三种表的区别。

- 内存表

   可以通过`read_csv`函数导入或者通过`DataFrame`函数创建。
  
   - `read_csv`函数导入
    
      以[第1节](#1-将数据导出到磁盘)例子中的csv文件为例，像这样导入后能够直接访问表内数据的表，我们称之为Orca内存表。

      ```Python
      >>> df1 = orca.read_csv(path=YOUR_DIR + "/demo.csv")
      >>> df1
      # output
        type      value
      0    c  93.697510
      1    c  64.533273
      2    e  11.699053
      3    c  46.758312
      4    d   0.262836
      5    e  30.315109
      6    a  72.641846
      7    e  60.980473
      8    c  89.597063
      9    d  25.223624
      ```

   - `DataFrame`函数创建

      通过orca.DataFrame函数创建的内存表，也能够直接访问表内数据：

      ```Python
      >>> df = orca.DataFrame({"date":orca.date_range("20190101", periods=10),"price":np.random.sample(10)*100})
      >>> df
      # output
               date      price
      0 2019-01-01  35.218404
      1 2019-01-02  24.066378
      2 2019-01-03   6.336181
      3 2019-01-04  24.786319
      4 2019-01-05  35.021376
      5 2019-01-06  14.014935
      6 2019-01-07   7.454209
      7 2019-01-08  86.430214
      8 2019-01-09  80.033767
      9 2019-01-10  45.410883
      ```

- 磁盘表

   磁盘表分为本地磁盘表和磁盘分区表，其中本地磁盘表与内存表的区别就在于本地磁盘表是保存在磁盘上的内存表，不需要进行分区。而磁盘分区表则是保存在磁盘上的分区表，以下具体解释本地磁盘表。

   通过`read_table`函数可以在Orca中加载本地磁盘表。
   
   Orca提供`read_table`函数，通过该函数指定DolphinDB数据库和表名来加载DolphinDB数据表的数据，该函数支持的参数如下:

   |参数|用法|
   |:--|:--|
   |database|数据库名称|
   |table|表名|
   |partition|需要导入的分区,可选参数|

   > 请注意: `read_table`函数要求所要导入的数据库和表在DolphinDB服务器上已经存在，若只存在数据库和没有创建表，则不能将数据成功导入到Python中。

   从函数定义可以看出，`read_table`函数可以用于导入Orca的分区表,但是当导入的表是DolphinDB的磁盘表时，Orca会将表数据全都加载到内存，作为Orca内存表以供访问。
      
   **示例**

   假设DolphinDB Server上已有数据库和表如下，以下的脚本中'YOUR_DIR'表示用户保存磁盘表的路径。

   ```
   rows=10
   tdata=table(rand(`a`b`c`d`e, rows) as type, rand(100.0, rows) as value)
   saveTable(YOUR_DIR + "/testOnDiskDB", tdata, `tb1)
   ```

   脚本中创建的数据库的路径为YOUR_DIR + "/testOnDiskDB"，存储的表名为"tb1"。在Python客户端中，我们可以通过`read_table`函数将这个磁盘表表加载到内存中，存放在一个Orca的DataFrame对象里。
    
   ```Python
   >>> df = orca.read_table(YOUR_DIR + "/testOnDiskDB", "tb1")
   ```

   将上述过程整合成的Python中可执行的脚本如下：
      
   ```Python
   >>> s = orca.default_session()
   >>> data_dir = "/dolphindb/database" # e.g. data_dir
   >>> tableName = "tb1"
   >>> create_onDisk_table = """
   rows=10
   tdata=table(rand(`a`b`c`d`e, rows) as type, rand(100.0, rows) as value)
   saveTable("{YOUR_DIR}" + "/testOnDiskDB", tdata, `{tbName})
   """.format(YOUR_DIR=data_dir, tbName=tableName)
   >>> s.run(create_onDisk_table)
   >>> df = orca.read_table(data_dir + "/testOnDiskDB", tableName)
   >>> df
     type      value
   0    e  42.537911
   1    b  44.813589
   2    d  28.939636
   3    a  73.719393
   4    b  66.576416
   5    c  36.265364
   6    a  43.936593
   7    e  56.951759
   8    e   4.290316
   9    d  29.229366
   ```

   上述脚本中，我们使用的`defalut_session`实际上就是通过`orca.connect`函数创建的会话，在Python端，我们可以通过这个会话与DolphinDB服务端进行交互。关于更多功能，请参见[DolphinDB Python API](https://github.com/dolphindb/python3_api_experimental)。

- 分布式表

   分布式表是DolphinDB推荐在生产环境下使用的数据存储方式，它支持快照级别的事务隔离，保证数据一致性。分布式表支持多副本机制，既提供了数据容错能力，又能作为数据访问的负载均衡。在Orca中，可以通过`read_csv`函数指定分布式表导入数据并加载分布式表信息，或者通过`read_table`函数加载分布式表信息。
   
   - `read_csv`函数

      Orca在调用`read_csv`函数时指定db_handle, table_name以及partition_columns参数，可以直接将数据直接导入DolphinDB的DFS表，关于`read_csv`函数将数据导入分区表的详细介绍请参考[Orca的分区表](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/api_differences.md#81-orca%E7%9A%84%E5%88%86%E5%8C%BA%E8%A1%A8)。
     
      **示例**     

      请注意只有启用enableDFS=1的集群环境或者DolphinDB单例模式才能使用分布式表。 
     
      以[第1节](#1-将数据导出到磁盘)例子中的csv文件为例，我们在DolphinDB服务端创建一个DFS数据库，将demo.csv导入数据库：
            
      ```dolphindb
      dbPath="dfs://demoDB"
      login('admin', '123456')
      if(existsDatabase(dbPath))
            dropDatabase(dbPath)
      db=database(dbPath, VALUE, `a`b`c`d`e)
      ```
      
     > 请注意：以上脚本需要在DolphinDB服务端执行，在Python客户端中则可以通过DolphinDB Python API执行脚本。
     
     在Python客户端中调用Orca的`read_csv`函数，指定数据库db_handle为DFS数据库"dfs://demoDB"，指定表名table_name为"tb1"和进行分区的列partition_columns为"type"，将数据导入到DolphinDB分区表，这时，`read_csv`函数返回的是一个表示DolphinDB分区表的对象，客户端并不能直接访问表内的数据。在后续的计算中，Orca才会从服务端下载计算所需数据。
     
     ```Python
     >>> df = orca.read_csv(path=YOUR_DIR + "/demo.csv", dtype={"type": "SYMBOL", "value": np.float64}, db_handle="dfs://demoDB", table_name="tb1", partition_columns="type")
     >>> df
     # output
     <'dolphindb.orca.core.frame.DataFrame' object representing a column in a DolphinDB segmented table>
     ```

     若需要查看df内的数据，可以调用`to_pandas`函数查看，由于分区表的数据分布在各个分区上，调用`to_pandas`函数会将所有数据下载到客户端，且按照分区的顺序输出数据。

     ```Python
     >>> df.to_pandas()
     # output
        type      value
      0    a  72.641846
      1    c  93.697510
      2    c  64.533273
      3    c  46.758312
      4    c  89.597063
      5    d   0.262836
      6    d  25.223624
      7    e  11.699053
      8    e  30.315109
      9    e  60.980473
     ```
    
     将上述过程整合成的Python中可执行的脚本如下：
     
     ```Python
     >>> YOUR_DIR = "/dolphindb/database" # e.g. data_dir
     >>> s = orca.default_session()
     >>> dfsDatabase = "dfs://demoDB"
     >>> create_database = """
     dbPath='{dbPath}'
     login('admin', '123456')
     if(existsDatabase(dbPath))
         dropDatabase(dbPath)
     db=database(dbPath, VALUE, `a`b`c`d`e)
     """.format(dbPath=dfsDatabase)
     >>> s.run(create_database)
     >>> df=orca.read_csv(path=YOUR_DIR +"/demo.csv", dtype={"type": "SYMBOL", "value": np.float64},
                          db_handle=dfsDatabase, table_name="tb1", partition_columns="type")
     ```

     > 请注意：在通过`read_csv`函数指定数据库导入数据之前，需要确保在DolphinDB服务器上已经创建了对应的数据库。`read_csv`函数根据指定的数据库，表名和分区字段导入数据到DolphinDB数据库中，若表存在则追加数据，若表不存在则创建表并且导入数据。
   
   - `read_table`函数加载分区表信息

     若Orca调用`read_table`函数加载的是磁盘分区表或者dfs分区表，则数据不会在加载的时候被下载，以上述例子中创建的dfs分区表为例：

     ```Python
     >>> df = orca.read_table("dfs://demoDB", "tb1")
     >>> df
     # output
     <'orca.core.frame.DataFrame' object representing a column in a DolphinDB segmented table>
     ```

     对df进行计算，则下载数据进行计算：

     ```Python
     >>> df.groupby("type").mean()
     # output
               value
     type           
      a     72.641846
      c     73.646539
      d     12.743230
      e     34.331545
     ```

下面介绍向Orca的数据表写数据的过程。

#### 2.1 保存数据到Orca内存表

pandas提供的[append](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html#pandas.DataFrame.append)函数用于将一个DataFrame追加到另一个Dataframe，并返回一个新的DataFrame，不会对原有的DataFrame进行修改。在Orca中，`append`函数还支持inplace参数，当它为True时，会将追加的数据保存到Dataframe中，对原有的DataFrame进行了修改，这个过程就是将数据追加到Orca的内存表中。

```python
>>> df1 = orca.DataFrame({"date":orca.date_range("20190101", periods=10),
                          "price":np.random.sample(10)*100})
>>> df1
# output
        date      price
0 2019-01-01  17.884136
1 2019-01-02  57.840625
2 2019-01-03  29.781247
3 2019-01-04  89.968203
4 2019-01-05  19.355847
5 2019-01-06  74.684634
6 2019-01-07  91.678632
7 2019-01-08  93.927549
8 2019-01-09  47.041906
9 2019-01-10  96.810450

>>> df2 = orca.DataFrame({"date":orca.date_range("20190111", periods=3),
                          "price":np.random.sample(3)*100})
>>> df2
# output 
        date      price
0 2019-01-11  26.959939
1 2019-01-12  75.922693
2 2019-01-13  93.012894

>>> df1.append(df2, inplace=True)
>>> df1
# output
        date      price
0 2019-01-01  17.884136
1 2019-01-02  57.840625
2 2019-01-03  29.781247
3 2019-01-04  89.968203
4 2019-01-05  19.355847
5 2019-01-06  74.684634
6 2019-01-07  91.678632
7 2019-01-08  93.927549
8 2019-01-09  47.041906
9 2019-01-10  96.810450
0 2019-01-11  26.959939
1 2019-01-12  75.922693
2 2019-01-13  93.012894
```

> 请注意：当设置inplace参数为True时，index_ignore参数的值不允许设置，只能为False。

#### 2.2 保存数据到Orca磁盘表

Orca提供两种方式修改磁盘表的数据：

- `save_table`函数

- `append`函数

#### 2.2.1 保存数据到Orca本地磁盘表

Orca提供`save_table`函数，用于保存数据到磁盘表和分布式表，该函数参数如下：

|参数|用法|
|:--|:--|
|db_path|数据库路径|
|table_name|表名|
|df|需要保存的表|
|ignore_index|是否忽略index追加数据|

首先通过`read_table`函数导入上文中创建的磁盘表。

```Python
>>> df = orca.read_table(YOUR_DIR + "/testOnDiskDB", "tb1")
>>> df
# output 
  type      value
0    e  42.537911
1    b  44.813589
2    d  28.939636
3    a  73.719393
4    b  66.576416
5    c  36.265364
6    a  43.936593
7    e  56.951759
8    e   4.290316
9    d  29.229366
```

生成要追加的数据，追加数据到df，并通过`save_table`保存数据。

```Python
>>> df2 = orca.DataFrame({"type": np.random.choice(list("abcde"),3), 
                          "value": np.random.sample(3)*100})
>>> df.append(df2, inplace=True)
>>> df
# output
  type      value
0    e  42.537911
1    b  44.813589
2    d  28.939636
3    a  73.719393
4    b  66.576416
5    c  36.265364
6    a  43.936593
7    e  56.951759
8    e   4.290316
9    d  29.229366
0    d  20.702066
1    c  21.241707
2    a  97.333201
>>> orca.save_table(YOUR_DIR + "/testOnDiskDB", "tb1", df)
```

需要注意的是，对于磁盘表，若该指定的表名不存在于数据库中，`save_table`会创建对应的表；若数据库中已有同名的表，`save_table`会覆盖该表。

#### 2.2.2 保存数据到Orca磁盘分区表

磁盘分区表与分布式表的差异就在于分布式表的数据库路径以"dfs://"开头，而磁盘分区表的数据库路径是本地的一个绝对路径。

- 通过`save_table`函数将数据保存到磁盘分区表
  
   直接调用`save_table`函数，可以将一个内存表以分区的形式保存到磁盘上，与磁盘非分区表类似，若表已存在，会覆盖该表。

   ```Python
   >>> df2 = orca.DataFrame({"type": np.random.choice(list("abcde"),3), 
                             "value": np.random.sample(3)*100})
   >>> orca.save_table(YOUR_DIR + "/testOnDisPartitionedkDB", "tb1", df2)
   >>> df = orca.read_table(YOUR_DIR + "/testOnDisPartitionedkDB", "tb1")
   >>> df
   # output
     type      value
   0    d  86.549417
   1    e  61.852710
   2    d  28.747059
   ```

- 通过`append`函数追加数据到磁盘分区表
  
   对于磁盘分区表，调用`append`函数可以向磁盘分区表追加数据。

   首先，在DolphinDB中创建磁盘分区表：
  
   ```
   dbPath=YOUR_DIR + "/testOnDisPartitionedkDB"
   login('admin', '123456')
   if(existsDatabase(dbPath))
      dropDatabase(dbPath)
   db=database(dbPath, VALUE, `a`b`c`d`e)
   ```

   在Python客户端中导入[第1节](#1-将数据导出到磁盘)例子中的csv文件

   ```Python
   >>> df = orca.read_csv(path=YOUR_DIR + "/demo.csv", dtype={"type": "SYMBOL", "value": np.float64}, db_handle=YOUR_DIR + "/testOnDisPartitionedkDB", table_name="tb1", partition_columns="type")
   >>> df.to_pandas()
   # output
   type      value
   0    a  72.641846
   1    c  93.697510
   2    c  64.533273
   3    c  46.758312
   4    c  89.597063
   5    d   0.262836
   6    d  25.223624
   7    e  11.699053
   8    e  30.315109
   9    e  60.980473
   ```

   调用append函数向df表追加数据，重新加载该磁盘分区表，发现数据已经追加：

   ```Python
   >>> df2 = orca.DataFrame({"type": np.random.choice(list("abcde"),3), 
                             "value": np.random.sample(3)*100})
   >>> df.append(df2,inplace=True)
   >>> df = orca.read_table(YOUR_DIR + "/testOnDisPartitionedkDB", "tb1")
   >>> df.to_pandas()
   # output
      type      value
   0     a  72.641846
   1     c  93.697510
   2     c  64.533273
   3     c  46.758312
   4     c  89.597063
   5     c  29.233253
   6     c  38.753028
   7     d   0.262836
   8     d  25.223624
   9     d  55.085909
   10    e  11.699053
   11    e  30.315109
   12    e  60.980473
   ```

   将上述过程整合成Python端的可执行脚本如下：

   ```Python
   >>> YOUR_DIR = "/dolphindb/database" # e.g. data_dir
   >>> s = orca.default_session()
   >>> create_database = """
   dbPath='{dbPath}'
   login('admin', '123456')
   if(existsDatabase(dbPath))
      dropDatabase(dbPath)
   db=database(dbPath, VALUE, `a`b`c`d`e)
   """.format(dbPath=YOUR_DIR + "/testOnDisPartitionedkDB")
   >>> s.run(create_database)
   >>> df = orca.read_csv(path=YOUR_DIR + "/demo.csv", dtype={"type": "SYMBOL", "value": np.float64}, db_handle=YOUR_DIR + "/testOnDisPartitionedkDB", table_name="tb1", partition_columns="type")
   >>> df2 = orca.DataFrame({"type": np.random.choice(list("abcde"),3), 
                             "value": np.random.sample(3)*100})
   >>> df.append(df2,inplace=True)
   >>> df = orca.read_table(YOUR_DIR + "/testOnDisPartitionedkDB", "tb1")
   ```

#### 2.3 保存数据到Orca分布式表

- 通过`append`函数追加数据到分布式表
  
   对于分布式表，可以直接通过`append`函数追加数据。

   首先，在DolphinDB中创建分布式表：
  
   ```
   dbPath="dfs://demoDB"
   login('admin', '123456')
   if(existsDatabase(dbPath))
      dropDatabase(dbPath)
   db=database(dbPath, VALUE, `a`b`c`d`e)
   ```

   在Python客户端中导入[第1节](#1-将数据导出到磁盘)例子中的csv文件：

   ```Python
   >>> df = orca.read_csv(path=YOUR_DIR + "/demo.csv", dtype={"type": "SYMBOL", "value": np.float64}, db_handle="dfs://demoDB", table_name="tb1", partition_columns="type")
   >>> df.to_pandas()
   # output
   type      value
   0    a  72.641846
   1    c  93.697510
   2    c  64.533273
   3    c  46.758312
   4    c  89.597063
   5    d   0.262836
   6    d  25.223624
   7    e  11.699053
   8    e  30.315109
   9    e  60.980473
   ```

   调用`append`函数向df表追加数据，重新加载该分布式表，发现数据已经追加：

   ```Python
   >>> df2 = orca.DataFrame({"type": np.random.choice(list("abcde"),3), 
                             "value": np.random.sample(3)*100})
   >>> df.append(df2,inplace=True)
   >>> df = orca.read_table("dfs://demoDB", "tb1")
   >>> df.to_pandas()
   # output
      type      value
   0     a  72.641846
   1     a  55.429765
   2     a  51.230669
   3     c  93.697510
   4     c  64.533273
   5     c  46.758312
   6     c  89.597063
   7     c  71.821263
   8     d   0.262836
   9     d  25.223624
   10    e  11.699053
   11    e  30.315109
   12    e  60.980473
   ```

   将上述过程整合成Python端的可执行脚本如下：

   ```Python
   >>> YOUR_DIR = "/dolphindb/database" # e.g. data_dir
   >>> s = orca.default_session()
   >>> create_database = """
   dbPath='{dbPath}'
   login('admin', '123456')
   if(existsDatabase(dbPath))
      dropDatabase(dbPath)
   db=database(dbPath, VALUE, `a`b`c`d`e)
   """.format(dbPath="dfs://demoDB")
   >>> s.run(create_database)
   >>> df = orca.read_csv(path=YOUR_DIR + "/demo.csv", dtype={"type": "SYMBOL", "value": np.float64}, db_handle="dfs://demoDB", table_name="tb1", partition_columns="type")
   >>> df2 = orca.DataFrame({"type": np.random.choice(list("abcde"),3), 
                             "value": np.random.sample(3)*100})
   >>> df.append(df2,inplace=True)
   >>> df = orca.read_table("dfs://demoDB", "tb1")
   ```

- 通过`save_table`函数追加数据到分布式表
  
   与磁盘表不同的是，对分布式表调用`save_table`函数，可以直接追加数据，而不是覆盖数据。且与`append`函数相比，`save_table`函数无需先在客户端通过`read_table`获得将要追加的表信息，就直接在DolphinDB服务端上追加数据的操作。

   下面的例子中，通过`save_table`函数直接将内存表的数据追加到指定表：

   ```Python
   >>> df2 = orca.DataFrame({"type": np.random.choice(list("abcde"),3), 
                             "value": np.random.sample(3)*100})
   >>> orca.save_table("dfs://demoDB", "tb1", df2)
   >>> df = orca.read_table("dfs://demoDB", "tb1")
   >>> df.to_pandas()
   # output
      type      value
   0     a  72.641846
   1     a  55.429765
   2     a  51.230669
   3     b  40.724064
   4     c  93.697510
   5     c  64.533273
   6     c  46.758312
   7     c  89.597063
   8     c  71.821263
   9     c  93.533380
   10    d   0.262836
   11    d  25.223624
   12    d  47.238962
   13    e  11.699053
   14    e  30.315109
   15    e  60.980473
   ```

### 3 小结
1. Orca的`to_csv`函数在engine='dolphindb'的默认状态下只支持sep和append两个参数。
2. 对于普通磁盘表以外的表，inplce参数置为True时，`append`方法将追加数据。
3. `save_table`函数，对于本地磁盘表会覆盖原表；对于dfs表，数据会被追加到表中