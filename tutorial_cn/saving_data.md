 # orca教程：数据写入

本文介绍如何在orca中保存数据。

- [1 将数据导出到磁盘](#1-将数据导出到磁盘)
- [2 将数据保存到orca数据表](#2-将数据保存到orca数据表)
  - [2.1 保存数据到orca内存表](#21-保存数据到orca内存表)
  - [2.2 保存数据到orca分布式表](#22-保存数据到orca分布式表)
  - [2.3 保存数据到磁盘表](#23-保存数据到磁盘表)


### 1 将数据导出到磁盘

orca的Series和DataFrame均支持`to_csv`，`to_excel`等将数据导出为指定格式的文件的方法。下面对`to_csv`进行说明。

- `to_csv`函数

   与pandas的`to_csv`函数相比，orca的`to_csv`函数添加了两个参数：engine与append。engine参数的取值可以是'python'或'dolphindb'，表示使用哪一种引擎导出数据，默认取值为'dolphindb'。
   
    - engine='dolphindb'时，`to_csv`函数会将数据导出到DolphinDB服务器，且只支持sep和append两个参数。append参数仅适用于当engine='dolphindb'时，设为True时，导出数据会在文件末尾追加，不会覆盖文件。
   
    - engine='python'时，`to_csv`函数会把数据转换成pandas dataframe然后用pandas的`to_csv`函数导出到Python客户端。此时，支持pandas的`to_csv`函数支持的所有参数。
   
   **示例**

   调用`to_csv`函数导出数据，并通过`read_csv`函数再将数据导入。以下的脚本中'YOUR_DIR'表示用户保存csv文件的路径。

   ```python
   >>> YOUR_DIR = "/dolphindb/database"
   >>> odf = orca.DataFrame({"type": np.random.choice(list("abcde"),100), "value": np.random.sample(100)})
   >>> odf.to_csv(path_or_buf=YOUR_DIR + "/sample.csv")
   >>> df1 = orca.read_csv(path=YOUR_DIR + "/sample.csv")
   ```
   此处产生的csv文件 sample.csv 会在以下章节中反复使用。

### 2 将数据保存到orca数据表

基于DolphinDB的数据表设计，orca数据表按存储方式分为三种:

- 内存表：数据仅保存在DolphinDB服务端内存中，存取速度最快。可以使用`read_csv`函数导入或者通过`DataFrame`函数创建。
- 分布式表：数据存储在DolphinDB服务端。可以通过`read_csv`函数将文本文件导入分布式表并加载分布式表信息，或者通过`read_table`函数加载分布式表信息。加载后，客户端并没有将数据加载到内存，而只是获得了数据库和表的元数据。分布式表是DolphinDB推荐在生产环境下使用的数据存储方式。它支持快照级别的事务隔离，保证数据一致性。分布式表支持多副本机制，支持数据容错与负载均衡。
- 磁盘表：数据存储在DolphinDB服务端，可以使用`read_table`函数加载到内存。不建议在生产环境下使用。

#### 2.1 保存数据到orca内存表

pandas提供的[`append`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html#pandas.DataFrame.append)函数用于将一个DataFrame追加到另一个Dataframe，返回一个新的DataFrame，不会对原有的DataFrame进行修改。orca中的`append`函数还支持inplace参数，当它为True时，会将数据直接追加到原有orca的内存表中。

```python
>>> df1 = orca.DataFrame({"date":orca.date_range("20190101", periods=10), "price":np.random.sample(10)})
>>> len(df1)
10

>>> df2 = orca.DataFrame({"date":orca.date_range("20190111", periods=3), "price":np.random.sample(3)})
>>> len(df2)
3

>>> df1.append(df2, inplace=True)
>>> len(df1)
13
```
> 请注意：当inplace参数为True时，ignore_index参数的值不允许设置为True，只能为默认值False。

#### 2.2 保存数据到orca分布式表

orca提供两种方式修改保存至磁盘上的数据：

- `append`函数

- `save_table`函数，对DFS表会追加数据。

该函数参数如下：
|参数|用法|
|:--|:--|
|df|需要保存的数据表|
|db_path|数据库路径|
|table_name|表名|
|ignore_index|是否忽略index追加数据|


首先，创建分布式数据库demoDB：
```python
>>> s = orca.default_session()
>>> create_database = """
dbPath='{dbPath}'
login('admin', '123456')
if(existsDatabase(dbPath))
  dropDatabase(dbPath)
db=database(dbPath, VALUE, `a`b`c`d`e)
""".format(dbPath="dfs://demoDB")
>>> s.run(create_database)
```
将sample.csv文件载入数据库demoDB，存为数据表tb1：
```python
>>> YOUR_DIR = "/dolphindb/database" 
>>> df = orca.read_csv(path=YOUR_DIR + "/sample.csv", dtype={"type": "SYMBOL", "value": np.float64}, db_handle="dfs://demoDB", table_name="tb1", partition_columns="type")
```

- 使用`append`函数

   调用`append`函数向df表追加数据：
   ```python
   >>> df1 = orca.DataFrame({"type": np.random.choice(list("abcde"),3), "value": np.random.sample(3)*100})
   >>> df.append(df1,inplace=True)
   ```   
   重新加载该分布式表，可见数据已经追加：
   ```python
   >>> df = orca.read_table("dfs://demoDB", "tb1")
   >>> len(df)
   103
   ```
   继续添加数据：
   ```python
   >>> df1 = orca.DataFrame({"type": np.random.choice(list("abcde"),3), "value": np.random.sample(3)*100})
   >>> df.append(df1,inplace=True)
   >>> df = orca.read_table("dfs://demoDB", "tb1")
   >>> len(df)
   106
   ```      
- 使用`save_table`函数
  
   对分布式表调用`save_table`函数，会追加数据。与`append`函数相比，`save_table`函数无需先在客户端通过`read_table`函数获得将要追加的表信息，就可直接在DolphinDB服务端上追加数据。

   下面的例子中，通过`save_table`函数直接将内存表的数据追加到指定表：

   ```python
   >>> df1 = orca.DataFrame({"type": np.random.choice(list("abcde"),3), "value": np.random.sample(3)*100})
   >>> orca.save_table(df1, "dfs://demoDB", "tb1")
   >>> df = orca.read_table("dfs://demoDB", "tb1")
   >>> len(df)
   109
   ```

#### 2.3 保存数据到磁盘表

磁盘分区表与分布式表的差异在于分布式表的数据库路径以"dfs://"开头，而磁盘分区表的数据库路径是本地的一个绝对路径。保存数据到orca磁盘表，2.2小节中的脚本只需将数据库路径以"dfs://"开头改为本地绝对路径即可。唯一要注意的是，使用`save_table`函数将数据保存到磁盘表，若该表已存在，会覆盖该表。
