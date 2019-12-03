本文介绍如何使用`append`方法，`to_csv`方法与`save_table`函数来向DolphinDB添加数据。

DolphinDB中的表分为内存表，普通磁盘表，磁盘分区表和DFS表。

Orca中的DataFrame，若是通过orca.DataFrame创建，或是通过read_table读取磁盘表的，则存在于内存中。若是从DFS表中读取，则它是一个DolphinDB的分布式表。

 - [1 `append`方法与inplace参数](#1-append方法与inplace参数)    
 - [2 `save_table`函数](#2-save_table函数)
 - [3 `to_csv`函数](#3-to_csv函数)
  
`append`与`save_table`的具体操作会根据表的类型而不同。

### `append`方法与inplace参数

Orca为DataFrame提供了一个向DataFrame添加数据的方法，`append`：

```python
>>> odf = orca.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
>>> odf2 = orca.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
>>> odf
Out: 
   A  B
0  1  2
1  3  4
>>> odf2
Out: 
   A  B
0  5  6
1  7  8
>>> odf.append(odf2)
Out: 
   A  B
0  1  2
1  3  4
0  5  6
1  7  8
```

对于磁盘分区表与DFS表，将inplace参数置为True，会自动写入数据库中的原表。

例如，创建本教程的示例数据库dfs://whereDB 。创建数据库的DolphinDB脚本如下所示：

```
dbPath='dfs://whereDB'                   
if(existsDatabase(dbPath))                // 若dbPath上的数据库已存在，先drop这个数据库
    dropDatabase(dbPath)
db=database(dbPath, VALUE, 1..10)         // 创建一个值分区的数据库
tdata=table(1:0,`id`date`name, [INT,DATE,SYMBOL]) 
db.createPartitionedTable(tdata, `tb, `id) 
```

执行以下Python代码：

```python
>>> n = 10
>>> df = orca.DataFrame({         
...     "id": np.arange(1, n + 1, 1, dtype='int32'),
...     'date': pd.date_range('2019.08.01', periods=10, freq='D'),
...     'name': ['a', 'b', 'c', 'd', 'e', 'QWW', 'FEA', 'FFW', 'DER', 'POD']})

>>> s = orca.default_session()  # orca.default_session()是Orca提供的默认DolphinDB连接
>>> s.run("tableInsert{loadTable('dfs://whereDB',`tb)}", df) # 运行dolphindb脚本

>>> odf = orca.read_table("dfs://whereDB", 'tb')     # 使用read_table加载表

>>> len(odf)
Out:
10

>>> odf.append(odf, inplace=True)        # 使用append方法追加数据
>>> x = orca.read_table('dfs://whereDB','tb')  # 使用read_table读取表名为tb的表
>>> len(x)

Out:
20

```

而对于非分区普通磁盘表，我们需要调用`save_table`函数来保存，具体见下文。


### `save_table`函数

`orca.save_table`函数可以将DataFrame以dolphindb表的形式保存到磁盘或dfs表中。

对于磁盘表或磁盘分区表，若该表名不存在于数据库中，`save_table`会创建对应的表；若数据库中已有同名的表，`save_table`会覆盖该表：

```python
>>> odf = orca.DataFrame({
...     'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
...     'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
... }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

>>> orca.default_session().run(f"db = database('home/yourdir/imd');")
>>> orca.save_table("home/yourdir/" + "imd", "imdb", odf)
>>> x = orca.read_table("home/yourdir/" + "imd", "imdb")

>>> x

Out:
   a  b
0  1  4
1  2  5
2  3  6
3  4  3
4  5  2
5  6  1
6  7  0
7  8  0
8  9  0
```

可以发现，原来DataFrame的Index信息丢失，这一点需要注意。


而对于dfs表，数据库中需有对应的表才可以写入，且save_table不会覆盖该表，而是追加数据到表中

以下为创建分区数据库和分区表的DolphinDB脚本
```
dbPath='dfs://whereDB'                   
if(existsDatabase(dbPath))                // 若dbPath上的数据库已存在，先drop这个数据库
    dropDatabase(dbPath)
db=database(dbPath, VALUE, 1..10)         // 创建一个值分区的数据库
tdata=table(1:0,`id`date`name, [INT,DATE,SYMBOL]) 
db.createPartitionedTable(tdata, `tb, `id) 

```

以下为调用Orca的python脚本
```python
>>> n = 10
>>> df = orca.DataFrame({         
...     "id": np.arange(1, n + 1, 1, dtype='int32'),
...     'date': pd.date_range('2019.08.01', periods=10, freq='D'),
...     'name': ['a', 'b', 'c', 'd', 'e', 'QWW', 'FEA', 'FFW', 'DER', 'POD']})

>>> s = orca.default_session()
>>> s.run("tableInsert{loadTable('dfs://whereDB',`tb)}", df)

>>> odf = orca.read_table("dfs://whereDB", 'tb')

>>> len(odf)
Out:
10

>>> orca.save_table("dfs://whereDB", 'tb', odf)        # 使用save_table追加数据到表中
>>> x = orca.read_table("dfs://whereDB", 'tb')
>>> len(x)

Out:
20
```

### `to_csv`方法

对DataFrame调用`to_csv`方法，可以直接把DataFrame保存到csv文件中。

```python
>>> df = orca.DataFrame({'name': ['Raphael', 'Donatello'], 'mask': ['red', 'purple'], 'weapon': ['sai', 'bo staff']})
>>> df.to_csv(path_or_buf=f"home/yourdir/tocsv.csv")
>>> x = orca.read_csv(path = f"home/yourdir/tocsv.csv")

>>> x

Out:
        name    mask    weapon
0    Raphael     red       sai
1  Donatello  purple  bo staff
```
engine参数：
Orca的to_csv函数的engine参数的取值可以是{'c', 'python', 'dolphindb'}，且该参数默认取值为'dolphindb'。
当取值为'dolphindb'时，to_csv只支持sep和append两个参数。
当取值为'python'或'c'时，to_csv的参数列表与pandas一致。


### 小结
1. 对于非分区的普通磁盘表以外的表，inplce参数置为Ture时，append方法将追加数据。
2. save_table函数，对于本地磁盘表会覆盖原表，DataFrame的Index会丢失；对于dfs表，数据库中需有原表，且数据会被追加到表中
3. to_csv函数在engine='dolphindb'的默认状态下只支持sep和append两个参数。