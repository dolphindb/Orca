# Orca: pandas API on DolphinDB

温馨提示：目前，DolphinDB 已不再维护该包。

Orca项目在DolphinDB之上实现了pandas API，使用户能更高效地分析处理海量数据。

如果你已经熟悉pandas，你就能通过Orca包，充分利用DolphinDB的高性能和并发，处理海量数据，而不需要额外的学习曲线。如果你已经有现成的pandas代码，你不需要对已有的pandas代码进行大量修改，就能迁移到Orca。

## Orca的设计理念

Python的第三方库pandas是一个强大的分析结构化数据的工具，具有高性能、接口易用、易学习的特点，在数据科学和量化金融领域广受欢迎。然而，当我们开始处理TB级别的海量数据时，单核运行的pandas就显得力不从心；pandas的高内存占用也是影响其发挥的限制之一。此外，如今每台服务器拥有许多CPU核，每个公司部署的服务器数量也较多，若可以充分利用并发的优势，可以提高数据处理的效率。

DolphinDB是一个分布式数据分析引擎，它可以将TB级的海量数据存储在多台物理机器上，并能充分利用CPU，对海量数据进行高性能分析计算。在进行同样功能的计算时，DolphinDB在性能上比pandas快1~2个数量级，并且[内存占用通常小于pandas的1/2](https://zhuanlan.zhihu.com/p/41979956)。

DolphinDB开发的Orca项目是基于DolphinDB引擎的pandas DataFrame API的实现。它让用户能够以pandas的编程风格，同时利用DolphinDB的性能优势，对海量数据进行高效分析。相比pandas的全内存计算，Orca支持分布式存储和计算。对于同样的数据量，内存占用一般小于pandas的1/2。

### Orca的架构

Orca的顶层是pandas API，底层是DolphinDB数据库，通过DolphinDB Python API实现Orca客户端与DolphinDB服务端的通信。Orca的基本工作原理是，在客户端通过Python生成DolphinDB脚本，将脚本通过DolphinDB Python API发送到DolphinDB服务端解析执行。Orca的DataFrame中只存储对应的DolphinDB的表的元数据，真正的存储和计算都是在服务端。

因此，Orca的接口有部分限制：

- Orca的DataFrame中的每个列不能是混合类型，列名也必须是合法DolphinDB变量名。
- 如果DataFrame对应的DolphinDB表是一个分区表，数据存储并非连续，因此就没有RangeIndex的概念，且无法将一整个Series赋值给一个DataFrame的列。
- 对于DolphinDB分区表，一部分没有分布式版本实现的函数，例如median，暂不支持。
- DolphinDB的空值机制和pandas不同，pandas用float类型的nan作为空值，而DolphinDB的空值是每个类型的最小值。
- DolphinDB是列式存储的数据库。对于pandas接口中，一些axis=columns参数还没有支持。
- 目前无法解析Python函数，因此，例如`DataFrame.apply`, `DataFrame.agg`等函数无法接受一个Python函数作为参数。

关于Orca和pandas的详细差异，以及由此带来的Orca编程注意事项，请参考[Orca使用教程](../../../Orca/blob/master/tutorial_cn/user_guide.md)。

## 安装

Orca支持Linux和Windows系统，要求Python版本为3.6及以上，pandas版本为0.25.1及以上。

orca项目已经集成到[DolphinDB Python API](../../../Tutorials_CN/blob/master/python_api.md)中。通过pip工具安装DolphinDB Python API，就可以使用orca。

```
pip install dolphindb
```

Orca是基于DolphinDB Python API开发的，因此，你需要有一个DolphinDB服务器，并通过`connect`函数连接到这个服务器，然后运行Orca：

```python
>>> import dolphindb.orca as orca
>>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)
```

如果你已经有现成的pandas程序，可以将 `import pandas as pd` 替换为：

```python
import dolphindb.orca as pd
pd.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)
```

## 更多信息

- [使用教程和注意事项](./user_guide.md)
- [Orca与pandas API的详细差异](./api_differences.md)
- [Orca访问DolphinDB分布式数据库教程](./operations_on_distributed_database.md)
- [Orca保存数据教程](./saving_data.md)
- [用Orca开发量化策略](./momentum_strategy.md)
- [DolphinDB Python API](../../../../api_python3)
