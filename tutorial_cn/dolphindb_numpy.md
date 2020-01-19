# DolphinDB NumPy使用教程

NumPy是一个用于科学计算的基础库，常和pandas配合使用，实现复杂计算。Orca的底层实现基于DolphinDB，如果用NumPy函数直接处理Orca对象，会直接将Orca对象下载到本地计算，造成不必要的[性能损失](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/user_guide.md#避免用numpy函数处理orca对象)，甚至可能导致异常。为此，Orca提供了一个附属项目，DolphinDB NumPy。它包装了NumPy的接口，针对Orca对象有优化，又不影响其他情况的使用。

 - [1 安装](#1-安装)    
 - [2 快速入门](#2-快速入门)
 - [3 DolphinDB NumPy的功能限制](#3-dolphindb-numpy的功能限制和注意事项)

## 1 安装

DolphinDB NumPy项目已经集成到[DolphinDB Python API](https://github.com/dolphindb/Tutorials_CN/blob/master/python_api.md)中。通过pip工具安装DolphinDB Python API，就可以使用DolphinDB NumPy。

```
pip install dolphindb
```

如果你已经有现成的NumPy程序，可以将NumPy的import替换为：

```python
# import numpy as np
import dolphindb.dolphindb_numpy as np
```

如果程序用到了orca对象，请保证已经连接到DolphinDB。

## 2 快速入门

通过传入一列值创建一个DolphinDB NumPy ndarray对象。

```python
>>> a = dolphindb_numpy.array([1, 2])
>>> a
array([1, 2])
```

如果尝试获得a的类型，会发现它就是一个NumPy的ndarray，DolphinDB NumPy只是一个包装：

```python
>>> type(a)
<class 'numpy.ndarray'>
```

DolphinDB NumPy的使用与NumPy无异：

```python
>>> dolphindb_numpy.exp(range(5))
array([ 1.        ,  2.71828183,  7.3890561 , 20.08553692, 54.59815003])
>>> dolphindb_numpy.random.randint(0, 10, 3)
array([4, 7, 8])
```

DolphinDB NumPy的ndarray对象可以与Orca对象直接运算。返回结果是Orca中间表达式。

```python
>>> df = orca.DataFrame({"a": [1,2]})
>>> a + df
<orca.core.operator.ArithExpression object at 0x7ffa4a1d99d0>
```

## 3 DolphinDB NumPy的功能限制和注意事项

DolphinDB NumPy目前还在开发阶段，DolphinDB NumPy的接口函数，若参数中包括Orca对象，仅支持四则运算、逻辑运算、DolphinDB支持的数学函数和统计函数。

用DolphinDB NumPy函数操作Orca对象时，会采用Orca所使用的[惰性求值](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/user_guide.md#orca并非总是立刻求值)策略。因此，常见的四则运算、逻辑运算等，通常会返回一个中间表达式：

```python
>>> a = DolphinDB NumPy.float32(3.5)
>>> df = orca.Series([1,2])
>>> b = a + df
>>> b
<orca.core.operator.ArithExpression object at 0x7ffa4a1d99d0>

>>> b.compute()
0    4.5
1    5.5
dtype: float32
```
