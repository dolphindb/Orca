# DolphinDB NumPy使用教程

orca的底层实现基于DolphinDB。若用NumPy函数直接处理orca对象，会直接将orca对象下载到本地计算，造成不必要的[性能损失](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/user_guide.md#避免用numpy函数处理orca对象)，甚至可能导致异常。为此，orca提供了一个附属项目DolphinDB NumPy。它包装了NumPy的接口，针对orca对象进行优化。

 - [1 安装](#1-安装)    
 - [2 快速入门](#2-快速入门)
 - [3 功能限制与注意事项](#3-dolphindb-numpy的功能限制和注意事项)

## 1 安装

DolphinDB NumPy项目集成于[DolphinDB Python API](https://github.com/dolphindb/Tutorials_CN/blob/master/python_api.md)中。通过pip工具安装DolphinDB Python API，就可使用DolphinDB NumPy。

```python
pip install dolphindb
```

若要继续使用现有的使用NumPy的Python程序，可将 "import numpy as np"语句替换为：

```python
import dolphindb.numpy as np
```

如果程序用到了orca对象，请保证已经连接到DolphinDB。

## 2 快速入门

通过传入一列值创建一个DolphinDB NumPy ndarray对象：
```python
>>> import dolphindb.numpy as np
>>> a = np.array([1, 2])
>>> a
array([1, 2])
```

a是一个NumPy的ndarray。

```python
>>> type(a)
<class 'numpy.ndarray'>
```

DolphinDB NumPy的使用与NumPy无异：
```python
>>> import dolphindb.numpy as np
>>> np.exp(range(5))
array([ 1.,  2.71828183,  7.3890561, 20.08553692, 54.59815003])
>>> np.random.randint(0, 10, 3)
array([4, 7, 8])
```

DolphinDB NumPy的ndarray对象可以与orca对象进行运算，返回结果是orca中间表达式。
```python
>>> df = orca.DataFrame({"a": [1,2]})
>>> a + df
<orca.core.operator.ArithExpression object at 0x7ffa4a1d99d0>
```

## 3 功能限制与注意事项

目前DolphinDB NumPy的接口函数若应用于orca对象，仅支持四则运算、逻辑运算，以及DolphinDB支持的数学函数和统计函数。

用DolphinDB NumPy函数操作orca对象时，会采用orca所使用的[惰性求值](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/user_guide.md#orca并非总是立刻求值)策略。因此，常见的四则运算、逻辑运算等，通常会返回一个中间表达式：

```python
>>> import dolphindb.numpy as np
>>> a = np.float32(3.5)
>>> df = orca.Series([1,2])
>>> b = a + df
>>> b
<orca.core.operator.ArithExpression object at 0x7ffa4a1d99d0>

>>> b.compute()
0    4.5
1    5.5
dtype: float32
```
