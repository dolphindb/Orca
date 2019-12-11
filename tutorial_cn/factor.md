# 使用Orca计算策略因子

本教程讲述如何通过Orca实现一个因子的计算。教程用到的csv文件，由DolphinDB脚本生成。下面的脚本中，'DATA_DIR'为保存生成的csv文件的路径。

```
n=10000000
t=table(rand(100.0, n) as bv1, rand(100.0, n) as bv2, rand(100.0, n) as bv3, rand(100.0, n) as bv4, rand(100.0, n) as bv5, rand(100.0, n) as bv6, rand(100.0, n) as bv7, rand(100.0, n) as bv8, rand(100.0, n) as bv9, rand(100.0, n) as bv10, rand(100.0, n) as av1, rand(100.0, n) as av2, rand(100.0, n) as av3, rand(100.0, n) as av4, rand(100.0, n) as av5, rand(100.0, n) as av6, rand(100.0, n) as av7, rand(100.0, n) as av8, rand(100.0, n) as av9, rand(100.0, n) as av10, rand(100.0, n) as mp)
t.saveText(DATA_DIR + "/data/random.csv")
```

> 请注意，上述脚本需要在DolphinDB服务端执行，在Python客户端中请通过DolphinDB Python API执行脚本生成数据文件。

## 使用pandas计算因子

在pandas中，若需求DataFram中两个列的累积量比，可以通过如下步骤实现。

### １.导入数据

```Python
>>> import pandas as pd
>>> import numpy as np

>>> DATA_DIR = "/dolphindb/database"  # e.g. data directory
>>> pdf = pd.read_csv(DATA_DIR + "/data/random.csv")
>>> pdf
# output
                bv1        bv2        bv3  ...       av10         mp         wp
0        110.432940  36.866377  63.069911  ...  61.695058  12.116643  43.699660
1        133.311491  76.383935  51.522746  ...  38.012465  73.048654   6.267049
2         75.408190  14.717063  61.840130  ...  36.488270  33.466547  14.944621
3        121.934938  82.193587  34.547746  ...  45.023297  54.931852  25.739914
4        105.002563  30.773717  13.064657  ...  14.126881  92.764781   8.250228
             ...        ...        ...  ...        ...        ...        ...
9999995  138.958861  27.213705  31.930918  ...  57.000688  61.583911  74.309302
9999996  185.063615  66.317430  11.773088  ...  26.515529  65.975570  97.440483
9999997  186.630571  86.658859  19.990813  ...  51.496507  38.520961  91.533218
9999998  139.147857  75.301727  48.676274  ...  46.445336   8.588805  22.625894
9999999   82.616109  71.261632  15.186092  ...  92.291006  86.534062  25.115287
[10000000 rows x 22 columns]
```

### 2.计算因子

下面计算累积bid和ask量比，其中bid和ask均为表中的列。

```Python
>>> ask = pdf["av1"]
>>> bid = pdf["bv1"]
>>> p = pdf["mp"].iloc[0]
>>> for i in range(2, 11):
        ask += np.exp(-10*(i-1)/p) * pdf["av"+str(i)]
        bid += np.exp(-10*(i-1)/p) * pdf["bv"+str(i)]
>>> vol_diff = 0.5 * np.log(bid/ask)
>>> vol_diff
# output
0         -0.105137
1          0.371959
2         -0.256508
3          0.017896
4          0.093562
             ...   
9999995    0.100030
9999996    0.104554
9999997    0.153925
9999998   -0.149094
9999999   -0.073516
Length: 10000000, dtype: float64
```

## 使用Orca计算因子

在Orca中，对上述过程进行简单修改，即可执行：

- 以非分区方式导入数据

    Orca的分区表不支持通过`iloc`函数访问，因此在导入文件时，需指定partitioned参数为False，以非分区的方式导入数据。关于Orca分区表的具体限制请参见[Orca分区表的限制](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/api_differences.md#82-orca%E5%88%86%E5%8C%BA%E8%A1%A8%E7%9A%84%E7%89%B9%E6%AE%8A%E5%B7%AE%E5%BC%82)。

- 使用numpy库函数的限制

    - `np.exp(-10*(i-1)/p)`的计算结果与Orca的Series进行四则运算时，应保证Orca的Series在运算符左侧，numpy的运算结果在运算符右侧。详情参见[教程的说明](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/user_guide.md#操作数的顺序)。

    - 不应使用`np.log(bid/ask)`，而应改为对表达式直接调用`log`函数：`(bid/ask).log()`。详情参见[教程的说明](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/user_guide.md#避免用numpy函数处理orca对象)。

- Orca采用[惰性求值](https://github.com/dolphindb/Orca/blob/master/tutorial_cn/user_guide.md#orca并非总是立刻求值)策略，对于Orca的表达式，调用`compute`函数触发计算获得最终结果。

下面介绍在Orca中计算该因子的具体步骤。

### 1.建立数据库连接

在Orca中通过`connect`函数连接到DolphinDB服务器：

```Python
>>> import dolphindb.orca as orca
>>> orca.connect(MY_HOST, MY_PORT, MY_USERNAME, MY_PASSWORD)
```

### 2.导入数据

```Python
>>> DATA_DIR = "/dolphindb/database"  # e.g. data directory
>>> df = orca.read_csv(DATA_DIR + "/data/random.csv", >>> partitioned=False)
```

### 3.计算因子

```Python
>>> import numpy as np
>>> ask = df["av1"]
>>> bid = df["bv1"]
>>> p = df["mp"].iloc[0]
>>> for i in range(2, 11):
        ask += df["av"+str(i)] * np.exp(-10*(i-1)/p)
        bid += df["bv"+str(i)] * np.exp(-10*(i-1)/p)
>>> vol_diff = (0.5 * (bid/ask).log()).compute()
```

## Orca和pandas运算时间对比

在脚本添加入如下代码，分别记录Orca与pandas执行脚本所需时间：

```Python
>>> from contexttimer import Timer
>>> import time

>>> with Timer() as timer:
...     ask = df["av1"]
...     bid = df["bv1"]
...     p = df["mp"].iloc[0]
...     for i in range(2, 11):
...         ask += df["av" + str(i)] * np.exp(-10 * (i - 1) / p)
...         bid += df["bv" + str(i)] * np.exp(-10 * (i - 1) / p)
...     vol_diff = (0.5 * (bid / ask).log()).compute()

>>> timer.elapsed
# Orca output
0.6120876730419695

>>> with Timer() as timer:
...     ask = pdf["av1"]
...     bid = pdf["bv1"]
...     p = pdf["mp"].iloc[0]
...     for i in range(2, 11):
...         ask += np.exp(-10 * (i - 1) / p) * pdf["av" + str(i)]
...         bid += np.exp(-10 * (i - 1) / p) * pdf["bv" + str(i)]
...     vol_diff = 0.5 * np.log(bid / ask)

>>> timer.elapsed
# pandas output
2.0468595211859792
```

因子计算的实现，Orca和pandas都没用使用并行计算。Orca计算速度约为pandas的3倍。[点此](https://github.com/dolphindb/Orca/blob/master/examples/factor.py)查看完整的代码。