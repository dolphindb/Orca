benchmark目录下的脚本为性能测试脚本，用于测试orca项目的性能。由于性能测试所需的数据集较大，故需要手动生成数据。

首先分别在`testPerformance/setup`目录和`originalCases/setup`目录下新建settings.py文件，根据本机的DolphinDB server的配置情况设置所需参数：

```python
HOST = "localhost"
PORT = 8848
WORK_DIR = "/home/root/"
```

### 1. 执行`dataGeerator`目录下的两个文件生成原始数据

```console
$ cd dataGeerator/
$ ls
tickerCsvGenerator.py  valueCsvGenerator.py
```
首先通过如下指令生成ticker数据，参数指定生成文件的行数：

```bash
$ python tickerCsvGenerator.py 10000
```

指定生成文件的行数，生成value数据,（生成value表主要是为了进行join操作）。请注意，value文件是基于ticker文件中股票的名字，经过乱序排序以后生成的，因此需要先生成ticker文件，再生成value文件

```bash
$ python valueCsvGenerator.py 10000
```

生成的这两个文件自动存放在`/benchmark/testPerformance/setup/data`目录下
 
### 2.执行`testPerformance`目录下的driver文件

```console
$ cd testPerformance/
$ ls
columns.py  driver.py  orca_driver.py  orca_partition_driver.py  pandas_driver.py  __pycache__  setup
```

driver文件需要指定两个参数：第一个参数为文件中记录的条数，第二个参数为进行测试的工具，可以是pandas，orca或者orcapartition。
在命令行中执行下列指令分别测试性能

```bash
/testPerformance$ PYTHONPATH=/YOUR_ORCA_DIR/orca python driver.py 10000 pandas
/testPerformance$ PYTHONPATH=/YOUR_ORCA_DIR/orca python driver.py 10000 orca
/testPerformance$ PYTHONPATH=/YOUR_ORCA_DIR/orca python driver.py 10000 orcapartition
```

测试完成后，可以在/tests/reports目录下查看性能测试报告。

目前测试文件比较简单，后续会增加更多列，或者直接使用金融数据进行测试。

### 3.`originalCases`目录下的测试脚本直接点击运行即可，但是由于数据集太大，并不在这个gitlab项目中。