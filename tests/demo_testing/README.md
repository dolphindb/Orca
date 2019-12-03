demo-testing目录下的脚本为实例脚本，用于测试orca项目的实用性。

首先需要在`setup`目录下新建settings.py文件，根据本机的DolphinDB server的配置情况设置所需参数：

```python
HOST = "localhost"
PORT = 8848
WORK_DIR = "/home/root/"
```