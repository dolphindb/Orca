benchmark目录下的脚本为性能测试脚本，用于测试orca项目的性能。

demo_testing目录下的脚本为实例脚本，用于测试orca项目的实用性。

unit_testing目录下的脚本为单元测试脚本，覆盖orca提供的各种功能。

**注意事项**

- 需要先在这三个目录下的setup目录下创建settings.py文件，根据本机的DolphinDB server的配置情况设置所需参数：

```python
HOST = "localhost"
PORT = 8848
WORK_DIR = "/home/root/"
```

- linux下命令行执行测试脚本需要先配置orca项目根目录路径,例如：

```bash
$ cd demo_testing
$ PYTHONPATH=/home/usr/orca python demo1.py
```
- windows下以类似的方式设置路径