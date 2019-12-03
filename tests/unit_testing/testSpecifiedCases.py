import unittest
import os
from test_dataframe import DataFrameTest
from test_series import SeriesTest
from test_index import IndexTest
from test_indexing import IndexingTest
from test_groupby import GroupByTest
from test_resample import ResampleTest
from test_rolling import RollingTest
from test_where import WhereTest
from test_join import JoinTest
from test_plotting import PLottingTest

from test_series_partition import DfsSeriesTest
from test_groupby_partition import DfsGroupByTest
from test_resample_partition import DfsResampleTest
from test_rolling_partition import DfsRollingTest
from test_where_partition import DfsWhereTest
from test_join_partition import DfsJoinTest
from test_plotting_partition import DfsPLottingTest

test_cases = (DataFrameTest, SeriesTest, IndexTest, IndexingTest, GroupByTest, ResampleTest, RollingTest, WhereTest,
              JoinTest, PLottingTest, DfsSeriesTest, DfsGroupByTest, DfsResampleTest, DfsRollingTest, DfsWhereTest,
              DfsJoinTest, DfsPLottingTest,)


def whole_suite():
    # 创建测试加载器
    loader = unittest.TestLoader()
    # 创建测试包
    suite = unittest.TestSuite()
    # 遍历所有测试类
    for test_class in test_cases:
        # 从测试类中加载测试用例
        tests = loader.loadTestsFromTestCase(test_class)
        # 将测试用例添加到测试包中
        suite.addTests(tests)
    return suite


if __name__ == '__main__':
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))  # 获取当前工作目录
    # discover = unittest.defaultTestLoader.discover(test_dir, pattern="test_*.py")
    with open(test_dir + "/../reports/unit_report.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        # verbosity = 0 只输出error和fail的traceback
        # verbosity = 1 在0的基础上，在第一行给出每一个用例执行的结果的标识，成功是.，失败是F，出错是E，跳过是S，
        #               类似：.EEEEEEEEEEEEEE.EE.E.EEE.....F......F..
        # verbosity = 2 输出测试结果的所有信息
        runner.run(whole_suite())
