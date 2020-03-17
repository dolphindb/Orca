import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionStackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_reshaping_sorting_transposing_stack_dataframe(self):
        pdf = pd.DataFrame([[0, 1], [2, 3]], index=['cat', 'dog'], columns=['weight', 'height'])
        odf = orca.DataFrame(pdf)
        # TODO: orca的结果不分组
        # assert_series_equal(odf.stack().to_pandas(), pdf.stack())
        pdf = pd.DataFrame([[1, 2], [2, 4]], index=['cat', 'dog'],
                           columns=pd.MultiIndex.from_tuples([('weight', 'kg'), ('weight', 'pounds')]))
        odf = orca.DataFrame(pdf)
        # TODO： orca不支持multi level columns
        # assert_series_equal(odf.stack().to_pandas(), pdf.stack())
        pdf = pd.DataFrame({"time": pd.date_range("15:00:00", periods=9, freq="30s"),
                            "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                            "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                            "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                            "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                            "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
                            "F": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
        odf = orca.DataFrame(pdf)
        ptable = pdf.pivot_table(values='D', index=pdf.time.dt.minute, columns='A', aggfunc="sum")
        otable = odf.pivot_table(values='D', index=odf.time.dt.minute, columns='A', aggfunc="sum")
        # TODO： orca的结果不分组
        # assert_series_equal(otable.stack().to_pandas(), ptable.stack())


if __name__ == '__main__':
    unittest.main()