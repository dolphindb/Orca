import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionPivotTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_reshaping_sorting_transposing_pivot_table_dataframe(self):
        pdf = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                            "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                            "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                            "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                            "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
        odf = orca.DataFrame(pdf)
        ptable = pdf.pivot_table(values='D', index='A', columns='C', aggfunc="sum")
        otable = odf.pivot_table(values='D', index='A', columns='C', aggfunc="sum")
        assert_frame_equal(otable.to_pandas(), ptable)

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
        assert_frame_equal(otable.to_pandas(), ptable)
        ptable = pdf.pivot_table(values='D', index='A', columns=pdf.time.dt.minute, aggfunc="sum")
        otable = odf.pivot_table(values='D', index='A', columns=odf.time.dt.minute, aggfunc="sum")
        # TODO:DIFFERENT COLNAME
        # assert_frame_equal(otable.to_pandas(), ptable)

        # TODO:orca不支持index，columns参数为多个列组成的list


if __name__ == '__main__':
    unittest.main()