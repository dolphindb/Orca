import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionMeltTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_reshaping_sorting_transposing_melt_dataframe(self):
        pdf = pd.DataFrame({"time": pd.date_range("15:00:00", periods=9, freq="30s"),
                            "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                            "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                            "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                            "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                            "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
                            "F": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
        odf = orca.DataFrame(pdf)
        ptable=pdf.melt(id_vars=['A', 'B'], value_vars=['D', 'E'])
        otable = odf.melt(id_vars=['A', 'B'], value_vars=['D', 'E'])
        assert_frame_equal(otable.to_pandas(), ptable)

        ptable = pdf.melt(id_vars='A', value_vars='D', var_name="myVarname", value_name="myValname")
        otable = odf.melt(id_vars='A', value_vars='D', var_name="myVarname", value_name="myValname")
        assert_frame_equal(otable.to_pandas(), ptable)

        pdf = pd.DataFrame({'A': {0: 101, 1: 102, 2: 103}, 'B': {0: 1, 1: 3, 2: 5}, 'C': {0: 2, 1: 4, 2: 6}})
        odf = orca.DataFrame(pdf)
        ptable = pdf.melt()
        otable = odf.melt()
        assert_frame_equal(otable.to_pandas(), ptable)

        ptable = pdf.melt(id_vars=['A'])
        otable = odf.melt(id_vars=['A'])
        assert_frame_equal(otable.to_pandas(), ptable)

        ptable = pdf.melt(value_vars=['A'])
        otable = odf.melt(value_vars=['A'])
        assert_frame_equal(otable.to_pandas(), ptable)


if __name__ == '__main__':
    unittest.main()