import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionPivotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_reshaping_sorting_transposing_pivot_dataframe(self):
        pdf = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                            'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                            'baz': [1, 2, 3, 4, 5, 6],
                            'zoo': [2, 4, 5, 6, 4, 7]})
        odf = orca.DataFrame(pdf)
        ptable = pdf.pivot(index='foo', columns='bar', values='baz')
        otable = odf.pivot(index='foo', columns='bar', values='baz')
        assert_frame_equal(otable.to_pandas(), ptable)

        # TODO:orca不支持多个values
        ptable = pdf.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
        # otable = odf.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
        # assert_frame_equal(otable.to_pandas(), ptable)
        # TODO:orcab不支持下标
        ptable = pdf.pivot(index='foo', columns='bar')['baz']
        # otable = odf.pivot(index='foo', columns='bar')['baz']
        # assert_frame_equal(otable.to_pandas(), ptable)
        # TODO:orca不支持index，columns参数为多个列组成的list


if __name__ == '__main__':
    unittest.main()