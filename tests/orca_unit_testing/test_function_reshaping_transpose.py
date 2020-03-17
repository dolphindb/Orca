import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionTransposeTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_reshaping_sorting_transposing_transpose_T_dataframe(self):
        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.T.to_pandas(), pdf.T)
        assert_frame_equal(odf.transpose().to_pandas(), pdf.transpose())
        assert_frame_equal(odf.transpose(copy=True).to_pandas(), pdf.transpose(copy=True))

        # TODO：transpose on non-numerical types of matrix is not allowed in Orca
        pdf = pd.DataFrame({'name': ['Alice', 'Bob'], 'grade': ['A', 'B'], 'Gender': ['female', 'male']})
        odf = orca.DataFrame(pdf)


if __name__ == '__main__':
    unittest.main()