import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionSubTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    @property
    def pdf(self):
        return pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                            index=['circle', 'triangle', 'rectangle'])

    @property
    def odf(self):
        return orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                              index=['circle', 'triangle', 'rectangle'])

    def test_function_binary_operator_sub_dataframe_sub_with_scalar(self):
        assert_frame_equal(self.odf.to_pandas(), self.pdf)

        pre = self.pdf - 1
        ore = self.odf - 1
        assert_frame_equal(ore.to_pandas(), pre)

        pre = self.pdf.sub(1)
        ore = self.odf.sub(1)
        assert_frame_equal(ore.to_pandas(), pre)

    def test_function_binary_operator_sub_dataframe_sub_with_list(self):
        assert_frame_equal(self.odf.to_pandas(), self.pdf)

        pre = self.pdf - [1, 2]
        ore = (self.odf - [1, 2]).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.sub([1, 2])
        ore = self.odf.sub([1, 2]).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.sub([1, 2, 3], axis='index')
        ore = self.odf.sub([1, 2, 3], axis='index').to_pandas()
        # TODO：diffs
        # assert_frame_equal(ore, pre)

    def test_function_binary_operator_sub_dataframe_sub_with_series(self):
        assert_frame_equal(self.odf.to_pandas(), self.pdf)

        # TODO: defalt axis= 'columns' or 0: orca.DataFrame - orca.Series or orca.DataFrame.sub(orca.Series)
        # pre = pdf - pd.Series([1, 2], index=["angles", "degrees"])
        # ore = (odf - orca.Series([1, 2], index=["angles", "degrees"])).to_pandas()
        # assert_frame_equal(ore, pre)

        # pre = pdf.sub(pd.Series([1, 2], index=["angles","degrees"]))
        # ore = (odf.sub(orca.Series([1, 2], index=["angles","degrees"]))).to_pandas()
        # assert_frame_equal(ore, pre)

        pser = pd.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle'])
        oser = orca.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle'])
        pre = self.pdf.sub(pser, axis='index')
        ore = self.odf.sub(oser, axis='index').to_pandas()
        assert_frame_equal(ore, pre)