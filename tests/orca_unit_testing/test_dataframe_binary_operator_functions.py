import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class DataFrameBinaryOPTest(unittest.TestCase):
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

    def test_dataframe_binary_operator_function_add_scalar(self):
        assert_frame_equal(self.odf.to_pandas(), self.pdf)

        pre = self.pdf + 1
        ore = self.odf + 1
        assert_frame_equal(ore.to_pandas(), pre)

        pre = self.pdf.add(1)
        ore = self.odf.add(1)
        assert_frame_equal(ore.to_pandas(), pre)

    def test_dataframe_binary_operator_function_add_list(self):
        assert_frame_equal(self.odf.to_pandas(), self.pdf)

        pre = self.pdf + [1, 2]
        ore = (self.odf + [1, 2]).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.add([1, 2])
        ore = self.odf.add([1, 2]).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.add([1, 2, 3], axis='index')
        ore = self.odf.add([1, 2, 3], axis='index').to_pandas()
        # TODO：diffs
        # assert_frame_equal(ore, pre)

    def test_dataframe_binary_operator_function_add_series(self):
        assert_frame_equal(self.odf.to_pandas(), self.pdf)

        # TODO: defalt axis= 'columns' or 0: orca.DataFrame + orca.Series or orca.DataFrame.add(orca.Series)
        # pre = pdf + pd.Series([1, 2], index=["angles", "degrees"])
        # ore = (odf + orca.Series([1, 2], index=["angles", "degrees"])).to_pandas()
        # assert_frame_equal(ore, pre)

        # pre = pdf.add(pd.Series([1, 2], index=["angles","degrees"]))
        # ore = (odf.add(orca.Series([1, 2], index=["angles","degrees"]))).to_pandas()
        # assert_frame_equal(ore, pre)

        pser = pd.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle'])
        oser = orca.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle'])
        pre = self.pdf.add(pser, axis='index')
        ore = self.odf.add(oser, axis='index').to_pandas()
        assert_frame_equal(ore, pre)

    def test_dataframe_binary_operator_function_sub_scalar(self):
        assert_frame_equal(self.odf.to_pandas(), self.pdf)

        pre = self.pdf - 1
        ore = (self.odf - 1).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.sub(1)
        ore = self.odf.sub(1).to_pandas()
        assert_frame_equal(ore, pre)

    def test_dataframe_binary_operator_function_sub_list(self):
        assert_frame_equal(self.odf.to_pandas(), self.pdf)

        pre = self.pdf - [1, 2]
        ore = (self.odf - [1, 2]).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.sub([1, 2])
        ore = self.odf.sub([1, 2]).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.sub([1, 2, 3], axis='index')
        ore = self.odf.sub([1, 2, 3], axis='index').to_pandas()
        # TODO：NOT IMPLEMENTED
        # assert_frame_equal(ore, pre)

    def test_dataframe_binary_operator_function_sub_series(self):
        # TODO: defalt axis= 'columns' or 0: orca.DataFrame - orca.Series or orca.DataFrame.sub(orca.Series)
        # pre = pdf - pd.Series([1, 2, 3])
        # ore = (odf - orca.Series([1, 2, 3])).to_pandas()
        # assert_frame_equal(ore, pre)

        # pre = pdf.sub(pd.Series([1, 2, 3]))
        # ore = (odf.sub(orca.Series([1, 2, 3]))).to_pandas()
        # assert_frame_equal(ore, pre)

        pser = pd.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle'])
        oser = orca.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle'])
        pre = self.pdf.sub(pser, axis='index')
        ore = self.odf.sub(oser, axis='index').to_pandas()
        assert_frame_equal(ore, pre)

    def test_dataframe_binary_operator_function_mul_dataframe(self):
        p_other = pd.DataFrame({'angles': [0, 3, 4]}, index=['circle', 'triangle', 'rectangle'])
        o_other = orca.DataFrame({'angles': [0, 3, 4]}, index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(o_other.to_pandas(), p_other)

        p_index = pd.DataFrame({'angles': [3, 5, 8], 'degrees': [2, 5, 7]}, index=['circle', 'triangle', 'rectangle'])
        o_index = orca.DataFrame({'angles': [3, 5, 8], 'degrees': [2, 5, 7]}, index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(o_index.to_pandas(), p_index)

        pre = self.pdf * p_other
        ore = (self.odf * o_other).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.mul(p_other)
        ore = self.odf.mul(o_other).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.mul(p_index)
        ore = self.odf.mul(o_index).to_pandas()
        assert_frame_equal(ore, pre)

    def test_dataframe_binary_operator_function_div_scalar(self):
        pre = self.pdf / 2
        ore = (self.odf / 2).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.div(2)
        ore = self.odf.div(2).to_pandas()
        assert_frame_equal(ore, pre)

    def test_dataframe_binary_operator_function_rdiv_scalar(self):
        pre = self.pdf.rdiv(2)
        ore = self.odf.rdiv(2).to_pandas()
        # 2/0 in pandas equals to inf while in orca equals to nan,
        # thus we replace these values with zeros for correctness assertion
        pre[np.isinf(pre)] = 0
        ore[np.isnan(ore)] = 0
        assert_frame_equal(ore, pre)

    def test_dataframe_binary_operator_function_div_multiIndex_param_level_param_fill_value(self):
        pdf_multi = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6], 'degrees': [360, 180, 360, 360, 540, 720]},
                                 index=[['A', 'A', 'A', 'B', 'B', 'B'],
                                        ['circle', 'triangle', 'rectangle', 'square', 'pentagon', 'hexagon']])
        odf_multi = orca.DataFrame({'angles': [0, 3, 4, 4, 5, 6], 'degrees': [360, 180, 360, 360, 540, 720]},
                                   index=[['A', 'A', 'A', 'B', 'B', 'B'],
                                          ['circle', 'triangle', 'rectangle', 'square', 'pentagon', 'hexagon']])
        assert_frame_equal(odf_multi.to_pandas(), pdf_multi)

        # TODO: orca.DataFrame.rdiv(orca.DataFrame, level=1, fill_value=0)
        # pre = self.pdf_sh.rdiv(pdf_multi, level=1, fill_value=0)
        # ore = self.odf_sh.rdiv(odf_multi, level=1, fill_value=0).to_pandas()
        # assert_frame_equal(ore, pre)

    def test_dataframe_binary_operator_function_truediv(self):
        pre = self.pdf.truediv(2)
        ore = (self.odf.truediv(2)).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.truediv(2, axis='index')
        ore = (self.odf.truediv(2, axis='index')).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.truediv(2, fill_value=0)
        ore = (self.odf.truediv(2, fill_value=0)).to_pandas()
        assert_frame_equal(ore, pre)

        pdf_multi = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6], 'degrees': [360, 180, 360, 360, 540, 720]},
                                 index=[['A', 'A', 'A', 'B', 'B', 'B'],
                                        ['circle', 'triangle', 'rectangle', 'square', 'pentagon', 'hexagon']])
        odf_multi = orca.DataFrame({'angles': [0, 3, 4, 4, 5, 6], 'degrees': [360, 180, 360, 360, 540, 720]},
                                   index=[['A', 'A', 'A', 'B', 'B', 'B'],
                                          ['circle', 'triangle', 'rectangle', 'square', 'pentagon', 'hexagon']])
        assert_frame_equal(pdf_multi, odf_multi.to_pandas())
        # TODO: BUG cannot join with no overlapping index names
        # pre = self.pdf_sh.truediv(pdf_multi, level=1, fill_value=0)
        # ore = self.odf_sh.truediv(odf_multi, level=1, fill_value=0).to_pandas()
        # assert_frame_equal(ore, pre)

    def test_dataframe_binary_operator_function_compare(self):
        df = pd.DataFrame({'cost': [250, 150, 100],
                           'revenue': [100, 250, 300]},
                          index=['A', 'B', 'C'])
        assert_frame_equal((orca.DataFrame(df) == 100).to_pandas(), df == 100)
        assert_frame_equal((orca.DataFrame(df).eq(100)).to_pandas(), df.eq(100))
        assert_frame_equal((orca.DataFrame(df).ne(100)).to_pandas(), df.ne(100))

        other = pd.DataFrame({'revenue': [300, 250, 100, 150]},
                             index=['A', 'B', 'C', 'D'])

        # TODO BUG pandas result is False while orca result is nan
        # assert_frame_equal((orca.DataFrame(df).gt(orca.DataFrame(other))).to_pandas(), df.gt(other),
        #                    check_dtype=False)
        # assert_frame_equal((orca.DataFrame(df).eq(orca.DataFrame(other))).to_pandas(), df.eq(other),
        #                    check_dtype=False)
        # assert_frame_equal((orca.DataFrame(df).ne(orca.DataFrame(other))).to_pandas(), df.ne(other),
        #                    check_dtype=False)

    def test_dataframe_binary_operator_function_mod(self):
        odf = orca.DataFrame(self.pdf)

        assert_frame_equal(odf.mod(2).to_pandas(), self.pdf.mod(2))

        # TODO NOT IMPLEMENTED ERROR
        # pre = pdf.mod(pd.Series([1, 2], index=["angles", "degrees"]))
        # ore = (odf.mod(orca.Series([1, 2], index=["angles", "degrees"]))).to_pandas()
        # assert_frame_equal(ore, pre)

    def test_dataframe_binary_operator_function_pow(self):
        odf = orca.DataFrame(self.pdf)

        assert_frame_equal(odf.pow(2).to_pandas(), self.pdf.pow(2), check_dtype=False)

        # TODO NOT IMPLEMENTED ERROR
        # pre = pdf.pow(pd.Series([1, 2], index=["angles", "degrees"]))
        # ore = (odf.pow(orca.Series([1, 2], index=["angles", "degrees"]))).to_pandas()
        # assert_frame_equal(ore, pre)

    def test_dataframe_binary_operator_function_combine(self):
        # TODO NOT IMPLEMENTED ERROR
        pass


if __name__ == '__main__':
    unittest.main()
