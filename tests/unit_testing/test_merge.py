import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv_left = None
    pdf_csv_right = None
    odf_csv_left = None
    odf_csv_right = None


class MergeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        left_fileName = 'test_merge_left_table.csv'
        right_fileName = 'test_merge_right_table.csv'

        data_left = os.path.join(DATA_DIR, left_fileName)
        data_left = data_left.replace('\\', '/')

        data_right = os.path.join(DATA_DIR, right_fileName)
        data_right = data_right.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        # import
        Csv.odf_csv_left = orca.read_csv(data_left)
        Csv.pdf_csv_left = pd.read_csv(data_left, parse_dates=[0, 1])
        Csv.odf_csv_right = orca.read_csv(data_right)
        Csv.pdf_csv_right = pd.read_csv(data_right)


    @property
    def odf_csv_left(self):
        return Csv.odf_csv_left

    @property
    def odf_csv_right(self):
        return Csv.odf_csv_right

    @property
    def pdf_csv_left(self):
        return Csv.pdf_csv_left

    @property
    def pdf_csv_right(self):
        return Csv.pdf_csv_right

    @property
    def odf_csv_left_index(self):
        return Csv.odf_csv_left.set_index("type")

    @property
    def odf_csv_right_index(self):
        return Csv.odf_csv_right.set_index("type")

    @property
    def pdf_csv_left_index(self):
        return Csv.pdf_csv_left.set_index("type")

    @property
    def pdf_csv_right_index(self):
        return Csv.pdf_csv_right.set_index("type")

    @property
    def pdf_series_right(self):
        return Csv.pdf_series_left

    def test_assert_original_dataframe_equal(self):
        assert_frame_equal(self.odf_csv_left.to_pandas(), self.pdf_csv_left, check_dtype=False)
        assert_frame_equal(self.odf_csv_right.to_pandas(), self.pdf_csv_right, check_dtype=False)
        assert_frame_equal(self.odf_csv_left_index.to_pandas(), self.pdf_csv_left_index, check_dtype=False)
        assert_frame_equal(self.odf_csv_right_index.to_pandas(), self.pdf_csv_right_index, check_dtype=False)

    def test_merge_from_csv_param_suffix(self):
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, on="type",suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, on="type",suffixes=('_left', '_right'))
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.sort_values("date").to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_csv_param_how(self):
        # how = left
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, how="left", on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="left", on="type")
        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False)
        # how = right
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, how="right", on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="right", on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # default how = inner
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, how="inner", on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="inner", on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # how = outer
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, how="outer", on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="outer", on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_csv_param_on(self):
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_csv_param_leftonrighton(self):
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, left_on="type", right_on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, left_on="type", right_on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_csv_param_index(self):
        odf_merge = self.odf_csv_left_index.merge(self.odf_csv_right_index, left_index=True, right_index=True)
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, left_index=True, right_index=True)
        assert_frame_equal(odf_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False)

    def test_merge_from_csv_index_param_suffix(self):
        # TODO: 对于string类型的空值，orca的空字符串即为空值，而pandas不认为空字符串为空值
        odf_merge = self.odf_csv_left_index.merge(self.odf_csv_right_index, left_index=True, right_index=True, suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, left_index=True, right_index=True, suffixes=('_left', '_right'))
        assert_frame_equal(odf_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False)

    def test_merge_from_csv_index_param_on(self):
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right_index, left_on="type", right_index = True)
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right_index, left_on="type", right_index = True)
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.sort_index().to_pandas(), pdf_merge.sort_index(), check_dtype=False, check_like= False)

        odf_merge = self.odf_csv_left_index.merge(self.odf_csv_right, right_on="type", left_index=True)
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right, right_on="type", left_index=True)
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.sort_index().to_pandas(), pdf_merge.sort_index(), check_dtype=False, check_like= False)

    def test_merge_from_csv_index_param_how(self):
        # how = left
        odf_merge = self.odf_csv_left_index.merge(self.odf_csv_right_index, how="left", left_index=True, right_index=True)
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="left", left_index=True, right_index=True)
        assert_frame_equal(odf_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False)
        # how = right
        odf_merge = self.odf_csv_left_index.merge(self.odf_csv_right_index, how="right", left_index=True, right_index=True)
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="right", left_index=True, right_index=True)
        assert_frame_equal(odf_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False)

        # default how = inner
        odf_merge = self.odf_csv_left_index.merge(self.odf_csv_right_index, left_index=True, right_index=True)
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, left_index=True, right_index=True)
        assert_frame_equal(odf_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False)

        # how = outer
        odf_merge = self.odf_csv_left_index.merge(self.odf_csv_right_index, how="outer", left_index=True, right_index=True)
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="outer", left_index=True, right_index=True)
        assert_frame_equal(odf_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False)

    def test_merge_from_csv_param_suffix_param_how(self):
        # how = left
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, how="left", on="type", suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="left", on="type", suffixes=('_left', '_right'))
        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)
        # how = right
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, how="right", on="type", suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="right", on="type", suffixes=('_left', '_right'))
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # default how = inner
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, on="type", suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, on="type", suffixes=('_left', '_right'))
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # how = outer
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, how="outer", on="type", suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="outer", on="type", suffixes=('_left', '_right'))
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_csv_param_how_param_leftonrighton(self):
        # how = left
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, how="left", left_on="type", right_on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="left", left_on="type", right_on="type")
        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)
        # how = right
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, how="right", left_on="type", right_on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="right", left_on="type", right_on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # default how = inner
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, left_on="type", right_on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, left_on="type", right_on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # how = outer
        odf_merge = self.odf_csv_left.merge(self.odf_csv_right, how="outer", left_on="type", right_on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="outer", left_on="type", right_on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_csv_index_param_suffix_param_how(self):
        # how = left
        odf_merge = self.odf_csv_left_index.merge(self.odf_csv_right_index, how="left", left_index=True,
                                                    right_index=True, suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="left", left_index=True,
                                                  right_index=True, suffixes=('_left', '_right'))
        assert_frame_equal(odf_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False, check_like=True)
        # how = right
        odf_merge = self.odf_csv_left_index.merge(self.odf_csv_right_index, how="right", left_index=True,
                                                    right_index=True, suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="right", left_index=True,
                                                  right_index=True, suffixes=('_left', '_right'))
        assert_frame_equal(odf_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # default how = inner
        odf_merge = self.odf_csv_left_index.merge(self.odf_csv_right_index, left_index=True, right_index=True,
                                                    suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, left_index=True, right_index=True,
                                                  suffixes=('_left', '_right'))
        assert_frame_equal(odf_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # how = outer
        odf_merge = self.odf_csv_left_index.merge(self.odf_csv_right_index, how="outer", left_index=True,
                                                    right_index=True, suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="outer", left_index=True,
                                                  right_index=True, suffixes=('_left', '_right'))
        assert_frame_equal(odf_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_dataframe_param_suffix(self):
        odf = orca.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})
        odf_other = orca.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})
        odf_merge = odf.merge(odf_other, on="key", suffixes=('_left', '_right'))

        pdf = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})
        pdf_other = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})
        pdf_merge = pdf.merge(pdf_other, on="key", suffixes=('_left', '_right'))

        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False)

    def test_merge_from_dataframe_param_leftonrighton(self):
        odf = orca.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})
        odf_other = orca.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})
        odf_merge = odf.merge(odf_other, left_on = "key",right_on="key")

        pdf = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})
        pdf_other = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})
        pdf_merge = pdf.merge(pdf_other, left_on = "key",right_on="key")

        # pdf_merge.loc[:, 'key_other'].fillna("", inplace=True)
        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False)

    def test_merge_from_dataframe_how(self):
        odf = orca.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})
        odf_other = orca.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})

        pdf = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})
        pdf_other = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})

        # how = left
        odf_merge = odf.merge(odf_other, how="left", on='key')
        pdf_merge = pdf.merge(pdf_other, how="left", on="key")
        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False)
        # how = right
        odf_merge = odf.merge(odf_other, how="right",on = 'key')
        pdf_merge = pdf.merge(pdf_other, how="right",on = "key")
        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False)

        # how = inner
        odf_merge = odf.merge(odf_other, how="inner",on = 'key')
        pdf_merge = pdf.merge(pdf_other, how="inner",on = 'key')
        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False)

        # how = outer
        odf_merge = odf.merge(odf_other, how="outer",on = 'key')
        pdf_merge = pdf.merge(pdf_other, how="outer",on = 'key')
        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False)

    def test_merge_from_dataframe_index(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['K0', 'K1', 'K2'])
        orca_right = orca.DataFrame({'C': [111, 222, 333], 'D': [1111, 2222, 3333]}, index=['K0', 'K2', 'K3'])
        odf_merge = orca_left.merge(orca_right,left_index=True,right_index=True)

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['K0', 'K1', 'K2'])
        pd_right = pd.DataFrame({'C': [111, 222, 333], 'D': [1111, 2222, 3333]}, index=['K0', 'K2', 'K3'])
        pdf_merge = pd_left.merge(pd_right, left_index=True, right_index=True)

        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False)

    def test_merge_from_dataframe_index_param_how(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['K0', 'K1', 'K2'])
        orca_right = orca.DataFrame({'C': [111, 222, 333], 'D': [1111, 2222, 3333]}, index=['K0', 'K2', 'K3'])

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['K0', 'K1', 'K2'])
        pd_right = pd.DataFrame({'C': [111, 222, 333], 'D': [1111, 2222, 3333]}, index=['K0', 'K2', 'K3'])

        # by default, how = left

        # how = right
        odf_merge = orca_left.merge(orca_right, how="right",left_index=True,right_index=True)
        pdf_merge = pd_left.merge(pd_right, how="right",left_index=True,right_index=True)
        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False)

        # how = inner
        odf_merge = orca_left.merge(orca_right, how="inner",left_index=True,right_index=True)
        pdf_merge = pd_left.merge(pd_right, how="inner",left_index=True,right_index=True)
        # pdf_merge.loc[:, 'key_other'].fillna("", inplace=True)
        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False)

        # how = outer
        odf_merge = orca_left.merge(orca_right, how="outer",left_index=True,right_index=True)
        pdf_merge = pd_left.merge(pd_right, how="outer",left_index=True,right_index=True)
        # pdf_merge.loc[:, 'key_other'].fillna("", inplace=True)
        assert_frame_equal(odf_merge.to_pandas(), pdf_merge, check_dtype=False)

if __name__ == '__main__':
    unittest.main()
