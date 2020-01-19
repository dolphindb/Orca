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


class JoinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        left_fileName = 'test_join_left_table.csv'
        right_fileName = 'test_join_right_table.csv'

        data_left = os.path.join(DATA_DIR, left_fileName)
        data_left = data_left.replace('\\', '/')

        data_right = os.path.join(DATA_DIR, right_fileName)
        data_right = data_right.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        # import
        Csv.odf_csv_left = orca.read_csv(data_left, dtype={"TRDSTAT": "SYMBOL"})
        Csv.pdf_csv_left = pd.read_csv(data_left, parse_dates=[1])

        Csv.odf_csv_right = orca.read_csv(data_right, dtype={"TRDSTAT": "SYMBOL"})
        Csv.pdf_csv_right = pd.read_csv(data_right, parse_dates=[0])


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
        return Csv.odf_csv_left.set_index("date")

    @property
    def odf_csv_right_index(self):
        return Csv.odf_csv_right.set_index("date")

    @property
    def pdf_csv_left_index(self):
        return Csv.pdf_csv_left.set_index("date")

    @property
    def pdf_csv_right_index(self):
        return Csv.pdf_csv_right.set_index("date")

    def test_join_assert_original_dataframe_equal(self):
        assert_frame_equal(self.odf_csv_left.to_pandas(), self.pdf_csv_left, check_dtype=False)
        assert_frame_equal(self.odf_csv_right.to_pandas(), self.pdf_csv_right, check_dtype=False)
        assert_frame_equal(self.odf_csv_left_index.to_pandas(), self.pdf_csv_left_index, check_dtype=False)
        assert_frame_equal(self.odf_csv_right_index.to_pandas(), self.pdf_csv_right_index, check_dtype=False)

    def test_join_from_csv_param_lsuffix_rsuffix(self):
        odf_join = self.odf_csv_left.join(self.odf_csv_right, lsuffix='_caller', rsuffix='_other')
        pdf_join = self.pdf_csv_left.join(self.pdf_csv_right, lsuffix='_caller', rsuffix='_other')
        pdf_join.loc[:, 'TICKER_other'].fillna("", inplace=True)
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_csv_param_lsuffix_rsuffix_how(self):
        # by default, how = left

        # how = right
        assert_frame_equal(
            self.odf_csv_left.join(self.odf_csv_right, how="right", lsuffix='_caller', rsuffix='_other').to_pandas(),
            self.pdf_csv_left.join(self.pdf_csv_right, how="right", lsuffix='_caller', rsuffix='_other'),
            check_dtype=False)

        # how = inner
        assert_frame_equal(
            self.odf_csv_left.join(self.odf_csv_right, how="inner", lsuffix='_caller', rsuffix='_other').to_pandas(),
            self.pdf_csv_left.join(self.pdf_csv_right, how="inner", lsuffix='_caller', rsuffix='_other'),
            check_dtype=False)

        # how = outer
        odf_join = self.odf_csv_left.join(self.odf_csv_right, how="outer", lsuffix='_caller', rsuffix='_other')
        pdf_join = self.pdf_csv_left.join(self.pdf_csv_right, how="outer", lsuffix='_caller', rsuffix='_other')
        pdf_join.loc[:, 'TICKER_other'].fillna("", inplace=True)
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_csv_param_lsuffix_rsuffix_sort(self):
        # TODO：NOT IMPLEMENTED
        pd_ll = self.pdf_csv_right.sample(frac=1)
        orca_ll = orca.DataFrame(pd_ll)
        # odf_join = orca_ll.join(self.odf_csv_right, lsuffix='_caller', rsuffix='_other', sort=True)
        pdf_join = pd_ll.join(self.pdf_csv_right, lsuffix='_caller', rsuffix='_other', sort=True)
        pdf_join.loc[:, 'TICKER_other'].fillna("", inplace=True)
        # assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_csv_index_param_lsuffix_rsuffix(self):
        # TODO: 对于string类型的空值，orca的空字符串即为空值，而pandas不认为空字符串为空值
        odf_join = self.odf_csv_left_index.join(self.odf_csv_right_index, lsuffix='_caller', rsuffix='_other')
        pdf_join = self.pdf_csv_left_index.join(self.pdf_csv_right_index, lsuffix='_caller', rsuffix='_other')
        pdf_join.loc[:, 'TICKER_other'].fillna("", inplace=True)
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_csv_index_param_lsuffix_rsuffix_how(self):
        # by default, how = left

        # # how = right
        odf_join = self.odf_csv_left_index.join(self.odf_csv_right_index, how="right", lsuffix='_caller',
                                                rsuffix='_other')
        pdf_join = self.pdf_csv_left_index.join(self.pdf_csv_right_index, how="right", lsuffix='_caller',
                                                rsuffix='_other')
        pdf_join.iloc[:, 2].fillna("", inplace=True)
        pdf_join.iloc[:, 3].fillna("", inplace=True)
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

        # how = inner
        assert_frame_equal(
            self.odf_csv_left_index.join(self.odf_csv_right_index, how="inner", lsuffix='_caller',
                                         rsuffix='_other').to_pandas(),
            self.pdf_csv_left_index.join(self.pdf_csv_right_index, how="inner", lsuffix='_caller', rsuffix='_other'),
            check_dtype=False)

        # how = outer
        odf_join = self.odf_csv_left_index.join(self.odf_csv_right_index, how="outer", lsuffix='_caller',
                                                rsuffix='_other')
        pdf_join = self.pdf_csv_left_index.join(self.pdf_csv_right_index, how="outer", lsuffix='_caller',
                                                rsuffix='_other')
        pdf_join.iloc[:, 2].fillna("", inplace=True)
        pdf_join.iloc[:, 3].fillna("", inplace=True)
        pdf_join.iloc[:, 11].fillna("", inplace=True)
        assert_frame_equal(odf_join.sort_index().to_pandas(), pdf_join.sort_index(), check_dtype=False)

    def test_join_from_csv_index_param_lsuffix_rsuffix_on(self):
        odf_join = self.odf_csv_left.join(self.odf_csv_right_index, on="date", lsuffix='_caller', rsuffix='_other')
        pdf_join = self.pdf_csv_left.join(self.pdf_csv_right_index, on="date", lsuffix='_caller', rsuffix='_other')
        pdf_join.loc[:, 'TICKER_other'].fillna("", inplace=True)
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

        odf_csv_ri = self.odf_csv_right.set_index('TICKER')
        pdf_csv_ri = self.pdf_csv_right.set_index('TICKER')

        assert_frame_equal(
            self.odf_csv_left.join(odf_csv_ri, on="TICKER", lsuffix='_caller', rsuffix='_other').to_pandas(),
            self.pdf_csv_left.join(pdf_csv_ri, on="TICKER", lsuffix='_caller', rsuffix='_other'),
            check_dtype=False)

    def test_join_from_csv_index_param_lsuffix_rsuffix_on_how(self):
        # by default, how = left

        # how = right
        odf_join = self.odf_csv_left.join(self.odf_csv_right_index, on="date", how="right", lsuffix='_caller', rsuffix='_other')
        pdf_join = self.pdf_csv_left.join(self.pdf_csv_right_index, on="date", how="right", lsuffix='_caller', rsuffix='_other')

        assert_frame_equal(odf_join[odf_join.index.notnull()].to_pandas(), pdf_join[pdf_join.index.notnull()],
                           check_dtype=False, check_index_type=False)

        # how = inner
        assert_frame_equal(
            self.odf_csv_left.join(self.odf_csv_right_index, on="date", how="inner", lsuffix='_caller', rsuffix='_other').to_pandas(),
            self.pdf_csv_left.join(self.pdf_csv_right_index, on="date", how="inner", lsuffix='_caller', rsuffix='_other'),
            check_dtype=False)

        # how = outer
        odf_join = self.odf_csv_left.join(self.odf_csv_right_index, on="date", how="outer",
                                          lsuffix='_caller', rsuffix='_other')
        pdf_join = self.pdf_csv_left.join(self.pdf_csv_right_index, on="date", how="outer",
                                          lsuffix='_caller', rsuffix='_other')
        pdf_join.loc[:, 'TICKER_caller'].fillna("", inplace=True)
        pdf_join.loc[:, 'TICKER_other'].fillna("", inplace=True)
        assert_frame_equal(odf_join[odf_join.index.notnull()].to_pandas(), pdf_join[pdf_join.index.notnull()],
                           check_dtype=False, check_index_type=False)

    def test_join_from_csv_index_param_lsuffix_rsuffix_sort(self):
        # TODO：NOT IMPLEMENTED
        # odf_join = self.odf_csv_left_index.join(self.odf_csv_right_index, lsuffix='_caller',
        #                                      rsuffix='_other', sort=True)
        pdf_join = self.pdf_csv_left_index.join(self.pdf_csv_right_index, lsuffix='_caller',
                                                rsuffix='_other', sort=True)
        pdf_join.loc[:, 'TICKER_other'].fillna("", inplace=True)
        # assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_dataframe_param_lsuffix_rsuffix(self):
        odf = orca.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})
        odf_other = orca.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})
        odf_join = odf.join(odf_other, lsuffix='_caller', rsuffix='_other')

        pdf = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})
        pdf_other = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})
        pdf_join = pdf.join(pdf_other, lsuffix='_caller', rsuffix='_other')

        pdf_join.loc[:, 'key_other'].fillna("", inplace=True)
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_dataframe_param_lsuffix_rsuffix_how(self):
        odf = orca.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})
        odf_other = orca.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})

        pdf = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})
        pdf_other = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})

        # by default, how = left

        # how = right
        odf_join = odf.join(odf_other, how="right", lsuffix='_caller', rsuffix='_other')
        pdf_join = pdf.join(pdf_other, how="right", lsuffix='_caller', rsuffix='_other')
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

        # how = inner
        odf_join = odf.join(odf_other, how="inner", lsuffix='_caller', rsuffix='_other')
        pdf_join = pdf.join(pdf_other, how="inner", lsuffix='_caller', rsuffix='_other')
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

        # how = outer
        odf_join = odf.join(odf_other, how="outer", lsuffix='_caller', rsuffix='_other')
        pdf_join = pdf.join(pdf_other, how="outer", lsuffix='_caller', rsuffix='_other')
        pdf_join.loc[:, 'key_other'].fillna("", inplace=True)
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_dataframe_param_lsuffix_rsuffix_on(self):
        odf = orca.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})
        odf_other = orca.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})
        odf_join = odf.join(odf_other, on='A', lsuffix='_caller', rsuffix='_other')

        pdf = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})
        pdf_other = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})
        pdf_join = pdf.join(pdf_other, on='A', lsuffix='_caller', rsuffix='_other')

        pdf_join.loc[:, 'key_other'].fillna("", inplace=True)
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_dataframe_param_lsuffix_rsuffix_sort(self):
        pdf_other = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})
        odf_other = orca.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': [11, 22, 33]})
        pdf = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [1, 2, 3, 4, 5, 6]})

        pdf_ll = pdf.sample(frac=1)
        odf_ll = orca.DataFrame(pdf_ll)
        # TODO：NOT IMPLEMENTED
        # odf_join = odf_ll.join(odf_other, lsuffix='_caller', rsuffix='_other', sort=True)
        pdf_join = pdf_ll.join(pdf_other, lsuffix='_caller', rsuffix='_other', sort=True)

        pdf_join.loc[:, 'key_other'].fillna("", inplace=True)
        # assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_dataframe_index(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['K0', 'K1', 'K2'])
        orca_right = orca.DataFrame({'C': [111, 222, 333], 'D': [1111, 2222, 3333]}, index=['K0', 'K2', 'K3'])
        odf_join = orca_left.join(orca_right)

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['K0', 'K1', 'K2'])
        pd_right = pd.DataFrame({'C': [111, 222, 333], 'D': [1111, 2222, 3333]}, index=['K0', 'K2', 'K3'])
        pdf_join = pd_left.join(pd_right)

        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_dataframe_index_param_lsuffix_rsuffix_how(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['K0', 'K1', 'K2'])
        orca_right = orca.DataFrame({'C': [111, 222, 333], 'D': [1111, 2222, 3333]}, index=['K0', 'K2', 'K3'])

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['K0', 'K1', 'K2'])
        pd_right = pd.DataFrame({'C': [111, 222, 333], 'D': [1111, 2222, 3333]}, index=['K0', 'K2', 'K3'])

        # by default, how = left

        # how = right
        odf_join = orca_left.join(orca_right, how="right")
        pdf_join = pd_left.join(pd_right, how="right")
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

        # how = inner
        odf_join = orca_left.join(orca_right, how="inner")
        pdf_join = pd_left.join(pd_right, how="inner")
        # pdf_join.loc[:, 'key_other'].fillna("", inplace=True)
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

        # how = outer
        odf_join = orca_left.join(orca_right, how="outer")
        pdf_join = pd_left.join(pd_right, how="outer")
        # pdf_join.loc[:, 'key_other'].fillna("", inplace=True)
        assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_dataframe_index_param_lsuffix_rsuffix_sort(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['K0', 'K1', 'K2'])
        orca_right = orca.DataFrame({'C': [111, 222, 333], 'D': [1111, 2222, 3333]}, index=['K0', 'K2', 'K3'])
        # TODO：NOT IMPLEMENTED
        # odf_join = orca_left.join(orca_right, sort=True)

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['K0', 'K1', 'K2'])
        pd_right = pd.DataFrame({'C': [111, 222, 333], 'D': [1111, 2222, 3333]}, index=['K0', 'K2', 'K3'])
        pdf_join = pd_left.join(pd_right, sort=True)

        # assert_frame_equal(odf_join.to_pandas(), pdf_join, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
