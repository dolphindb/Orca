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


class MergeAsofTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        left_fileName = 'test_merge_asof_left_table.csv'
        right_fileName = 'test_merge_asof_right_table.csv'

        data_left = os.path.join(DATA_DIR, left_fileName)
        data_left = data_left.replace('\\', '/')

        data_right = os.path.join(DATA_DIR, right_fileName)
        data_right = data_right.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        # import
        Csv.odf_csv_left = orca.read_csv(data_left)
        Csv.pdf_csv_left = pd.read_csv(data_left, parse_dates=[0])
        Csv.odf_csv_right = orca.read_csv(data_right)
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

    @property
    def odf_bid_csv_left(self):
        return self.odf_csv_left.sort_values(by=['bid', 'date']).reset_index(drop=True)

    @property
    def odf_bid_csv_right(self):
        return self.odf_csv_right.sort_values(by=['bid', 'date']).reset_index(drop=True)

    @property
    def pdf_bid_csv_left(self):
        return self.pdf_csv_left.sort_values(by=['bid', 'date']).reset_index(drop=True)

    @property
    def pdf_bid_csv_right(self):
        return self.pdf_csv_right.sort_values(by=['bid', 'date']).reset_index(drop=True)

    @property
    def odf_bid_csv_left_index(self):
        return self.odf_csv_left.sort_values(by=['bid', 'date']).set_index('bid')

    @property
    def odf_bid_csv_right_index(self):
        return self.odf_csv_right.sort_values(by=['bid', 'date']).set_index('bid')

    @property
    def pdf_bid_csv_left_index(self):
        return self.pdf_csv_left.sort_values(by=['bid', 'date']).set_index('bid')

    @property
    def pdf_bid_csv_right_index(self):
        return self.pdf_csv_right.sort_values(by=['bid', 'date']).set_index('bid')

    def test_assert_original_dataframe_equal(self):
        assert_frame_equal(self.odf_csv_left.to_pandas(), self.pdf_csv_left, check_dtype=False)
        assert_frame_equal(self.odf_csv_right.to_pandas(), self.pdf_csv_right, check_dtype=False)
        assert_frame_equal(self.odf_csv_left_index.to_pandas(), self.pdf_csv_left_index, check_dtype=False)
        assert_frame_equal(self.odf_csv_right_index.to_pandas(), self.pdf_csv_right_index, check_dtype=False)
        assert_frame_equal(self.odf_bid_csv_left.to_pandas(), self.pdf_bid_csv_left, check_dtype=False)
        assert_frame_equal(self.odf_bid_csv_right.to_pandas(), self.pdf_bid_csv_right, check_dtype=False)
        assert_frame_equal(self.odf_bid_csv_left_index.to_pandas(), self.pdf_bid_csv_left_index, check_dtype=False)
        assert_frame_equal(self.odf_bid_csv_right_index.to_pandas(), self.pdf_bid_csv_right_index, check_dtype=False)

    def test_merge_asof_from_csv_param_on(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date')
        odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right, on='date')
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid')
        odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right, on='bid')
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_csv_param_leftonrighton(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, left_on='date', right_on='date')
        odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right, left_on='date', right_on='date')
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, left_on='bid', right_on='bid')
        odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right, left_on='bid', right_on='bid')
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_csv_param_index(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right_index, left_index=True, right_index=True)
        odf = orca.merge_asof(self.odf_csv_left_index, self.odf_csv_right_index, left_index=True, right_index=True)
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right_index, left_index=True, right_index=True)
        odf = orca.merge_asof(self.odf_bid_csv_left_index, self.odf_bid_csv_right_index, left_index=True, right_index=True)
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_csv_param_by(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date', by='ticker')
        odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right, on='date', by='ticker')
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid', by='ticker')
        odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right, on='bid', by='ticker')
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_csv_param_leftbyrightby(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date', left_by='ticker', right_by='ticker')
        odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right, on='date', left_by='ticker', right_by='ticker')
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid', left_by='ticker', right_by='ticker')
        odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right, on='bid', left_by='ticker', right_by='ticker')
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_csv_param_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right, on='date', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right, on='bid', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_csv_param_leftonrighton_param_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, left_on='date', right_on='date',
                            suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right, left_on='date', right_on='date',
                              suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, left_on='bid', right_on='bid',
                            suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right, left_on='bid', right_on='bid',
                              suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_csv_param_leftonrighton_param_by(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, left_on='date', right_on='date', by='ticker')
        odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right, left_on='date', right_on='date', by='ticker')
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, left_on='bid', right_on='bid', by='ticker')
        odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right, left_on='bid', right_on='bid', by='ticker')
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_csv_param_leftonrighton_param_leftbyrightby(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date', left_by='ticker', right_by='ticker')
        odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right, on='date', left_by='ticker', right_by='ticker')
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid', left_by='ticker', right_by='ticker')
        odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right, on='bid', left_by='ticker', right_by='ticker')
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_csv_param_index_param_by(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right_index, left_index=True, right_index=True,
                            by='ticker')
        odf = orca.merge_asof(self.odf_csv_left_index, self.odf_csv_right_index, left_index=True, right_index=True,
                              by='ticker')
        # TODO: MERGE_ASOF BUG
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right_index, left_index=True, right_index=True, by='ticker')
        odf = orca.merge_asof(self.odf_bid_csv_left_index, self.odf_bid_csv_right_index, left_index=True, right_index=True, by='ticker')
        # TODO: MERGE_ASOF BUG
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_csv_param_index_param_leftbyrightby(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right_index, left_index=True, right_index=True,
                            left_by='ticker', right_by='ticker')
        odf = orca.merge_asof(self.odf_csv_left_index, self.odf_csv_right_index, left_index=True, right_index=True,
                              left_by='ticker', right_by='ticker')
        # TODO: MERGE_ASOF BUG
        # assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right_index, left_index=True, right_index=True, left_by='ticker',
                            right_by='ticker')
        odf = orca.merge_asof(self.odf_bid_csv_left_index, self.odf_bid_csv_right_index, left_index=True, right_index=True, left_by='ticker',
                              right_by='ticker')
        # TODO: MERGE_ASOF BUG
        # assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_csv_param_index_param_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right_index, left_index=True, right_index=True,
                            suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_csv_left_index, self.odf_csv_right_index, left_index=True, right_index=True,
                              suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right_index,
                            left_index=True, right_index=True, suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_bid_csv_left_index, self.odf_bid_csv_right_index,
                              left_index=True, right_index=True, suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_csv_param_on_param_by_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right,
                            on='date', by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right,
                              on='date', by='ticker', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right,
                            on='bid', by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right,
                              on='bid', by='ticker', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_csv_param_on_param_leftbyrightby_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date',
                            left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right, on='date',
                              left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid',
                            left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right, on='bid',
                              left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_csv_param_leftonrighton_param_by_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, left_on='date', right_on='date',
                            by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right, left_on='date', right_on='date',
                              by='ticker', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right,
                            left_on='bid', right_on='bid', by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right,
                              left_on='bid', right_on='bid', by='ticker', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_csv_param_leftonrighton_param_leftbyrightby_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, left_on='date', right_on='date',
                            left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right, left_on='date', right_on='date',
                              left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, left_on='bid', right_on='bid',
                            left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right, left_on='bid', right_on='bid',
                              left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_csv_param_index_param_by_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right_index,
                            left_index=True, right_index=True, by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_csv_left_index, self.odf_csv_right_index,
                              left_index=True, right_index=True, by='ticker', suffixes=('_left', '_right'))
        # TODO: MERGE_ASOF BUG
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right_index,
                            left_index=True, right_index=True, by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odf_bid_csv_left_index, self.odf_bid_csv_right_index,
                              left_index=True, right_index=True, by='ticker', suffixes=('_left', '_right'))
        # TODO: MERGE_ASOF BUG
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_csv_param_index_param_leftbyrightby_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right_index, left_index=True, right_index=True,
                            left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # TODO: MERGE_ASOF BUG
        # odf = orca.merge_asof(self.odf_csv_left_index, self.odf_csv_right_index, left_index=True, right_index=True,
        #                       left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right_index,
                            left_index=True, right_index=True,
                            left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # TODO: MERGE_ASOF BUG
        # odf = orca.merge_asof(self.odf_bid_csv_left_index, self.odf_bid_csv_right_index,
        #                       left_index=True, right_index=True,
        #                       left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_csv_param_on_param_index(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right, left_index=True, right_on='date')
        # TODO:ORCA error left_index, right_on not supported
        # odf = orca.merge_asof(self.odf_csv_left_index, self.odf_csv_right_index, left_index=True, right_on='date')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right_index, right_index=True, left_on='date')
        # TODO:ORCA error left_index, right_on not supported
        # odf = orca.merge_asof(self.odf_csv_left, self.odf_csv_right_index, right_index=True, left_on='date')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right, left_index=True, right_on='bid')
        # TODO:ORCA error left_index, right_on not supported
        # odf = orca.merge_asof(self.odf_bid_csv_left_index, self.odf_bid_csv_right, left_index=True, right_on='bid')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right_index, right_index=True, left_on='bid')
        # TODO:ORCA error left_index, right_on not supported
        # odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right_index, right_index=True, left_on='bid')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_csv_param_on_param_index_param_by(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right, left_index=True, right_on='date', by='ticker')
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odf_csv_left_index, self.odf_csv_right_index, left_index=True,
        # right_on='date', by='ticker')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right_index, right_index=True, left_on='date', by='ticker')
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odf_csv_left_index, self.odf_csv_right_index, right_index=True,
        # left_on='date', by='ticker')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right, left_index=True, right_on='bid', by='ticker')
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odf_bid_csv_left_index, self.odf_bid_csv_right, left_index=True,
        # right_on='bid', by='ticker')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right_index, right_index=True, left_on='bid', by='ticker')
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right_index, right_index=True,
        # left_on='bid', by='ticker')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_csv_param_on_param_index_param_by_param_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right, left_index=True, right_on='date',
                            by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odf_csv_left_index, self.odf_csv_right_index, left_index=True, right_on='date',
        #                       by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right_index, right_index=True, left_on='date',
                            by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odf_csv_left_index, self.odf_csv_right_index, right_index=True, left_on='date',
        #                       by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right, left_index=True, right_on='bid',
                            by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odf_bid_csv_left_index, self.odf_bid_csv_right, left_index=True, right_on='bid',
        #                       by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right_index, right_index=True, left_on='bid',
                            by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odf_bid_csv_left, self.odf_bid_csv_right_index, right_index=True, left_on='bid',
        #                       by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_dataframe_param_on(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['0', '1', '2'])
        orca_right = orca.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333]}, index=['0', '1', '2'])

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['0', '1', '2'])
        pd_right = pd.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333]}, index=['0', '1', '2'])

        pdf = pd.merge_asof(pd_left, pd_right, on='A')
        odf = orca.merge_asof(orca_left, orca_right, on='A')
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_dataframe_param_leftonrighton(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['0', '1', '2'])
        orca_right = orca.DataFrame({'C': [2, 2, 4], 'D': [1111, 2222, 3333]}, index=['0', '1', '2'])

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33]}, index=['0', '1', '2'])
        pd_right = pd.DataFrame({'C': [2, 2, 4], 'D': [1111, 2222, 3333]}, index=['0', '1', '2'])

        pdf = pd.merge_asof(pd_left, pd_right, left_on='A', right_on='C')
        odf = orca.merge_asof(orca_left, orca_right, left_on='A', right_on='C')
        # TODO: ORCA BUG left_on, right_on
        # assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_dataframe_param_by(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=['0', '1', '2'])
        orca_right = orca.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'S': ['b', 'a', 'c']}, index=['0', '1', '2'])

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=['0', '1', '2'])
        pd_right = pd.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'S': ['b', 'a', 'c']}, index=['0', '1', '2'])

        pdf = pd.merge_asof(pd_left, pd_right, on='A', by='S')
        odf = orca.merge_asof(orca_left, orca_right, on='A', by='S')
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_dataframe_param_leftbyrightby(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=['0', '1', '2'])
        orca_right = orca.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'R': ['b', 'a', 'c']}, index=['0', '1', '2'])

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=['0', '1', '2'])
        pd_right = pd.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'R': ['b', 'a', 'c']}, index=['0', '1', '2'])

        pdf = pd.merge_asof(pd_left, pd_right, on='A', left_by='S', right_by='R')
        odf = orca.merge_asof(orca_left, orca_right, on='A', left_by='S', right_by='R')
        # TODO: ORCA BUG left_by, right_by
        # assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_dataframe_param_suffixes(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=['0', '1', '2'])
        orca_right = orca.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'S': ['b', 'a', 'c']}, index=['0', '1', '2'])

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=['0', '1', '2'])
        pd_right = pd.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'S': ['b', 'a', 'c']}, index=['0', '1', '2'])

        pdf = pd.merge_asof(pd_left, pd_right, on='A', suffixes=('_left', '_right'))
        odf = orca.merge_asof(orca_left, orca_right, on='A', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

    def test_merge_asof_from_dataframe_param_index(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=[0, 1, 2])
        orca_right = orca.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'S': ['b', 'a', 'c']}, index=[0, 1, 2])

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=[0, 1, 2])
        pd_right = pd.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'S': ['b', 'a', 'c']}, index=[0, 1, 2])

        pdf = pd.merge_asof(pd_left, pd_right, left_index=True, right_index=True)
        odf = orca.merge_asof(orca_left, orca_right, left_index=True, right_index=True)
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_dataframe_param_by_param_suffix(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=[0, 1, 2])
        orca_right = orca.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'S': ['b', 'a', 'c']}, index=[0, 1, 2])

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=[0, 1, 2])
        pd_right = pd.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'S': ['b', 'a', 'c']}, index=[0, 1, 2])

        pdf = pd.merge_asof(pd_left, pd_right, on='A', suffixes=('_left', '_right'), by='S')
        odf = orca.merge_asof(orca_left, orca_right, on='A', suffixes=('_left', '_right'), by='S')
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_dataframe_param_index_param_by_param_suffix(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=[0, 1, 2])
        orca_right = orca.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'S': ['b', 'a', 'c']}, index=[0, 1, 2])

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=[0, 1, 2])
        pd_right = pd.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'S': ['b', 'a', 'c']}, index=[0, 1, 2])

        pdf = pd.merge_asof(pd_left, pd_right, left_index=True, right_index=True, suffixes=('_left', '_right'), by='S')
        odf = orca.merge_asof(orca_left, orca_right, left_index=True, right_index=True, suffixes=('_left', '_right'), by='S')
        # TODO:ORCA BUG left_index, right_index, suffixes, by
        # assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

    def test_merge_asof_from_dataframe_param_index_param_leftonrighton(self):
        orca_left = orca.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=[0, 1, 2])
        orca_right = orca.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'S': ['b', 'a', 'c']}, index=[0, 1, 2])

        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 22, 33], 'S': ['a', 'b', 'c']}, index=[0, 1, 2])
        pd_right = pd.DataFrame({'A': [2, 3, 4], 'D': [1111, 2222, 3333], 'S': ['b', 'a', 'c']}, index=[0, 1, 2])

        pdf = pd.merge_asof(pd_left, pd_right, left_index=True, right_on='A', suffixes=('_left', '_right'), by='S')
        # TODO:ORCA BUG left_index, right_on, suffixes, by

        # odf = orca.merge_asof(orca_left, orca_right, left_index=True, right_on='A', suffixes=('_left', '_right'), by='S')
        # assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
