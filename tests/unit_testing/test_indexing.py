from setup.settings import *
import unittest
import orca
import os.path as path
from pandas.util.testing import *


class IndexingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    @property
    def ps(self):
        return pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')

    @property
    def os(self):
        # return orca.Series(self.ps)
        return orca.Series([1, 2, 3, 4, 5, 6, 7], name='x')

    def test_indexing_dataframe_head_tail(self):
        pdf = pd.DataFrame(
            {'animal': ['alligator', 'bee', 'falcon', 'lion', 'monkey', 'parrot', 'shark', 'whale', 'zebra'],
             'id': [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        odf = orca.DataFrame(
            {'animal': ['alligator', 'bee', 'falcon', 'lion', 'monkey', 'parrot', 'shark', 'whale', 'zebra'],
             'id': [1, 2, 3, 4, 5, 6, 7, 8, 9]})

        # head
        assert_frame_equal(pdf.head(), odf.head().to_pandas())
        assert_frame_equal(pdf.head(5), odf.head(5).to_pandas())
        assert_frame_equal(pdf.head(3), odf.head(3).to_pandas())
        # TODO: orca.DataFrame.head(0)
        # assert_frame_equal(pdf.head(0), odf.head(0).to_pandas())
        assert_frame_equal(pdf.head(-3), odf.head(-3).to_pandas())
        assert_frame_equal(pdf[pdf['id'] > 5].head(-3), odf[odf['id'] > 5].head(-3).to_pandas())
        # TODO: orca.ArithExpression.head(-3)
        # assert_frame_equal((pdf['id']+1).head(-3), (odf['id']+1).head(-3).to_pandas())

        # tail
        assert_frame_equal(pdf.tail(), odf.tail().to_pandas())
        assert_frame_equal(pdf.tail(5), odf.tail(5).to_pandas())
        assert_frame_equal(pdf.tail(3), odf.tail(3).to_pandas())
        # TODO: orca.DataFrame.tail(0)
        # assert_frame_equal(pdf.tail(0), odf.tail(0).to_pandas())
        assert_frame_equal(pdf.tail(-3), odf.tail(-3).to_pandas())
        assert_frame_equal(pdf[pdf['id'] > 5].tail(-3), odf[odf['id'] > 5].tail(-3).to_pandas())
        # TODO: orca.ArithExpression.tail(-3)
        # assert_frame_equal((pdf['id']+1).tail(-3), (odf['id']+1).tail(-3).to_pandas())

    def test_indexing_dataframe_loc_get(self):
        pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                           columns=['max_speed', 'shield'])
        odf = orca.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                             columns=['max_speed', 'shield'])
        assert_series_equal(odf.loc['cobra'].to_pandas(), pdf.loc['cobra'])
        self.assertEqual(odf.loc['cobra', 'shield'], pdf.loc['cobra', 'shield'])
        assert_frame_equal(odf.loc[['cobra', 'viper']].to_pandas(), pdf.loc[['cobra', 'viper']])
        assert_frame_equal(odf.loc[[False, False, True]].to_pandas(), pdf.loc[[False, False, True]])
        assert_frame_equal(odf.loc[odf['shield'] > 5].to_pandas(), pdf.loc[pdf['shield'] > 5])
        assert_frame_equal(odf.loc[odf['shield'] > 6, ['max_speed']].to_pandas(),
                           pdf.loc[pdf['shield'] > 6, ['max_speed']])
        # assert_frame_equal(odf.loc['cobra':, 'max_speed':'shield'].to_pandas(), pdf.loc[pdf['shield'] > 6, ['max_speed']])
        assert_frame_equal(odf.loc['cobra':'viper'].to_pandas(), pdf.loc['cobra':'viper'])
        # TODO:odf.loc[:, 'max_speed'] 结果只有一列的DataFrame应该返回一个series
        # assert_series_equal(odf.loc['cobra':'viper', 'max_speed'].to_pandas(), pdf.loc['cobra':'viper', 'max_speed'])
        # assert_series_equal(odf.loc[:, 'max_speed'].to_pandas(), pdf.loc[:, 'max_speed'])

        pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8]], index=[7, 8, 9], columns=['max_speed', 'shield'])
        odf = orca.DataFrame([[1, 2], [4, 5], [7, 8]], index=[7, 8, 9], columns=['max_speed', 'shield'])
        assert_frame_equal(odf.loc[7:9].to_pandas(), pdf.loc[7:9])

        v = np.full((6, 4), 10)
        pdf = pd.DataFrame(v, index=list('abcdef'), columns=list('ABCD'))
        odf = orca.DataFrame(v, index=list('abcdef'), columns=list('ABCD'))
        pd.DataFrame()
        assert_frame_equal(odf.loc[['a', 'b', 'd'], :].to_pandas(), pdf.loc[['a', 'b', 'd'], :])
        # assert_frame_equal(odf.loc['d':, 'A':'C'].to_pandas(), pdf.loc['d':, 'A':'C'])
        # assert_frame_equal((odf.loc['a'] > 0).to_pandas(), pdf.loc['a'] > 0)

        # TODO:loc：当index中含有nan值，pandas的表现似乎不太正常
        # pdd = pd.DataFrame(
        #     {'id': [1, 2, 2, 3, 3], 'sym': ['s', 'a', 's', 'a', 's'], 'values': [np.nan, 2, 2, np.nan, 2]})
        # pdd.set_index('values', inplace=True)
        # odd = orca.DataFrame(pdd)
        # assert_frame_equal(pdd.loc[np.nan:], odd.loc[np.nan:].to_pandas())

        # TODO:loc：当index为时间类型的index
        pdd = pd.DataFrame(
            {'id': [1, 2, 2, 3, 3], 'sym': ['s', 'a', 's', 'a', 's'], 'values': [np.nan, 2, 2, np.nan, 2]},
            index=pd.date_range('20190101', '20190105', 5))
        odd = orca.DataFrame(pdd)
        # assert_frame_equal(pdd.loc['2019-01-02':], odd.loc['2019-01-02':].to_pandas())
        # TODO:loc：当index为时间类型的index,orca还不支持datetime函数
        # assert_frame_equal(pdd.loc[pd.datetime(2019, 1, 2):], odd.loc[orca.datetime(2019, 1, 2):].to_pandas())

    def test_indexing_dataframe_loc_set(self):
        # scalar
        pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                           columns=['max_speed', 'shield'])
        odf = orca.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                             columns=['max_speed', 'shield'])
        odf.loc['cobra'] = 10
        pdf.loc['cobra'] = 10
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc['cobra', 'shield'] = 10
        pdf.loc['cobra', 'shield'] = 10
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[['cobra', 'viper']] = 10
        pdf.loc[['cobra', 'viper']] = 10
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[[True, True, False]] = 10
        pdf.loc[[True, True, False]] = 10
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[odf['shield'] > 5] = 10
        pdf.loc[pdf['shield'] > 5] = 10
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[odf['shield'] > 6, ['max_speed']] = 10
        pdf.loc[pdf['shield'] > 6, ['max_speed']] = 10
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc['cobra':'viper'] = 10
        pdf.loc['cobra':'viper'] = 10
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc['cobra':'viper', 'max_speed'] = 10
        pdf.loc['cobra':'viper', 'max_speed'] = 10
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[:, 'max_speed'] = 10
        pdf.loc[:, 'max_speed'] = 10
        assert_frame_equal(odf.to_pandas(), pdf)

        # list
        pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                           columns=['max_speed', 'shield'])
        odf = orca.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                             columns=['max_speed', 'shield'])
        odf.loc['cobra'] = [10, 10]
        pdf.loc['cobra'] = [10, 10]
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[odf['shield'] > 5] = [9, 10]
        pdf.loc[pdf['shield'] > 5] = [9, 10]
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc['cobra':'viper', 'max_speed'] = [11, 15]
        pdf.loc['cobra':'viper', 'max_speed'] = [11, 15]
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[:, 'max_speed'] = [10, 11, 12]
        pdf.loc[:, 'max_speed'] = [10, 11, 12]
        assert_frame_equal(odf.to_pandas(), pdf)

        # np.array
        pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                           columns=['max_speed', 'shield'])
        odf = orca.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                             columns=['max_speed', 'shield'])
        odf.loc['cobra'] = np.array([10, 12])
        pdf.loc['cobra'] = np.array([10, 12])
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc['cobra', 'shield'] = np.array(10)
        pdf.loc['cobra', 'shield'] = np.array(10)
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[odf['shield'] > 6, ['max_speed']] = np.array(12)
        pdf.loc[pdf['shield'] > 6, ['max_speed']] = np.array(12)
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc['cobra':'viper', 'max_speed'] = np.array([20, 30])
        pdf.loc['cobra':'viper', 'max_speed'] = np.array([20, 30])
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[:, 'max_speed'] = np.array([20, 30, 4])
        pdf.loc[:, 'max_speed'] = np.array([20, 30, 4])
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        # Series
        pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                           columns=['max_speed', 'shield'])
        odf = orca.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                             columns=['max_speed', 'shield'])
        odf.loc['cobra'] = orca.Series([10, 11], index=['max_speed', 'shield'])
        pdf.loc['cobra'] = pd.Series([10, 11], index=['max_speed', 'shield'])
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[odf['shield'] > 6, ['max_speed']] = orca.Series([12], index=['sidewinder'])
        pdf.loc[pdf['shield'] > 6, ['max_speed']] = pd.Series([12], index=['sidewinder'])
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)
        odf.loc['cobra':'viper', 'max_speed'] = orca.Series([12, 13], index=['cobra', 'viper'])
        pdf.loc['cobra':'viper', 'max_speed'] = pd.Series([12, 13], index=['cobra', 'viper'])
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)
        odf.loc[:, 'max_speed'] = orca.Series([12, 13, 14], index=['cobra', 'viper', 'sidewinder'])
        pdf.loc[:, 'max_speed'] = pd.Series([12, 13, 14], index=['cobra', 'viper', 'sidewinder'])
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        # Dict
        pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                           columns=['max_speed', 'shield'])
        odf = orca.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                             columns=['max_speed', 'shield'])
        odf.loc['cobra'] = {'max_speed': 11, 'shield': 12}
        pdf.loc['cobra'] = {'max_speed': 11, 'shield': 12}
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[odf['shield'] > 6, ['max_speed']] = {'sidewinder': 11}
        pdf.loc[pdf['shield'] > 6, ['max_speed']] = {'sidewinder': 11}
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)
        odf.loc['cobra':'viper', 'max_speed'] = {'cobra': 11, 'viper': 40}
        pdf.loc['cobra':'viper', 'max_speed'] = {'cobra': 11, 'viper': 40}
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        # DataFrame
        pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                           columns=['max_speed', 'shield'])
        odf = orca.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                             columns=['max_speed', 'shield'])
        odf.loc[odf['shield'] > 5] = orca.DataFrame({'max_speed': 15, 'shield': 16}, index=['sidewinder'])
        pdf.loc[pdf['shield'] > 5] = pd.DataFrame({'max_speed': 15, 'shield': 16}, index=['sidewinder'])
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[['cobra', 'viper']] = orca.DataFrame({'max_speed': [15, 16], 'shield': [17, 18]},
                                                     index=['cobra', 'viper'])
        pdf.loc[['cobra', 'viper']] = pd.DataFrame({'max_speed': [15, 16], 'shield': [17, 18]},
                                                   index=['cobra', 'viper'])
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[[False, False, True]] = orca.DataFrame({'max_speed': 15, 'shield': 16}, index=['sidewinder'])
        pdf.loc[[False, False, True]] = pd.DataFrame({'max_speed': 15, 'shield': 16}, index=['sidewinder'])
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[odf['shield'] > 5] = orca.DataFrame({'max_speed': [15, 16], 'shield': [17, 18]},
                                                    index=['viper', 'sidewinder'])
        pdf.loc[pdf['shield'] > 5] = pd.DataFrame({'max_speed': [15, 16], 'shield': [17, 18]},
                                                  index=['viper', 'sidewinder'])
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc[odf['shield'] > 6, ['max_speed']] = orca.DataFrame({'max_speed': 15, 'shield': 16},
                                                                   index=['sidewinder'])
        pdf.loc[pdf['shield'] > 6, ['max_speed']] = pd.DataFrame({'max_speed': 15, 'shield': 16}, index=['sidewinder'])
        assert_frame_equal(odf.to_pandas(), pdf)
        odf.loc['cobra':'viper'] = orca.DataFrame({'max_speed': [15, 16], 'shield': [17, 18]}, index=['cobra', 'viper'])
        pdf.loc['cobra':'viper'] = pd.DataFrame({'max_speed': [15, 16], 'shield': [17, 18]}, index=['cobra', 'viper'])
        assert_frame_equal(odf.to_pandas(), pdf, check_dtype=False)

        pdf = pd.DataFrame([[1, 2, 1], [4, 5, 5], [7, 8, 7], [1, 5, 8], [7, 5, 1]], index=[7, 8, 9, 8, 9],
                           columns=['max_speed', 'shield', 'size'])
        odf = orca.DataFrame([[1, 2, 1], [4, 5, 5], [7, 8, 7], [1, 5, 8], [7, 5, 1]], index=[7, 8, 9, 8, 9],
                             columns=['max_speed', 'shield', 'size'])
        # TODO: orca不支持：当index有重复的列，通过一个dataframe去set
        # odf.loc[7:] = orca.DataFrame([[1, 1, 1], [5, 5, 5], [7, 7, 7], [8, 8, 8], [6, 6, 6]], index=[7, 8, 9, 8, 9],
        #                              columns=['max_speed', 'shield', 'size'])
        # pdf.loc[7:] = pd.DataFrame([[1, 1, 1], [5, 5, 5], [7, 7, 7], [8, 8, 8], [6, 6, 6]], index=[7, 8, 9, 8, 9],
        #                            columns=['max_speed', 'shield', 'size'])
        # assert_frame_equal(odf.to_pandas(), pdf)

        odf.loc[8] = orca.DataFrame([[1, 2, 1], [4, 5, 5]], index=[8, 8],
                                    columns=['max_speed', 'shield', 'size'])
        pdf.loc[8] = pd.DataFrame([[1, 2, 1], [4, 5, 5]], index=[8, 8],
                                  columns=['max_speed', 'shield', 'size'])

    def test_indexing_dataframe_iloc_get(self):
        pdf = pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4}, {'a': 100, 'b': 200, 'c': 300, 'd': 400},
                            {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}])
        odf = orca.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4}, {'a': 100, 'b': 200, 'c': 300, 'd': 400},
                              {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}])
        # integer
        assert_series_equal(pdf.iloc[0], odf.iloc[0].to_pandas())
        # list
        assert_frame_equal(pdf.iloc[[0, 1]], odf.iloc[[0, 1]].to_pandas())
        # slice
        assert_frame_equal(pdf.iloc[:3], odf.iloc[:3].to_pandas())
        # both axes integer
        # TODO: odf.iloc[0, 1]应该返回类型为numpy.int64的整型常量，而非一个series
        # self.assertEqual(pdf.iloc[0, 1], odf.iloc[0, 1])
        # both axes list
        assert_frame_equal(pdf.iloc[[0, 2], [1, 3]], odf.iloc[[0, 2], [1, 3]].to_pandas())
        # both axes slice
        assert_frame_equal(pdf.iloc[1:3, 0:3], odf.iloc[1:3, 0:3].to_pandas())
        assert_frame_equal(pdf.iloc[[False, True, False]], odf.iloc[[False, True, False]].to_pandas())
        assert_frame_equal(pdf.iloc[:, [True, False, True, False]],
                           odf.iloc[:, [True, False, True, False]].to_pandas())
        assert_frame_equal(pdf.iloc[[False, True, False], :],
                           odf.iloc[[False, True, False], :].to_pandas())
        # both axes boolean array
        assert_frame_equal(pdf.iloc[[True, True, False], [True, False, True, False]],
                           odf.iloc[[True, True, False], [True, False, True, False]].to_pandas())

    def test_indexing_dataframe_iloc_set(self):
        pdf = pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4}, {'a': 100, 'b': 200, 'c': 300, 'd': 400},
                            {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}])
        odf = orca.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4}, {'a': 100, 'b': 200, 'c': 300, 'd': 400},
                              {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}])
        # integer
        # iloc[*]
        pdf.iloc[0] = 11
        odf.iloc[0] = 11
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0]] = 11
        odf.iloc[[0]] = 11
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[1, 2]] = 110
        odf.iloc[[1, 2]] = 110
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:] = 12
        odf.iloc[1:] = 12
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:2] = 12
        odf.iloc[:2] = 12
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[0:1] = 120
        odf.iloc[0:1] = 120
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:] = 13
        odf.iloc[:] = 13
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[True, False, True]] = 13
        odf.iloc[[True, False, True]] = 13
        assert_frame_equal(odf.to_pandas(), pdf)

        # iloc[0,*]
        pdf.iloc[0, 1] = 130
        odf.iloc[0, 1] = 130
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[0, [1]] = 11
        odf.iloc[0, [1]] = 11
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[0, [1, 3]] = 14
        odf.iloc[0, [1, 3]] = 14
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[0, 1:] = 140
        odf.iloc[0, 1:] = 140
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[0, :2] = 12
        odf.iloc[0, :2] = 12
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[0, 1:2] = 15
        odf.iloc[0, 1:2] = 15
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[0, 0] = 150
        odf.iloc[0, 0] = 150
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[0, [True, False, True, True]] = 13
        odf.iloc[0, [True, False, True, True]] = 13
        assert_frame_equal(odf.to_pandas(), pdf)

        # iloc[[0],*]
        pdf.iloc[0, 1] = 130
        odf.iloc[0, 1] = 130
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0], [1]] = 11
        odf.iloc[[0], [1]] = 11
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0], [1, 3]] = 14
        odf.iloc[[0], [1, 3]] = 14
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0], 1:] = 140
        odf.iloc[[0], 1:] = 140
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0], :2] = 12
        odf.iloc[[0], :2] = 12
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0], 1:2] = 15
        odf.iloc[[0], 1:2] = 15
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0], 0] = 150
        odf.iloc[[0], 0] = 150
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0], [True, False, True, True]] = 13
        odf.iloc[[0], [True, False, True, True]] = 13
        assert_frame_equal(odf.to_pandas(), pdf)

        # iloc[[0,2],*]
        pdf.iloc[[0, 2], 1] = 130
        odf.iloc[[0, 2], 1] = 130
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0, 2], [1]] = 11
        odf.iloc[[0, 2], [1]] = 11
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0, 2], [1, 3]] = 14
        odf.iloc[[0, 2], [1, 3]] = 14
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0, 2], 1:] = 140
        odf.iloc[[0, 2], 1:] = 140
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0, 2], :2] = 12
        odf.iloc[[0, 2], :2] = 12
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0, 2], 1:2] = 15
        odf.iloc[[0, 2], 1:2] = 15
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0, 2], 0] = 150
        odf.iloc[[0, 2], 0] = 150
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0, 2], [True, False, True, True]] = 13
        odf.iloc[[0, 2], [True, False, True, True]] = 13
        assert_frame_equal(odf.to_pandas(), pdf)

        # iloc[1:,*]
        pdf.iloc[1:, 1] = 130
        odf.iloc[1:, 1] = 130
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:, [1]] = 11
        odf.iloc[1:, [1]] = 11
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:, [1, 3]] = 14
        odf.iloc[1:, [1, 3]] = 14
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:, 1:] = 140
        odf.iloc[1:, 1:] = 140
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:, :2] = 12
        odf.iloc[1:, :2] = 12
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:, 1:2] = 15
        odf.iloc[1:, 1:2] = 15
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:, 0] = 150
        odf.iloc[1:, 0] = 150
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:, [True, False, True, True]] = 13
        odf.iloc[1:, [True, False, True, True]] = 13
        assert_frame_equal(odf.to_pandas(), pdf)

        # iloc[:2,*]
        pdf.iloc[:2, 1] = 130
        odf.iloc[:2, 1] = 130
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:2, [1]] = 11
        odf.iloc[:2, [1]] = 11
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:2, [1, 3]] = 14
        odf.iloc[:2, [1, 3]] = 14
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:2, 1:2] = 140
        odf.iloc[:2, 1:2] = 140
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:2, :2] = 12
        odf.iloc[:2, :2] = 12
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:2, 1:2] = 15
        odf.iloc[:2, 1:2] = 15
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:2, 0] = 150
        odf.iloc[:2, 0] = 150
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:2, [True, False, True, True]] = 13
        odf.iloc[:2, [True, False, True, True]] = 13
        assert_frame_equal(odf.to_pandas(), pdf)

        # iloc[1:2,*]
        pdf.iloc[1:2, 1] = 130
        odf.iloc[1:2, 1] = 130
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:2, [1]] = 11
        odf.iloc[1:2, [1]] = 11
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:2, [1, 3]] = 14
        odf.iloc[1:2, [1, 3]] = 14
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:2, 1:2] = 140
        odf.iloc[1:2, 1:2] = 140
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:2, :2] = 12
        odf.iloc[1:2, :2] = 12
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:2, 1:2] = 15
        odf.iloc[1:2, 1:2] = 15
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:2, 0] = 150
        odf.iloc[1:2, 0] = 150
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:2, [True, False, True, True]] = 13
        odf.iloc[1:2, [True, False, True, True]] = 13
        assert_frame_equal(odf.to_pandas(), pdf)

        # iloc[:,*]
        pdf.iloc[:, 1] = 130
        odf.iloc[:, 1] = 130
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:, [1]] = 11
        odf.iloc[:, [1]] = 11
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:, [1, 3]] = 14
        odf.iloc[:, [1, 3]] = 14
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:, 1:] = 140
        odf.iloc[:, 1:] = 140
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:, 1:2] = 15
        odf.iloc[:, 1:2] = 15
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:, :] = 150
        odf.iloc[:, :] = 150
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:, [True, False, True, True]] = 13
        odf.iloc[:, [True, False, True, True]] = 13
        assert_frame_equal(odf.to_pandas(), pdf)

        # list array series dataframe
        pdf = pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4}, {'a': 100, 'b': 200, 'c': 300, 'd': 400},
                            {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}])
        odf = orca.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4}, {'a': 100, 'b': 200, 'c': 300, 'd': 400},
                              {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}])
        # iloc[*]
        pdf.iloc[0] = [1, 2, 1, 2]
        odf.iloc[0] = [1, 2, 1, 2]
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[0] = np.array([1, 20, 10, 2])
        odf.iloc[0] = np.array([1, 20, 10, 2])
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[0] = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        odf.iloc[0] = orca.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[0] = {'a': 0, 'b': 2, 'c': 2, 'd': 3}
        odf.iloc[0] = {'a': 0, 'b': 2, 'c': 2, 'd': 3}
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[0]] = pd.DataFrame({'a': 0, 'b': 2, 'c': 2, 'd': 3}, index=[0])
        odf.iloc[[0]] = orca.DataFrame({'a': 0, 'b': 2, 'c': 2, 'd': 3}, index=[0])
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[[1, 2]] = pd.DataFrame({'a': [0, 3], 'b': [0, 3], 'c': [0, 3], 'd': [0, 3]}, index=[1, 2])
        odf.iloc[[1, 2]] = orca.DataFrame({'a': [0, 3], 'b': [0, 3], 'c': [0, 3], 'd': [0, 3]}, index=[1, 2])
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[1:] = pd.DataFrame({'a': [0, 3], 'b': [0, 3], 'c': [0, 3], 'd': [0, 3]}, index=[1, 2])
        odf.iloc[1:] = orca.DataFrame({'a': [0, 3], 'b': [0, 3], 'c': [0, 3], 'd': [0, 3]}, index=[1, 2])
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[:2] = pd.DataFrame({'a': [0, 3], 'b': [0, 3], 'c': [0, 3], 'd': [0, 3]}, index=[0, 1])
        odf.iloc[:2] = orca.DataFrame({'a': [0, 3], 'b': [0, 3], 'c': [0, 3], 'd': [0, 3]}, index=[0, 1])
        assert_frame_equal(odf.to_pandas(), pdf)
        pdf.iloc[0:1] = pd.DataFrame({'a': 0, 'b': 2, 'c': 2, 'd': 3}, index=[0])
        odf.iloc[0:1] = orca.DataFrame({'a': 0, 'b': 2, 'c': 2, 'd': 3}, index=[0])
        assert_frame_equal(odf.to_pandas(), pdf)

    def test_indexing_dataframe_case_1(self):
        np_mat = np.full((8, 4), 10)
        pdf = pd.DataFrame(np_mat, index=pd.date_range('1/1/2000', periods=8), columns=['A', 'B', 'C', 'D'])
        ps = pdf['A']
        odf = orca.DataFrame(np_mat, index=orca.date_range('1/1/2000', periods=8), columns=['A', 'B', 'C', 'D'])
        os = odf['A']
        assert_series_equal(ps, os.to_pandas())

    def test_indexing_dataframe_case_2(self):
        pdf = pd.DataFrame(np.full((8, 4), 10), index=pd.date_range('1/1/2000', periods=8),
                           columns=['A', 'B', 'C', 'D'])
        odf = orca.DataFrame(np.full((8, 4), 10), index=orca.date_range('1/1/2000', periods=8),
                             columns=['A', 'B', 'C', 'D'])

        # ok if A already exists
        pdf['A'] = list(range(len(pdf.index)))
        odf['A'] = list(range(len(odf.index)))
        ps = pdf['A']
        os = odf['A']
        assert_series_equal(ps, os.to_pandas(), check_dtype=False)

    def test_indexing_dataframe_case_3(self):
        pdf = pd.DataFrame(np.full((8, 4), 10), index=pd.date_range('1/1/2000', periods=8),
                           columns=['A', 'B', 'C', 'D'])
        odf = orca.DataFrame(np.full((8, 4), 10), index=orca.date_range('1/1/2000', periods=8),
                             columns=['A', 'B', 'C', 'D'])

        # use this form to create a new column
        pdf['A'] = list(range(len(pdf.index)))
        odf['A'] = list(range(len(odf.index)))
        ps = pdf['A']
        os = odf['A']
        assert_series_equal(ps, os.to_pandas(), check_dtype=False)

    def test_indexing_series_loc_get(self):
        ps = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
        os = orca.Series(list('abcde'), index=[0, 3, 2, 5, 4])
        assert_series_equal(ps, os.to_pandas())
        assert_series_equal(os.loc[3:5].to_pandas(), ps.loc[3:5])

    def test_indexing_series_loc_set(self):
        ps = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
        os = orca.Series(list('abcde'), index=[0, 3, 2, 5, 4])
        assert_series_equal(os.to_pandas(), ps)
        ps.loc[3:5] = "5"
        os.loc[3:5] = "5"
        assert_series_equal(os.to_pandas(), ps)
        ps = pd.Series(np.repeat(1, 5), index=[0, 3, 2, 5, 4])
        os = orca.Series(np.repeat(1, 5), index=[0, 3, 2, 5, 4])
        ps.loc[3:5] = 2
        os.loc[3:5] = 2
        assert_series_equal(os.to_pandas(), ps)
        ps = pd.Series(list('abcde'), index=["a", "a", "a", "b", "b"])
        os = orca.Series(list('abcde'), index=["a", "a", "a", "b", "b"])
        assert_series_equal(os.to_pandas(), ps)
        ps.loc["a"] = "5"
        os.loc["a"] = "5"
        assert_series_equal(os.to_pandas(), ps)
        # TODO：orca不支持：当index中的取值是重复的
        # ps = pd.Series(np.repeat(1, 5), index=["a", "a", "b", "b", "c"])
        # os = orca.Series(np.repeat(1, 5), index=["a", "a", "b", "b", "c"])
        # ps.loc["a":"b"] = 2
        # os.loc["a":"b"] = 2
        # assert_series_equal(os.to_pandas(), ps)

    def test_indexing_series_iloc_get(self):
        v = [1, 2, 3, 4, 5]
        ps = pd.Series(v, index=list(range(0, 10, 2)))
        os = orca.Series(v, index=list(range(0, 10, 2)))
        assert_series_equal(os.iloc[:3].to_pandas(), ps.iloc[:3])

    def test_indexing_series_iloc_set(self):
        v1 = [1, 2, 3, 4, 5]
        v2 = ["a", "b", "c", "d", "e"]

        ps = pd.Series(v1, index=list(range(0, 10, 2)))
        os = orca.Series(v1, index=list(range(0, 10, 2)))
        ps.iloc[:3] = 4
        os.iloc[:3] = 4
        assert_series_equal(os.to_pandas(), ps)

        ps = pd.Series(v2, index=list(range(0, 10, 2)))
        os = orca.Series(v2, index=list(range(0, 10, 2)))
        ps.iloc[:3] = "s"
        os.iloc[:3] = "s"
        assert_series_equal(os.to_pandas(), ps)

        ps = pd.Series(v1, index=["a", "a", "b", "b", "c"])
        os = orca.Series(v1, index=["a", "a", "b", "b", "c"])
        ps.iloc[:3] = 4
        os.iloc[:3] = 4
        assert_series_equal(os.to_pandas(), ps)

        ps = pd.Series(v2, index=["a", "a", "b", "b", "c"])
        os = orca.Series(v2, index=["a", "a", "b", "b", "c"])
        ps.iloc[:3] = "s"
        os.iloc[:3] = "s"
        assert_series_equal(os.to_pandas(), ps)


    def test_indexing_series_sort_index(self):
        ps = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
        os = orca.Series(list('abcde'), index=[0, 3, 2, 5, 4])
        assert_series_equal(ps.sort_index(), os.sort_index().to_pandas())
        assert_series_equal(ps.sort_index().loc[0:3], os.sort_index().loc[0:3].to_pandas())
        # TODO: orca不支持 当loc访问的slice的下界在index中不存在
        # assert_series_equal(ps.sort_index().loc[1:3], os.sort_index().loc[1:3].to_pandas())

    # def test_indexing(self, df):
    #     df1 = df.set_index('month')
    #     yield df1
    #
    #     yield df.set_index('month', drop=False)
    #     yield df.set_index('month', append=True)
    #     yield df.set_index(['year', 'month'])
    #     yield df.set_index(['year', 'month'], drop=False)
    #     yield df.set_index(['year', 'month'], append=True)
    #
    #     yield df1.set_index('year', drop=False, append=True)
    #
    #     df2 = df1.copy()
    #     df2.set_index('year', append=True, inplace=True)
    #     yield df2
    #
    #     self.assertRaisesRegex(KeyError, 'unknown', lambda: df.set_index('unknown'))
    #     self.assertRaisesRegex(KeyError, 'unknown', lambda: df.set_index(['month', 'unknown']))
    #
    #     for d in [df, df1, df2]:
    #         yield d.reset_index()
    #         yield d.reset_index(drop=True)
    #
    #     yield df1.reset_index(level=0)
    #     yield df2.reset_index(level=1)
    #     yield df2.reset_index(level=[1, 0])
    #     yield df1.reset_index(level='month')
    #     yield df2.reset_index(level='year')
    #     yield df2.reset_index(level=['month', 'year'])
    #     yield df2.reset_index(level='month', drop=True)
    #     yield df2.reset_index(level=['month', 'year'], drop=True)
    #
    #     if LooseVersion("0.20.0") <= LooseVersion(pd.__version__):
    #         self.assertRaisesRegex(IndexError, 'Too many levels: Index has only 1 level, not 3',
    #                                lambda: df1.reset_index(level=2))
    #         self.assertRaisesRegex(IndexError, 'Too many levels: Index has only 1 level, not 4',
    #                                lambda: df1.reset_index(level=[3, 2]))
    #         self.assertRaisesRegex(KeyError, 'Level unknown must be same as name \\(month\\)',
    #                                lambda: df1.reset_index(level='unknown'))
    #     self.assertRaisesRegex(KeyError, 'Level unknown not found',
    #                            lambda: df2.reset_index(level='unknown'))
    #
    #     df3 = df2.copy()
    #     df3.reset_index(inplace=True)
    #     yield df3
    #
    #     yield df1.sale.reset_index()
    #     yield df1.sale.reset_index(level=0)
    #     yield df2.sale.reset_index(level=[1, 0])
    #     yield df1.sale.reset_index(drop=True)
    #     yield df1.sale.reset_index(name='s')
    #     yield df1.sale.reset_index(name='s', drop=True)
    #
    #     s = df1.sale
    #     self.assertRaisesRegex(TypeError,
    #                            'Cannot reset_index inplace on a Series to create a DataFrame',
    #                            lambda: s.reset_index(inplace=True))
    #     s.reset_index(drop=True, inplace=True)
    #     yield s
    #     yield df1

    # def test_Series_with_explicit_index(self):
    #     pdf = pd.DataFrame(np.full((8, 4), 10), index=pd.date_range('1/1/2000', periods=8),
    #                        columns=['A', 'B', 'C', 'D'])
    #     odf = orca.DataFrame(np.full((8, 4), 10), index=orca.date_range('1/1/2000', periods=8),
    #                          columns=['A', 'B', 'C', 'D'])
    #     df1 = orca.DataFrame(pdf.set_index('A'))
    #     pdf1 = pdf.set_index('A')
    #     pdf1.equals(df1.to_pandas())
