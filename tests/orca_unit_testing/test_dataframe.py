import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None


class DataFrameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        # Csv.odf_csv = orca.read_csv(data, dtype={"DLSTCD": np.float32, "DLPRC": np.float32})
        Csv.odf_csv = orca.read_csv(data, dtype={"PERMNO": np.int32, "date": 'DATE', "TRDSTAT": 'SYMBOL',
                                                           "DLSTCD": np.float32, "DLPRC": np.float32, "VOL": np.float32,
                                                           "SHROUT": np.float32, "CFACPR":np.float32, "CFACSHR": np.float32})
        # pdf from import
        Csv.pdf_csv = pd.read_csv(data, parse_dates=[1], dtype={"PERMNO": np.int32, "SHRCD": np.int32, "HEXCD": np.int32,
                                                                "DLSTCD": np.float32, "DLPRC": np.float32,
                                                                "VOL": np.float32, "SHROUT": np.float32})
        Csv.odf_csv = Csv.odf_csv.drop(columns=['DLRET'])
        Csv.pdf_csv.drop(columns=['DLRET'], inplace=True)

    @property
    def pdf_csv(self):
        return Csv.pdf_csv

    @property
    def odf_csv(self):
        return Csv.odf_csv

    @property
    def pdf(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

    @property
    def odf(self):
        return orca.DataFrame(self.pdf)

    def test_dataframe_constructor_from_dict_param_data(self):
        d = {'col1': [1, 2], 'col2': [3, 4]}
        pdf = pd.DataFrame(data=d)
        odf = orca.DataFrame(data=d).to_pandas()
        assert_frame_equal(odf, pdf)

    def test_dataframe_constructor_from_dict_param_data_hasNan(self):
        d = {'col1': [np.NaN, 2], 'col2': [np.NaN, 4]}
        pdf = pd.DataFrame(data=d)
        odf = orca.DataFrame(data=d).to_pandas()
        assert_frame_equal(odf, pdf)

    def test_dataframe_constructor_from_dict_param_dtype(self):
        d = {'col1': [1, 2], 'col2': [3, 4]}
        pdf = pd.DataFrame(data=d, dtype=np.int8)
        odf = orca.DataFrame(data=d, dtype=np.int8).to_pandas()
        assert_frame_equal(odf, pdf)

    def test_dataframe_constructor_from_dict_param_dtype_hasNan(self):
        d = {'col1': [1, np.NaN], 'col2': [np.NaN, 4]}
        pdf = pd.DataFrame(data=d, dtype=np.int32)
        # TODO: dataframe_hasNan: fail to initialize a dataframe with np.nan values
        # odf = orca.DataFrame(data=d, dtype=np.int32).to_pandas()
        # assert_frame_equal(odf, pdf)

    def test_dataframe_constructor_from_ndarray_param_columns(self):
        nd = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        pdf = pd.DataFrame(nd, columns=['a', 'b', 'c'])
        odf = orca.DataFrame(nd, columns=['a', 'b', 'c']).to_pandas()
        assert_frame_equal(odf, pdf)

    def test_dataframe_constructor_from_ndarray_param_columns_hasNan(self):
        nd = np.array([[1, 2, 3], [np.NaN, np.NaN, np.NaN], [7, np.NaN, 9]])
        pdf = pd.DataFrame(nd, columns=['a', 'b', 'c'])
        odf = orca.DataFrame(nd, columns=['a', 'b', 'c']).to_pandas()
        assert_frame_equal(odf, pdf)

    def test_dataframe_attributes_index(self):
        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        assert_index_equal(odf.index.to_pandas(), pdf.index)

    def test_dataframe_attributes_columns(self):
        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        assert_index_equal(odf.columns, pdf.columns)

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=pd.date_range("20190101", periods=4, freq="d"))
        # pd.to_datetime(["20190101","20190304"])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=orca.date_range("20190101", periods=4, freq="d"))
        assert_index_equal(odf.columns, pdf.columns)

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=[1, 2, 3, 4])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=[1, 2, 3, 4])
        assert_index_equal(odf.columns, pdf.columns)

    def test_dataframe_attributes_columns_names(self):
        column = pd.Index(['A', 'B', 'C'], name='X')
        pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=column)
        odf = orca.DataFrame(pdf)
        self.assertEqual(repr(odf), repr(pdf))
        self.assertEqual(repr(odf.columns.names), repr(pdf.columns.names))

    def test_dataframe_attributes_dtypes(self):
        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        assert_series_equal(odf.dtypes, pdf.dtypes)

    def test_dataframe_attributes_select_dtypes(self):
        pdf = pd.DataFrame({'a': [1, 2] * 3, 'b': [True, False] * 3, 'c': [1.0, 2.0] * 3})
        odf = orca.DataFrame({'a': [1, 2] * 3, 'b': [True, False] * 3, 'c': [1.0, 2.0] * 3})
        assert_frame_equal(odf.select_dtypes(include='bool').to_pandas(), pdf.select_dtypes(include='bool'))
        assert_frame_equal(odf.select_dtypes(include=['float64']).to_pandas(), pdf.select_dtypes(include=['float64']))
        assert_frame_equal(odf.select_dtypes(exclude=['int']).to_pandas(), pdf.select_dtypes(exclude=['int']))
        assert_frame_equal(odf.select_dtypes(include=['float64', 'int']).to_pandas(),
                           pdf.select_dtypes(include=['float64', 'int']))
        assert_frame_equal(odf.select_dtypes(exclude=['float64', 'int']).to_pandas(),
                           pdf.select_dtypes(exclude=['float64', 'int']))

    def test_dataframe_attributes_values(self):
        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        assert_numpy_array_equal(odf.values, pdf.values)

    def test_dataframe_attributes_axes(self):
        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        odf = orca.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        assert_index_equal(odf.axes[0].to_pandas(), pdf.axes[0])
        assert_index_equal(odf.axes[1], pdf.axes[1])

        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, pd.date_range("20190101", periods=2, freq="d"))
        odf = orca.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, orca.date_range("20190101", periods=2, freq="d"))
        assert_index_equal(odf.axes[0].to_pandas(), pdf.axes[0])
        assert_index_equal(odf.axes[1], pdf.axes[1])

        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, index=['a', 'b'])
        odf = orca.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, index=['a', 'b'])
        assert_index_equal(odf.axes[0].to_pandas(), pdf.axes[0])
        assert_index_equal(odf.axes[1], pdf.axes[1])

    def test_dataframe_attributes_ndim(self):
        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        odf = orca.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        self.assertEqual(odf.ndim, pdf.ndim)

    def test_dataframe_attributes_size(self):
        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        odf = orca.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        self.assertEqual(odf.size, pdf.size)
        self.assertEqual(self.odf.size, self.pdf.size)
        self.assertEqual(self.odf_csv.size, self.pdf_csv.size)

    def test_dataframe_attributes_shape(self):
        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        odf = orca.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        self.assertEqual(odf.shape, pdf.shape)
        self.assertEqual(self.odf.shape, self.pdf.shape)
        self.assertEqual(self.odf_csv.shape, self.pdf_csv.shape)

    def test_dataframe_attributes_empty(self):
        pdf = pd.DataFrame({})
        odf = orca.DataFrame({})
        self.assertEqual(odf.empty, pdf.empty)
        self.assertEqual(self.odf.empty, self.pdf.empty)
        self.assertEqual(self.odf_csv.empty, self.pdf_csv.empty)

    def test_dataframe(self):
        odf = self.odf
        pdf = self.pdf

        self.assertEqual(repr((odf['a'] + 1).to_pandas()), repr(pdf['a'] + 1))

        self.assertEqual(repr(odf.columns), repr(pd.Index(['a', 'b'])))

        self.assertEqual(repr((odf[odf['b'] > 2]).to_pandas()), repr(pdf[pdf['b'] > 2]))
        self.assertEqual(repr(odf[['a', 'b']]), repr(pdf[['a', 'b']]))
        self.assertEqual(repr(odf.a), repr(pdf.a))

        assert repr(odf)

        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        })
        ddf = orca.DataFrame(df)
        self.assertEqual(repr(ddf[['a', 'b']]), repr(df[['a', 'b']]), )

        # check orca.DataFrame(os.Series)
        pser = pd.Series([1, 2, 3], name='x')
        kser = orca.Series([1, 2, 3], name='x')
        self.assertEqual(repr(pd.DataFrame(pser)), repr(orca.DataFrame(kser)))

    def test_dataframe_topic_multiindex_columns(self):
        pdf = pd.DataFrame({
            ('x', 'a', '1'): [1, 2, 3],
            ('x', 'b', '2'): [4, 5, 6],
            ('y.z', 'c.d', '3'): [7, 8, 9],
            ('x', 'b', '4'): [10, 11, 12],
        }, index=[0, 1, 3])
        odf = orca.DataFrame(pdf)

        # TODO：NOT IMPLEMENTED
        # self.assertEqual(repr(odf), repr(pdf))
        # self.assertEqual(repr(odf['x']), repr(pdf['x']))
        # self.assertEqual(repr(odf['y.z']), repr(pdf['y.z']))
        # self.assertEqual(repr(odf['x']['b']), repr(pdf['x']['b']))
        # self.assertEqual(repr(odf['x']['b']['2']), repr(pdf['x']['b']['2']))
        #
        # self.assertEqual(repr(odf.x), repr(pdf.x))
        # self.assertEqual(repr(odf.x.b), repr(pdf.x.b))
        # self.assertEqual(repr(odf.x.b['2']), repr(pdf.x.b['2']))
        #
        # self.assertEqual(repr(odf[('x',)]), repr(pdf[('x',)]))
        # self.assertEqual(repr(odf[('x', 'a')]), repr(pdf[('x', 'a')]))
        # self.assertEqual(repr(odf[('x', 'a', '1')]), repr(pdf[('x', 'a', '1')]))

    def test_dataframe_topic_multiindex_names_level(self):
        columns = pd.MultiIndex.from_tuples([('X', 'A', 'Z'), ('X', 'B', 'Z'),
                                             ('Y', 'C', 'Z'), ('Y', 'D', 'Z')],
                                            names=['lvl_1', 'lvl_2', 'lv_3'])
        pdf = pd.DataFrame([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                            [17, 18, 19, 20]], columns=columns)
        odf = orca.DataFrame(pdf)

        self.assertEqual(repr(odf.columns.names), repr(pdf.columns.names))
        self.assertEqual(repr(odf.columns.names), repr(pdf.columns.names))

        odf1 = orca.DataFrame(pdf)
        self.assertEqual(repr(odf1.columns.names), repr(pdf.columns.names))

        # with self.assertRaisesRegex(ValueError, 'Column_index_names should '
        #                                         'be list-like or None for a MultiIndex'):
        #     orca.DataFrame(odf1._internal.copy(column_index_names='level'))

        # TODO：NOT IMPLEMENTED
        # self.assertEqual(repr(odf['X']), repr(pdf['X']))
        # self.assertEqual(repr(odf['X'].columns.names), repr(pdf['X'].columns.names))
        # self.assertEqual(repr(odf['X'].columns.names), repr(pdf['X'].columns.names))
        # self.assertEqual(repr(odf['X']['A']), repr(pdf['X']['A']))
        # self.assertEqual(repr(odf['X']['A'].columns.names), repr(pdf['X']['A'].columns.names))
        # self.assertEqual(repr(odf['X']['A'].DataFrame().columns.names), repr(pdf['X']['A'].columns.names))
        # self.assertEqual(repr(odf[('X', 'A')]), repr(pdf[('X', 'A')]))
        # self.assertEqual(repr(odf[('X', 'A')].columns.names), repr(pdf[('X', 'A')].columns.names))
        # self.assertEqual(repr(odf[('X', 'A')].DataFrame().columns.namesv), repr(pdf[('X', 'A')].columns.names))
        # self.assertEqual(repr(odf[('X', 'A', 'Z')]), repr(pdf[('X', 'A', 'Z')]))

    def test_dataframe_topic_multiindex_reset_index_with_multiindex_columns(self):
        index = pd.MultiIndex.from_tuples([('bird', 'falcon'),
                                           ('bird', 'parrot'),
                                           ('mammal', 'lion'),
                                           ('mammal', 'monkey')],
                                          names=['class', 'name'])
        columns = pd.MultiIndex.from_tuples([('speed', 'max'),
                                             ('species', 'type')])
        pdf = pd.DataFrame([(389.0, 'fly'),
                            (24.0, 'fly'),
                            (80.5, 'run'),
                            (np.nan, 'jump')],
                           index=index,
                           columns=columns)
        odf = orca.DataFrame(pdf)

        self.assertEqual(repr(odf), repr(pdf))
        self.assertEqual(repr(odf.reset_index()), repr(pdf.reset_index()))
        self.assertEqual(repr(odf.reset_index(level='class')), repr(pdf.reset_index(level='class')))
        self.assertEqual(repr(odf.reset_index(level='class', col_level=1)),
                         repr(pdf.reset_index(level='class', col_level=1)))
        self.assertEqual(repr(odf.reset_index(level='class', col_level=1, col_fill='species')),
                         repr(pdf.reset_index(level='class', col_level=1, col_fill='species')))
        self.assertEqual(repr(odf.reset_index(level='class', col_level=1, col_fill='genus')),
                         repr(pdf.reset_index(level='class', col_level=1, col_fill='genus')))

        # TODO：NOT IMPLEMENTED
        # with self.assertRaisesRegex(IndexError, 'Index has only 2 levels, not 3'):
        #     odf.reset_index(col_level=2)

    def test_dataframe_topic_multiindex_column_access(self):
        columns = pd.MultiIndex.from_tuples([('a', 'w', 'q', 'b'),
                                             ('c', 'w', 'd', 'c'),
                                             ('e', 's', 'f', 's'),
                                             ('m', 'g', 'e', 'r'),
                                             ('s', 's', 'd', 'h'),
                                             ('i', 's', 's', 's')])

        pdf = pd.DataFrame([(1, 'a', 'x', 10, 100, 1000),
                            (2, 'b', 'y', 20, 200, 2000),
                            (3, 'c', 'z', 30, 300, 3000)],
                           columns=columns)
        odf = orca.DataFrame(pdf)

        self.assertEqual(repr(odf), repr(pdf))
        # TODO：NOT IMPLEMENTED
        # self.assertEqual(repr(odf['a']), repr(pdf['a']))
        # self.assertEqual(repr(odf['a']['b']), repr(pdf['a']['b']))
        # self.assertEqual(repr(odf['c']), repr(pdf['c']))
        # self.assertEqual(repr(odf['c']['d']), repr(pdf['c']['d']))
        # self.assertEqual(repr(odf['e']), repr(pdf['e']))
        # self.assertEqual(repr(odf['e']['']['f']), repr(pdf['e']['']['f']))
        # self.assertEqual(repr(odf['e']['g']), repr(pdf['e']['g']))
        # self.assertEqual(repr(odf['']), repr(pdf['']))
        # self.assertEqual(repr(odf['']['h']), repr(pdf['']['h']))
        # self.assertEqual(repr(odf['i']), repr(pdf['i']))
        #
        # self.assertEqual(repr(odf[['a', 'e']]), repr(pdf[['a', 'e']]))
        # self.assertEqual(repr(odf[['e', 'a']]), repr(pdf[['e', 'a']]))
        #
        # self.assertEqual(repr(odf[('a',)]), repr(pdf[('a',)]))
        # self.assertEqual(repr(odf[('e', 'g')]), repr(pdf[('e', 'g')]))
        # self.assertEqual(repr(odf[('i',)]), repr(pdf[('i',)]))


if __name__ == '__main__':
    unittest.main()
