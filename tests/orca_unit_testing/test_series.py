import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None


class SeriesTest(unittest.TestCase):
    def setUp(self):
        self.PRECISION = 5

    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        Csv.pdf_csv = pd.read_csv(data)
        Csv.odf_csv = orca.read_csv(data)

    @property
    def ps(self):
        return pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')

    @property
    def os(self):
        return orca.Series([1, 2, 3, 4, 5, 6, 7], name='x')

    @property
    def psa(self):
        return pd.Series([10, 1, 19, np.nan], index=['a', 'b', 'c', 'd'])

    @property
    def psb(self):
        return pd.Series([-1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])

    def test_series_constructor(self):
        ps = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')
        os = orca.Series([1, 2, 3, 4, 5, 6, 7], name='x').to_pandas()
        assert_series_equal(ps, os)

    def test_series_constructor_hasNan(self):
        ps = pd.Series([7, np.NaN, 1, np.NaN])
        os = orca.Series([7, np.NaN, 1, np.NaN]).to_pandas()
        assert_series_equal(ps, os)

    def test_series_constructor_hasFloat(self):
        ps = pd.Series([7.4, 3.1415826535, np.NaN, -3.4], name='x')
        os = orca.Series([7.4, 3.1415826535, np.NaN, -3.4], name='x').to_pandas()
        assert_series_equal(ps, os)

    def test_series_constructor_with_index(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5]).to_pandas()
        assert_series_equal(ps, os)

    # def test_series_constructor_from_dict(self):
    #     d = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    #     ps = pd.Series(d)
    #     os = orca.Series(d)
    #     assert_series_equal(ps, os)

    def test_series_constructor_from_scalar(self):
        ps = pd.Series(1)
        os = orca.Series(1).to_pandas()
        assert_series_equal(ps, os)

    def test_series_attributes_index(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        assert_index_equal(ps.index, os.index.to_pandas())

        ps = pd.Series([7, 2, 1, 4], index=['a', 'b', 'c', 'd'])
        os = orca.Series([7, 2, 1, 4], index=['a', 'b', 'c', 'd'])
        assert_index_equal(ps.index, os.index.to_pandas())

        ps = pd.Series([7, 2, 1, 4], pd.date_range("20190101", periods=4, freq="d"))
        os = orca.Series([7, 2, 1, 4], orca.date_range("20190101", periods=4, freq="d"))
        assert_index_equal(ps.index, os.index.to_pandas())

    def test_series_attributes_array(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        # TODO: pandas.Seires 的array属性返回一个pandas Array，而orca.Series的array属性返回一个list
        self.assertEqual(list(ps.array), os.array)

    def test_series_attributes_values(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        assert_numpy_array_equal(ps.values, os.values)

        ps = pd.Series(['a', 'b', 'c', 'd'])
        os = orca.Series(['a', 'b', 'c', 'd'])
        assert_numpy_array_equal(ps.values, os.values)

        ps = pd.Series(pd.date_range("20190101", periods=10, freq="d"))
        # os = orca.Series(pd.date_range("20190101", periods=10, freq="d"))
        os = orca.Series(ps)
        assert_numpy_array_equal(ps.values, os.values)

    def test_series_attributes_dtype(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        self.assertEqual(ps.dtype, os.dtype)

        ps = pd.Series(['a', 'b', 'c', 'd'])
        os = orca.Series(['a', 'b', 'c', 'd'])
        self.assertEqual(ps.dtype, os.dtype)

        ps = pd.Series(pd.date_range("20190101", periods=10, freq="d"))
        # os = orca.Series(pd.date_range("20190101", periods=10, freq="d"))
        os = orca.Series(ps)
        self.assertEqual(ps.dtype, os.dtype)

    def test_series_attributes_shape(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        self.assertEqual(ps.shape, os.shape)

    def test_series_attributes_nbytes(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5])

    def test_series_attributes_ndim(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        self.assertEqual(ps.ndim, os.ndim)

    def test_series_attributes_size(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        self.assertEqual(ps.size, os.size)

    def test_series_attributes_T(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        assert_series_equal(ps.T, os.T.to_pandas())

    def test_series_attributes_hasnans(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        self.assertEqual(ps.hasnans, os.hasnans)

        ps = pd.Series([7, 2, 1, np.nan], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, np.nan], index=[3, 1, 5, 5])
        self.assertEqual(ps.hasnans, os.hasnans)

    def test_series_attributes_dtypes(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        self.assertEqual(ps.dtypes, os.dtypes)

    def test_series_attributes_name(self):
        ps = pd.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        os = orca.Series([7, 2, 1, 4], index=[3, 1, 5, 5])
        ps.name = "S1"
        os.name = "S1"
        self.assertEqual(ps.name, os.name)

    def test_series_binary_operator_function_series_hasnan(self):
        ps = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        os = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        # TODO: series_hasNan: fail to initialize a series with np.nan values
        # psb = pd.Series([10, 1, 19, np.nan], index=['a', 'b', 'c', 'd'])
        # osb = orca.Series([10, 1, 19, np.nan], index=['a', 'b', 'c', 'd'])
        # c1 = ps + psb
        # c2 = (os + osb).to_pandas()
        # assert_series_equal(c1, c2)
        # c1 = ps - psb
        # c2 = (os - osb).to_pandas()
        # assert_series_equal(c1, c2)
        # c1 = ps * psb
        # c2 = (os * osb).to_pandas()
        # assert_series_equal(c1, c2)
        # c1 = ps / psb
        # c2 = (os / osb).to_pandas()
        # assert_series_equal(c1, c2)
        # c1 = ps ** psb
        # c2 = (os ** osb).to_pandas()
        # assert_series_equal(c1, c2)
        # c1 = ps // psb
        # c2 = (os // osb).to_pandas()
        # assert_series_equal(c1, c2)
        # c1 = ps % psb
        # c2 = (os % osb).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_add_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps + 1
        c2 = (os + 1).to_pandas()
        assert_series_equal(c1, c2)

        # TODO: defalt axis=0: orca.Series.add(1)
        # c1 = ps.add(1)
        # c2 = os.add(1).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_add_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps + [1, 2, 12, 10]
        c2 = (os + [1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2)

        c1 = ps.add([1, 2, 12, 10])
        # TODO: orca.Series.add([1, 2, 12, 10])
        # c2 = os.add([1, 2, 12, 10]).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_add_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        c1 = ps + psb
        c2 = (os + osb).to_pandas()
        assert_series_equal(c1, c2)

        # series with series expression
        c1 = ps + (1 / psb)
        c2 = (os + (1 / osb)).to_pandas()
        assert_series_equal(c1, c2)

        # series expression with series expression
        c1 = (ps * [1, 3, 5, 4]) + (1 / psb)
        c2 = ((os * [1, 3, 5, 4]) + (1 / osb)).to_pandas()
        assert_series_equal(c1, c2)

        c1 = pdf["float"] + pdf["int"]
        c2 = (odf["float"] + odf["int"]).to_pandas()
        assert_series_equal(c1, c2)

        # series with series
        # default axis=0
        c1 = ps.add(psb)
        c2 = os.add(osb).to_pandas()
        assert_series_equal(c1, c2)
        # specify axis=0
        c1 = ps.add(psb, axis=0)
        c2 = os.add(osb, axis=0).to_pandas()
        assert_series_equal(c1, c2)
        # specify axis=1, ValueError expected
        # TODO: ValueError expected: orca.Series.add(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.add(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.add(osb, axis=1)

    def test_series_binary_operator_function_sub_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps - 1
        c2 = (os - 1).to_pandas()
        assert_series_equal(c1, c2)

        TODO: orca.Series.sub(1)
        # c1 = ps.sub(1)
        # c2 = os.sub(1).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_sub_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps - [1, 2, 12, 10]
        c2 = (os - [1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2)

        c1 = ps.sub([1, 2, 12, 10])
        TODO: orca.Series.sub([1, 2, 12, 10])
        # c2 = os.sub([1, 2, 12, 10]).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_sub_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        c1 = ps - psb
        c2 = (os - osb).to_pandas()
        assert_series_equal(c1, c2)

        # series with series expression
        c1 = ps - (1 / psb)
        c2 = (os - (1 / osb)).to_pandas()
        assert_series_equal(c1, c2)

        # series expression with series expression
        c1 = (ps * [1, 3, 5, 4]) - (1 / psb)
        c2 = ((os * [1, 3, 5, 4]) - (1 / osb)).to_pandas()
        assert_series_equal(c1, c2)

        TODO: odf["float"] - odf["int"]
        # c1 = pdf["float"] - pdf["int"]
        # c2 = (odf["float"] - odf["int"]).to_pandas()
        # assert_series_equal(c1, c2)

        # series with series
        # default axis=0
        TODO: orca.Series.sub(orca.Series())
        # c1 = ps.sub(psb)
        # c2 = os.sub(osb).to_pandas()
        # assert_series_equal(c1, c2)
        # specify axis=0
        c1 = ps.sub(psb, axis=0)
        c2 = os.sub(osb, axis=0).to_pandas()
        assert_series_equal(c1, c2)
        # specify axis=1, ValueError expected
        TODO: orca.Series.sub(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.sub(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.sub(osb, axis=1)

    def test_series_binary_operator_function_mul_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps * 1
        c2 = (os * 1).to_pandas()
        assert_series_equal(c1, c2)

        TODO: orca.Series.mul(1)
        # c1 = ps.mul(1)
        # c2 = os.mul(1).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_mul_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps * [1, 2, 12, 10]
        c2 = (os * [1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2)

        TODO: orca.Series.mul([1, 2, 12, 10])
        # c1 = ps.mul([1, 2, 12, 10])
        # c2 = os.mul([1, 2, 12, 10]).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_mul_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        c1 = ps * psb
        c2 = (os * osb).to_pandas()
        assert_series_equal(c1, c2)

        # series with series expression
        c1 = ps * (1 / psb)
        c2 = (os * (1 / osb)).to_pandas()
        assert_series_equal(c1, c2)

        # series expression with series expression
        c1 = (ps * [1, 3, 5, 4]) * (1 / psb)
        c2 = ((os * [1, 3, 5, 4]) * (1 / osb)).to_pandas()
        assert_series_equal(c1, c2)

        TODO: odf["float"] * odf["int"]
        # c1 = pdf["float"] * pdf["int"]
        # c2 = (odf["float"] * odf["int"]).to_pandas()
        # assert_series_equal(c1, c2)

        # series with series
        # default axis=0
        TODO: orca.Series.mul(orca.Series())
        # c1 = ps.mul(psb)
        # c2 = os.mul(osb).to_pandas()
        # assert_series_equal(c1, c2)
        # specify axis=0
        c1 = ps.mul(psb, axis=0)
        c2 = os.mul(osb, axis=0).to_pandas()
        assert_series_equal(c1, c2)
        # specify axis=1, ValueError expected
        TODO: orca.Series.mul(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.mul(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.mul(osb, axis=1)

    def test_series_binary_operator_function_div_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps / 1
        c2 = (os / 1).to_pandas()
        assert_series_equal(c1, c2)

        # TODO: orca.Series.div(1)
        # c1 = ps.div(1)
        # c2 = os.div(1).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_div_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps / [1, 2, 12, 10]
        c2 = (os / [1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2)

        TODO: orca.Series.div([1, 2, 12, 10])
        # c1 = ps.div([1, 2, 12, 10])
        # c2 = os.div([1, 2, 12, 10]).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_div_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        c1 = ps / psb
        c2 = (os / osb).to_pandas()
        assert_series_equal(c1, c2)

        # series with series expression
        c1 = ps / (1 + psb)
        c2 = (os / (1 + osb)).to_pandas()
        assert_series_equal(c1, c2)

        # series expression with series expression
        c1 = (ps - [1, 3, 5, 4]) / (1 + psb)
        c2 = ((os - [1, 3, 5, 4]) / (1 + osb)).to_pandas()
        assert_series_equal(c1, c2)

        TODO: odf["float"] / odf["int"]
        # c1 = pdf["float"] / pdf["int"]
        # c2 = (odf["float"] / odf["int"]).to_pandas()
        # assert_series_equal(c1, c2)

        # default axis=0
        TODO: orca.Series.div(orca.Series())
        # c1 = ps.div(psb)
        # c2 = os.div(osb).to_pandas()
        # assert_series_equal(c1, c2)
        # specify axis=0
        c1 = ps.div(psb, axis=0)
        c2 = os.div(osb, axis=0).to_pandas()
        assert_series_equal(c1, c2)
        # specify axis=1, ValueError expected
        TODO: orca.Series.div(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.div(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.div(osb, axis=1)

    def test_series_binary_operator_function_truediv_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        # TODO: orca.Series.truediv(1)
        # c1 = ps.truediv(1)
        # c2 = os.truediv(1).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_truediv_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        TODO: orca.Series.truediv([1, 2, 12, 10])
        # c1 = ps.truediv([1, 2, 12, 10])
        # c2 = os.truediv([1, 2, 12, 10]).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_truediv_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        # default axis=0
        TODO: orca.Series.truediv(orca.Series())
        # c1 = ps.truediv(psb)
        # c2 = os.truediv(osb).to_pandas()
        # assert_series_equal(c1, c2)

        # specify axis=0
        c1 = ps.truediv(psb, axis=0)
        c2 = os.truediv(osb, axis=0).to_pandas()
        assert_series_equal(c1, c2)

        # specify axis=1, ValueError expected
        TODO: orca.Series.truediv(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.truediv(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.truediv(osb, axis=1)

    def test_series_binary_operator_function_floordiv_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps // 1
        c2 = (os // 1).to_pandas()
        assert_series_equal(c1, c2)

        TODO: orca.Series.floordiv(1)
        # c1 = ps.floordiv(1)
        # c2 = os.floordiv(1).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_floordiv_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps // [1, 2, 12, 10]
        c2 = (os // [1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2)

        TODO: orca.Series.floordiv([1, 2, 12, 10])
        # c1 = ps.floordiv([1, 2, 12, 10])
        # c2 = os.floordiv([1, 2, 12, 10]).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_floordiv_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        c1 = ps // psb
        c2 = (os // osb).to_pandas()
        assert_series_equal(c1, c2)

        # series with series expression
        c1 = ps // (1 + psb)
        c2 = (os // (1 + osb)).to_pandas()
        assert_series_equal(c1, c2)

        # series expression with series expression
        c1 = (ps - [1, 3, 5, 4]) // (1 + psb)
        c2 = ((os - [1, 3, 5, 4]) // (1 + osb)).to_pandas()
        assert_series_equal(c1, c2)

        TODO: odf["float"] // odf["int"]
        # c1 = pdf["float"] // pdf["int"]
        # c2 = (odf["float"] // odf["int"]).to_pandas()
        # assert_series_equal(c1, c2)

        # series with series
        # default axis=0
        TODO: orca.Series.floordiv(orca.Series())
        # c1 = ps.floordiv(psb)
        # c2 = os.floordiv(osb).to_pandas()
        # assert_series_equal(c1, c2)
        # specify axis=0
        c1 = ps.floordiv(psb, axis=0)
        c2 = os.floordiv(osb, axis=0).to_pandas()
        assert_series_equal(c1, c2)
        # specify axis=1, ValueError expected
        TODO: orca.Series.floordiv(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.floordiv(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.floordiv(osb, axis=1)

    def test_series_binary_operator_function_mod_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps % 1
        c2 = (os % 1).to_pandas()
        assert_series_equal(c1, c2)

        TODO: orca.Series.mod(1)
        # c1 = ps.mod(1)
        # c2 = os.mod(1).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_mod_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps % [1, 2, 12, 10]
        c2 = (os % [1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2)

        TODO: orca.Series.mod([1, 2, 12, 10])
        # c1 = ps.mod([1, 2, 12, 10])
        # c2 = os.mod([1, 2, 12, 10]).to_pandas()
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_mod_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        c1 = ps % psb
        c2 = (os % osb).to_pandas()
        assert_series_equal(c1, c2)

        # series with series expression
        c1 = ps % (1 + psb)
        c2 = (os % (1 + osb)).to_pandas()
        assert_series_equal(c1, c2)

        # series expression with series expression
        c1 = (ps - [1, 3, 5, 4]) % (1 + psb)
        c2 = ((os - [1, 3, 5, 4]) % (1 + osb)).to_pandas()
        assert_series_equal(c1, c2)

        TODO: odf["float"] % odf["int"]
        # c1 = pdf["float"] % pdf["int"]
        # c2 = (odf["float"] % odf["int"]).to_pandas()
        # assert_series_equal(c1, c2)

        # series with series
        # default axis=0
        TODO: orca.Series.mod(orca.Series())
        # c1 = ps.mod(psb)
        # c2 = os.mod(osb).to_pandas()
        # assert_series_equal(c1, c2)
        # specify axis=0
        c1 = ps.mod(psb, axis=0)
        c2 = os.mod(osb, axis=0).to_pandas()
        assert_series_equal(c1, c2)
        # specify axis=1, ValueError expected
        TODO: orca.Series.mod(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.mod(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.mod(osb, axis=1)

    def test_series_binary_operator_function_pow_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps ** 1
        c2 = (os ** 1).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

        TODO: orca.Series.pow(1)
        # c1 = ps.pow(1)
        # c2 = os.pow(1).to_pandas()
        # assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_pow_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        c1 = ps ** [1, 2, 12, 10]
        c2 = (os ** [1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

        TODO: orca.Series.pow([1, 2, 12, 10])
        # c1 = ps.pow([1, 2, 12, 10])
        # c2 = os.pow([1, 2, 12, 10]).to_pandas()
        # assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_pow_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        c1 = ps ** psb
        c2 = (os ** osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

        # series with series expression
        c1 = ps ** (1 + psb)
        c2 = (os ** (1 + osb)).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

        # series expression with series expression
        c1 = (ps - [1, 3, 5, 4]) ** (1 + psb)
        c2 = ((os - [1, 3, 5, 4]) ** (1 + osb)).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

        TODO: odf["float"] ** odf["int"]
        # c1 = pdf["float"] ** pdf["int"]
        # c2 = (odf["float"] ** odf["int"]).to_pandas()
        # assert_series_equal(c1, c2, check_dtype=False)

        # series with series
        # default axis=0
        TODO: orca.Series.pow(orca.Series())
        # c1 = ps.pow(psb)
        # c2 = os.pow(osb).to_pandas()
        # assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=0
        c1 = ps.pow(psb, axis=0)
        c2 = os.pow(osb, axis=0).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=1, ValueError expected
        TODO: orca.Series.pow(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.pow(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.pow(osb, axis=1)

    def test_series_binary_operator_function_radd_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        c1 = ps.radd(1)
        c2 = os.radd(1).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_radd_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        c1 = ps.radd([1, 2, 12, 10])
        c2 = os.radd([1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_radd_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        # default axis=0
        c1 = ps.radd(psb)
        c2 = os.radd(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=0
        # TODO: raise error: orca.Series.radd(orca.Series(), axis=0)
        # c1 = ps.radd(psb, axis=0)
        # c2 = os.radd(osb, axis=0).to_pandas()
        # assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=1, ValueError expected
        # TODO: ValueError expected orca.Series.radd(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.radd(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.radd(osb, axis=1)

    def test_series_binary_operator_function_rsub_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rsub(1)
        c2 = os.rsub(1).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rsub_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rsub([1, 2, 12, 10])
        c2 = os.rsub([1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rsub_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        # default axis=0
        c1 = ps.rsub(psb)
        c2 = os.rsub(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=0
        c1 = ps.rsub(psb, axis=0)
        c2 = os.rsub(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=1, ValueError expected
        TODO: orca.Series.rsub(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.rsub(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.rsub(osb, axis=1)

    def test_series_binary_operator_function_rmul_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rmul(1)
        c2 = os.rmul(1).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rmul_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rmul([1, 2, 12, 10])
        c2 = os.rmul([1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rmul_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        # default axis=0
        c1 = ps.rmul(psb)
        c2 = os.rmul(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=0
        c1 = ps.rmul(psb, axis=0)
        c2 = os.rmul(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=1, ValueError expected
        TODO: orca.Series.rmul(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.rmul(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.rmul(osb, axis=1)

    def test_series_binary_operator_function_rdiv_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rdiv(1)
        c2 = os.rdiv(1).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rdiv_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rdiv([1, 2, 12, 10])
        c2 = os.rdiv([1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rdiv_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        # default axis=0
        c1 = ps.rdiv(psb)
        c2 = os.rdiv(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=0
        c1 = ps.rdiv(psb, axis=0)
        c2 = os.rdiv(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=1, ValueError expected
        TODO: orca.Series.rdiv(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.rdiv(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.rdiv(osb, axis=1)

    def test_series_binary_operator_function_rtruediv_scalar(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rtruediv(1)
        c2 = os.rtruediv(1).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rtruediv_list(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rtruediv([1, 2, 12, 10])
        c2 = os.rtruediv([1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rtruediv_series(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        # default axis=0
        c1 = ps.rtruediv(psb)
        c2 = os.rtruediv(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=0
        c1 = ps.rtruediv(psb, axis=0)
        c2 = os.rtruediv(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=1, ValueError expected
        TODO: orca.Series.rtruediv(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.rtruediv(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.rtruediv(osb, axis=1)

    def test_series_binary_operator_function_rfloordiv_scalar(self):
        # TODO： 负数的差异
        # ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        # os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        ps = pd.Series([10, 1, 19, 5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, 5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rfloordiv(1)
        c2 = os.rfloordiv(1).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rfloordiv_list(self):
        # TODO： 负数的差异
        # ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        # os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        ps = pd.Series([10, 1, 19, 5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, 5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rfloordiv([1, 2, 12, 10])
        c2 = os.rfloordiv([1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rfloordiv_series(self):
        # TODO： 负数的差异
        # ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        # os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        ps = pd.Series([10, 1, 19, 5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, 5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        # default axis=0
        c1 = ps.rfloordiv(psb)
        c2 = os.rfloordiv(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=0
        c1 = ps.rfloordiv(psb, axis=0)
        c2 = os.rfloordiv(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=1, ValueError expected
        TODO: orca.Series.rfloordiv(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.rfloordiv(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.rfloordiv(osb, axis=1)

    def test_series_binary_operator_function_rmod_scalar(self):
        # TODO： 负数的差异
        # ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        # os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        ps = pd.Series([10, 1, 19, 5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, 5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rmod(1)
        c2 = os.rmod(1).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rmod_list(self):
        # TODO： 负数的差异
        # ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        # os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        ps = pd.Series([10, 1, 19, 5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, 5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rmod([1, 2, 12, 10])
        c2 = os.rmod([1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rmod_series(self):
        # TODO： 负数的差异
        # ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        # os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        ps = pd.Series([10, 1, 19, 5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, 5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        # default axis=0
        c1 = ps.rmod(psb)
        c2 = os.rmod(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=0
        c1 = ps.rmod(psb, axis=0)
        c2 = os.rmod(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=1, ValueError expected
        TODO: orca.Series.rmod(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.rmod(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.rmod(osb, axis=1)

    def test_series_binary_operator_function_rpow_scalar(self):
        ps = pd.Series([10, 1, 9.0, 5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 9.0, 5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rpow(4)
        c2 = os.rpow(4).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rpow_list(self):
        ps = pd.Series([10, 1, 9.0, 5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 9.0, 5], index=['a', 'b', 'c', 'd'])
        c1 = ps.rpow([1, 2, 12, 10])
        c2 = os.rpow([1, 2, 12, 10]).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)

    def test_series_binary_operator_function_rpow_series(self):
        ps = pd.Series([10, 1, 9.0, 5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 9.0, 5], index=['a', 'b', 'c', 'd'])

        psb = pd.Series([1, 2, 12, 10.0, 11], index=['a', 'a', 'b', 'c', 'd'])
        osb = orca.Series([1, 2, 12, 10.0, 11], index=['a', 'a', 'b', 'c', 'd'])

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])

        # series with series
        # default axis=0
        c1 = ps.rpow(psb)
        c2 = os.rpow(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False, check_less_precise=1)
        # specify axis=0
        c1 = ps.rpow(psb, axis=0)
        c2 = os.rpow(osb).to_pandas()
        assert_series_equal(c1, c2, check_dtype=False)
        # specify axis=1, ValueError expected
        TODO: orca.Series.rpow(orca.Series(), axis=1)
        # msg = "No axis named 1 for object type <class 'pandas.core.series.Series'>"
        # with self.assertRaisesRegex(ValueError, msg):
        #     ps.rpow(psb, axis=1)
        # with self.assertRaisesRegex(ValueError, msg):
        #     os.rpow(osb, axis=1)

    def test_series_binary_operator_function_combine(self):
        ps1 = pd.Series({'falcon': 330.0, 'eagle': 160.0})
        ps2 = pd.Series({'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0})

        os1 = orca.Series({'falcon': 330.0, 'eagle': 160.0})
        os2 = orca.Series({'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0})

        # TODO: orca.Series().combine()
        # c1 = ps1.combine(os2, max)
        # c2 = os1.combine(ps2, max)
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_combine_first(self):
        ps1 = pd.Series([1, np.nan])
        ps2 = pd.Series([3, 4])

        os1 = orca.Series([1, np.nan])
        os2 = orca.Series([3, 4])

        # TODO: orca.Series().combine_first()
        # c1 = ps1.combine_first(os2)
        # c2 = os1.combine_first(ps2)
        # assert_series_equal(c1, c2)

    def test_series_binary_operator_function_round(self):
        ps = pd.Series([0.1, 1.3, 2.7])
        os = orca.Series([0.1, 1.3, 2.7])
        # TODO: orca.Series().round()
        c1 = ps.round(1)
        c2 = os.round(1)
        assert_series_equal(c1, c2.to_pandas())

        pser = pd.Series([0.028208, 0.038683, 0.877076], name='x')
        oser = orca.Series(pser)
        # TODO: TypeError expected: integer argument expected, got float
        # msg = "integer argument expected, got float"
        # with self.assertRaisesRegex(TypeError, msg):
        #     pser.round(1.5)
        # with self.assertRaisesRegex(TypeError, msg):
        #     oser.round(1.5)

    @property
    def psla(self):
        return pd.Series({'dog': 1, 'cat': 2, 'pig': 3, 'cow': 4}, index=['dog', 'cat', 'pig', 'cow'])

    @property
    def pslb(self):
        return pd.Series({'dog': 1, 'cat': 3, 'pig': 2, 'cow': 5}, index=['dog', 'cat', 'pig', 'cow'])

    @property
    def osla(self):
        return orca.Series({'dog': 1, 'cat': 2, 'pig': 3, 'cow': 4}, index=['dog', 'cat', 'pig', 'cow'])

    @property
    def oslb(self):
        return orca.Series({'dog': 1, 'cat': 3, 'pig': 2, 'cow': 5}, index=['dog', 'cat', 'pig', 'cow'])

    def test_series_binary_operator_function_lt(self):
        # other = scalar value
        pc1 = self.psla.lt(3)
        oc1 = self.osla.lt(3).to_pandas()
        assert_series_equal(pc1, oc1)

        # other = Series
        pc2 = self.psla.lt(self.pslb)
        oc2 = self.osla.lt(self.oslb).to_pandas()
        assert_series_equal(pc2, oc2)

    def test_series_binary_operator_function_gt(self):
        # other = scalar value
        pc1 = self.psla.gt(3)
        oc1 = self.osla.gt(3).to_pandas()
        assert_series_equal(pc1, oc1)

        # other = Series
        pc2 = self.psla.gt(self.pslb)
        oc2 = self.osla.gt(self.oslb).to_pandas()
        assert_series_equal(pc2, oc2)

    def test_series_binary_operator_function_le(self):
        # other = scalar value
        pc1 = self.psla.le(3)
        oc1 = self.osla.le(3).to_pandas()
        assert_series_equal(pc1, oc1)

        # other = Series
        pc2 = self.psla.le(self.pslb)
        oc2 = self.osla.le(self.oslb).to_pandas()
        assert_series_equal(pc2, oc2)

    def test_series_binary_operator_function_ge(self):
        # other = scalar value
        pc1 = self.psla.ge(3)
        oc1 = self.osla.ge(3).to_pandas()
        assert_series_equal(pc1, oc1)

        # other = Series
        pc2 = self.psla.ge(self.pslb)
        oc2 = self.osla.ge(self.oslb).to_pandas()
        assert_series_equal(pc2, oc2)

    def test_series_binary_operator_function_ne(self):
        # other = scalar value
        pc1 = self.psla.ne(3)
        oc1 = self.osla.ne(3).to_pandas()
        assert_series_equal(pc1, oc1)

        # other = Series
        pc2 = self.psla.ne(self.pslb)
        oc2 = self.osla.ne(self.oslb).to_pandas()
        assert_series_equal(pc2, oc2)

    def test_series_binary_operator_function_eq(self):
        # other = scalar value
        pc1 = self.psla.eq(3)
        oc1 = self.osla.eq(3).to_pandas()
        assert_series_equal(pc1, oc1)

        # other = Series
        pc2 = self.psla.eq(self.pslb)
        oc2 = self.osla.eq(self.oslb).to_pandas()
        assert_series_equal(pc2, oc2)

    def test_series_binary_operator_function_product(self):
        # TODO: orca.Series().prod()
        pc1 = pd.Series([1]).prod()
        # oc1 = orca.Series([1]).prod()
        # assert_series_equal(pc1, oc1)
        #
        # pc2 = pd.Series([]).prod()
        # oc2 = orca.Series([]).prod()
        # assert_series_equal(pc2, oc2)
        #
        # pc3 = pd.Series([np.nan]).prod()
        # oc3 = orca.Series([np.nan]).prod()
        # assert_series_equal(pc3, oc3)

    def test_series_binary_operator_function_dot(self):
        ps = pd.Series([0, 1, 2, 3])
        psother = pd.Series([-1, 2, -3, 4])
        os = orca.Series([0, 1, 2, 3])
        osother = orca.Series([-1, 2, -3, 4])
        # TODO: orca.Series().dot()
        # # dot with series
        # pc1 = ps.dot(psother)
        # pc2 = ps @ psother
        # oc1 = os.dot(osother)
        # oc2 = os @ osother
        # assert_series_equal(pc1, oc1)
        # assert_series_equal(pc2, oc2)
        #
        # # dot with dataframe
        # pdfn = pd.DataFrame([[0, 1], [-2, 3], [4, -5], [6, 7]])
        # odfn = orca.DataFrame([[0, 1], [-2, 3], [4, -5], [6, 7]])
        # pc1 = ps.dot(pdfn)
        # pc2 = ps @ pdfn
        # oc1 = os.dot(odfn)
        # oc2 = os @ odfn
        # assert_series_equal(pc1, oc1)
        # assert_series_equal(pc2, oc2)
        #
        # # dot with array
        # arr = np.array([[0, 1], [-2, 3], [4, -5], [6, 7]])
        # pc1 = ps.dot(arr)
        # pc2 = ps @ arr
        # oc1 = os.dot(arr)
        # oc2 = os @ arr
        # assert_series_equal(pc1, oc1)
        # assert_series_equal(pc2, oc2)

    def test_series_function_application_groupby_window_ewm(self):
        ewmp = self.ps.ewm(com=0.5)
        ewmo = self.os.ewm(com=0.5)
        assert_series_equal(ewmp.mean(), ewmo.mean().to_pandas())
        assert_series_equal(ewmp.std(), ewmo.std().to_pandas())
        assert_series_equal(ewmp.var(), ewmo.var().to_pandas())
        # TODO: pairwise
        # assert_series_equal(ewmp.corr(), ewmo.corr().to_pandas())
        # assert_series_equal(ewmp.cov(), ewmo.cov().to_pandas())

        ewmp = self.ps.ewm(span=5)
        ewmo = self.os.ewm(span=5)
        assert_series_equal(ewmp.mean(), ewmo.mean().to_pandas())
        assert_series_equal(ewmp.std(), ewmo.std().to_pandas())
        assert_series_equal(ewmp.var(), ewmo.var().to_pandas())

        ewmp = self.ps.ewm(halflife=7)
        ewmo = self.os.ewm(halflife=7)
        assert_series_equal(ewmp.mean(), ewmo.mean().to_pandas())
        assert_series_equal(ewmp.std(), ewmo.std().to_pandas())
        assert_series_equal(ewmp.var(), ewmo.var().to_pandas())

        ewmp = self.ps.ewm(alpha=0.2)
        ewmo = self.os.ewm(alpha=0.2)
        assert_series_equal(ewmp.mean(), ewmo.mean().to_pandas())
        assert_series_equal(ewmp.std(), ewmo.std().to_pandas())
        assert_series_equal(ewmp.var(), ewmo.var().to_pandas())

        ewmp = self.ps.ewm(alpha=0.7, min_periods=2, adjust=False, ignore_na=True)
        ewmo = self.os.ewm(alpha=0.7, min_periods=2, adjust=False, ignore_na=True)
        assert_series_equal(ewmp.mean(), ewmo.mean().to_pandas())
        assert_series_equal(ewmp.std(), ewmo.std().to_pandas())
        assert_series_equal(ewmp.var(), ewmo.var().to_pandas())

    def test_series_missing_data_handling_fillna(self):
        ps = pd.Series([np.nan, 2, 3, 4, np.nan, 6], name='x')
        os = orca.Series(ps).to_pandas()

        self.assertEqual(repr(os.fillna(0)), repr(ps.fillna(0)))

        os.fillna(0, inplace=True)
        ps.fillna(0, inplace=True)
        assert_series_equal(os, ps)

    def test_series_missing_data_handling_dropna(self):
        ps = pd.Series([np.nan, 2, 3, 4, np.nan, 6], name='x')
        os = orca.Series(ps)
        # TODO：NOT IMPLEMENTED
        # assert_series_equal(os.dropna().to_pandas(), ps.dropna())
        #
        # os.dropna(inplace=True)
        # assert_series_equal(os.to_pandas(), ps.dropna())

    def test_series_missing_data_handling_isnull(self):
        ps = pd.Series([1, 2, 3, 4, np.nan, 6], name='x')
        os = orca.Series(ps)

        self.assertEqual(repr(os.notnull().to_pandas()), repr(ps.notnull()))
        self.assertEqual(repr(os.isnull().to_pandas()), repr(ps.isnull()))

        ps = self.ps
        os = self.os

        self.assertEqual(repr(os.notnull().to_pandas()), repr(ps.notnull()))
        self.assertEqual(repr(os.isnull().to_pandas()), repr(ps.isnull()))

    def test_series_reshaping_sorting_sort_values(self):
        ps = pd.Series([1, 2, 3, 4, 5, None, 7], name='0')
        os = orca.Series([1, 2, 3, 4, 5, None, 7], name='0')
        # TODO: orca.Series object has no attribute 'sort_values'
        # self.assertEqual(repr(os.sort_values()), repr(ps.sort_values()))
        # self.assertEqual(repr(os.sort_values(ascending=False)),
        #                  repr(ps.sort_values(ascending=False)))
        # self.assertEqual(repr(os.sort_values(na_position='first')),
        #                  repr(ps.sort_values(na_position='first')))
        # self.assertRaises(ValueError, lambda: os.sort_values(na_position='invalid'))
        # self.assertEqual(os.sort_values(inplace=True), ps.sort_values(inplace=True))
        # assert_series_equal(os, ps)

    def test_series_combining_joining_merging_append(self):
        ps1 = pd.Series([1, 2, 3], name='0')
        ps2 = pd.Series([4, 5, 6], name='0')
        ps3 = pd.Series([4, 5, 6], index=[3, 4, 5], name='0')
        os1 = orca.Series(ps1)
        os2 = orca.Series(ps2)
        os3 = orca.Series(ps3)
        # TODO：NOT IMPLEMENTED
        # os1 = os1.append(os2)
        # ps1 = ps1.append(ps2)
        # ps1.equals(os1)
        # self.assertTrue(ps1.equals(os1))
        # os1 = os1.append(os3)
        # ps1 = ps1.append(ps3)
        # self.assertTrue(ps1.equals(os1))
        # os1 = os1.append(os2, ignore_index=True)
        # ps1 = ps1.append(ps2, ignore_index=True)
        # self.assertTrue(ps1.equals(os1))
        #
        # os1.append(os3, verify_integrity=True)
        # msg = "Indices have overlapping values"
        # with self.assertRaises(ValueError, msg=msg):
        #     os1.append(os2, verify_integrity=True)

    # def test_series_map(self):
    #     pser = pd.Series(['cat', 'dog', None, 'rabbit'])
    #     oser = orca.DataFrame(pser)
    #     # Currently orca doesn't return NaN as Pandas does.
    #     self.assertEqual(
    #         repr(oser.map({})),
    #         repr(pser.map({}).replace({pd.np.nan: None}).rename(0)))
    #
    #     d = defaultdict(lambda: "abc")
    #     self.assertTrue("abc" in repr(oser.map(d)))
    #     self.assertEqual(
    #         repr(oser.map(d)),
    #         repr(pser.map(d).rename(0)))
