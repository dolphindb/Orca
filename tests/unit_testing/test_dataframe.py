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

        Csv.odf_csv = orca.read_csv(data, dtype={"DLSTCD": np.float32, "DLPRC": np.float32})
        # pdf from import
        Csv.pdf_csv = pd.read_csv(data)
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
        assert_frame_equal(pdf, odf)

    def test_dataframe_constructor_from_dict_param_data_hasNan(self):
        d = {'col1': [np.NaN, 2], 'col2': [np.NaN, 4]}
        pdf = pd.DataFrame(data=d)
        odf = orca.DataFrame(data=d).to_pandas()
        assert_frame_equal(pdf, odf)

    def test_dataframe_constructor_from_dict_param_dtype(self):
        d = {'col1': [1, 2], 'col2': [3, 4]}
        pdf = pd.DataFrame(data=d, dtype=np.int8)
        odf = orca.DataFrame(data=d, dtype=np.int8).to_pandas()
        assert_frame_equal(pdf, odf)

    def test_dataframe_constructor_from_dict_param_dtype_hasNan(self):
        d = {'col1': [1, np.NaN], 'col2': [np.NaN, 4]}
        pdf = pd.DataFrame(data=d, dtype=np.int32)
        # TODO: dataframe_hasNan: fail to initialize a dataframe with np.nan values
        # odf = orca.DataFrame(data=d, dtype=np.int32).to_pandas()
        # assert_frame_equal(pdf, odf)

    def test_dataframe_constructor_from_ndarray_param_columns(self):
        nd = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        pdf = pd.DataFrame(nd, columns=['a', 'b', 'c'])
        odf = orca.DataFrame(nd, columns=['a', 'b', 'c']).to_pandas()
        assert_frame_equal(pdf, odf)

    def test_dataframe_constructor_from_ndarray_param_columns_hasNan(self):
        nd = np.array([[1, 2, 3], [np.NaN, np.NaN, np.NaN], [7, np.NaN, 9]])
        pdf = pd.DataFrame(nd, columns=['a', 'b', 'c'])
        odf = orca.DataFrame(nd, columns=['a', 'b', 'c']).to_pandas()
        assert_frame_equal(pdf, odf)

    def test_dataframe_attributes_index(self):
        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        assert_index_equal(pdf.index, odf.index.to_pandas())

    def test_dataframe_attributes_columns(self):
        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        assert_index_equal(pdf.columns, odf.columns)

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=pd.date_range("20190101", periods=4, freq="d"))
        # pd.to_datetime(["20190101","20190304"])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=orca.date_range("20190101", periods=4, freq="d"))
        assert_index_equal(pdf.columns, odf.columns)

        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=[1, 2, 3, 4])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=[1, 2, 3, 4])
        assert_index_equal(pdf.columns, odf.columns)

    def test_dataframe_attributes_dtypes(self):
        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        assert_series_equal(pdf.dtypes, odf.dtypes)

    def test_dataframe_attributes_ftypes(self):
        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        # TODO: Attributes: orca.DataFrame.ftypes
        # assert_series_equal(pdf.ftypes, odf.ftypes)

    def test_datagrame_attributes_select_dtypes(self):
        pdf = pd.DataFrame({'a': [1, 2] * 3, 'b': [True, False] * 3, 'c': [1.0, 2.0] * 3})
        odf = orca.DataFrame({'a': [1, 2] * 3, 'b': [True, False] * 3, 'c': [1.0, 2.0] * 3})
        assert_frame_equal(pdf.select_dtypes(include='bool'), odf.select_dtypes(include='bool').to_pandas())
        assert_frame_equal(pdf.select_dtypes(include=['float64']), odf.select_dtypes(include=['float64']).to_pandas())
        assert_frame_equal(pdf.select_dtypes(exclude=['int']), odf.select_dtypes(exclude=['int']).to_pandas())
        assert_frame_equal(pdf.select_dtypes(include=['float64', 'int']),
                           odf.select_dtypes(include=['float64', 'int']).to_pandas())
        assert_frame_equal(pdf.select_dtypes(exclude=['float64', 'int']),
                           odf.select_dtypes(exclude=['float64', 'int']).to_pandas())

    def test_dataframe_attributes_values(self):
        pdf = pd.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        assert_equal(pdf.values, odf.values)

    def test_dataframe_attributes_axes(self):
        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        odf = orca.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        assert_index_equal(pdf.axes[0], odf.axes[0].to_pandas())
        assert_index_equal(pdf.axes[1], odf.axes[1])

        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, pd.date_range("20190101", periods=2, freq="d"))
        odf = orca.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, orca.date_range("20190101", periods=2, freq="d"))
        assert_index_equal(pdf.axes[0], odf.axes[0].to_pandas())
        assert_index_equal(pdf.axes[1], odf.axes[1])

        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, index=['a', 'b'])
        odf = orca.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, index=['a', 'b'])
        assert_index_equal(pdf.axes[0], odf.axes[0].to_pandas())
        assert_index_equal(pdf.axes[1], odf.axes[1])

    def test_dataframe_attributes_ndim(self):
        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        odf = orca.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        self.assertEqual(pdf.ndim, odf.ndim)

    def test_dataframe_attributes_size(self):
        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        odf = orca.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        self.assertEqual(pdf.size, odf.size)
        self.assertEqual(self.pdf.size, self.odf.size)
        self.assertEqual(self.pdf_csv.size, self.odf_csv.size)

    def test_dataframe_attributes_shape(self):
        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        odf = orca.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        self.assertEqual(pdf.shape, odf.shape)
        self.assertEqual(self.pdf.shape, self.odf.shape)
        self.assertEqual(self.pdf_csv.shape, self.odf_csv.shape)

    def test_dataframe_attributes_empty(self):
        pdf = pd.DataFrame({})
        odf = orca.DataFrame({})
        self.assertEqual(pdf.empty, odf.empty)
        self.assertEqual(self.pdf.empty, self.odf.empty)
        self.assertEqual(self.pdf_csv.empty, self.odf_csv.empty)

    def test_dataframe_binary_operator_function_add_scalar(self):
        pdf = pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
        odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                             index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(pdf, odf.to_pandas())

        pre = pdf + 1
        ore = odf + 1
        assert_frame_equal(pre, ore.to_pandas())

        pre = pdf.add(1)
        ore = odf.add(1)
        assert_frame_equal(pre, ore.to_pandas())

    def test_dataframe_binary_operator_function_add_list(self):
        pdf = pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                           index=['circle', 'triangle', 'rectangle'])
        odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                             index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(pdf, odf.to_pandas())

        pre = pdf + [1, 2]
        ore = (odf + [1, 2]).to_pandas()
        assert_frame_equal(pre, ore)

        pre = pdf.add([1, 2])
        ore = odf.add([1, 2]).to_pandas()
        assert_frame_equal(pre, ore)

        pre = pdf.add([1, 2, 3], axis='index')
        ore = odf.add([1, 2, 3], axis='index').to_pandas()
        # TODO：NOT IMPLEMENTED
        # assert_frame_equal(pre, ore)

    def test_dataframe_binary_operator_function_add_series(self):
        pdf = pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                           index=['circle', 'triangle', 'rectangle'])
        odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                             index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(pdf, odf.to_pandas())

        # TODO: defalt axis= 'columns' or 0: orca.DataFrame + orca.Series or orca.DataFrame.add(orca.Series)
        # pre = pdf + pd.Series([1, 2], index=["angles", "degrees"])
        # ore = (odf + orca.Series([1, 2], index=["angles", "degrees"])).to_pandas()
        # assert_frame_equal(pre, ore)

        # pre = pdf.add(pd.Series([1, 2], index=["angles","degrees"]))
        # ore = (odf.add(orca.Series([1, 2], index=["angles","degrees"]))).to_pandas()
        # assert_frame_equal(pre, ore)

        # TODO: orca.DataFrame.add(orca.Series, axis = 'index' or 1)
        # a = pd.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle'])
        # b = orca.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle'])
        # pre = pdf.add(a, axis='index')
        # ore = odf.add(b, axis='index').to_pandas()
        # assert_frame_equal(pre, ore)

    def test_dataframe_binary_operator_function_sub_scalar(self):
        pdf = pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
        odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                             index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(pdf, odf.to_pandas())

        pre = pdf - 1
        ore = (odf - 1).to_pandas()
        assert_frame_equal(pre, ore)

        # pre = pdf.sub(1)
        # ore = odf.sub(1).to_pandas()
        # assert_frame_equal(pre, ore)

    def test_dataframe_binary_operator_function_sub_list(self):
        pdf = pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
        odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                             index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(pdf, odf.to_pandas())

        pre = pdf - [1, 2]
        ore = (odf - [1, 2]).to_pandas()
        assert_frame_equal(pre, ore)

        pre = pdf.sub([1, 2])
        ore = odf.sub([1, 2]).to_pandas()
        assert_frame_equal(pre, ore)

        pre = pdf.sub([1, 2, 3], axis='index')
        ore = odf.sub([1, 2, 3], axis='index').to_pandas()
        # TODO：NOT IMPLEMENTED
        # assert_frame_equal(pre, ore)

    def test_dataframe_binary_operator_function_sub_series(self):
        pdf = pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                           index=['circle', 'triangle', 'rectangle'])
        odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                             index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(pdf, odf.to_pandas())

        # TODO: defalt axis= 'columns' or 0: orca.DataFrame + orca.Series or orca.DataFrame.sub(orca.Series)
        # pre = pdf - pd.Series([1, 2, 3])
        # ore = (odf - orca.Series([1, 2, 3])).to_pandas()
        # assert_frame_equal(pre, ore)

        # pre = pdf.sub(pd.Series([1, 2, 3]))
        # ore = (odf.sub(orca.Series([1, 2, 3]))).to_pandas()
        # assert_frame_equal(pre, ore)

        # TODO: orca.DataFrame.sub(orca.Series, axis = 'index' or 1)
        # a = pd.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle'])
        # b = orca.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle'])
        # pre = pdf.sub(a, axis='index')
        # ore = odf.sub(b, axis='index').to_pandas()
        # assert_frame_equal(pre, ore)

    def test_dataframe_binary_operator_function_mul_dataframe(self):
        pdf = pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
        odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                             index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(pdf, odf.to_pandas())

        p_other = pd.DataFrame({'angles': [0, 3, 4]}, index=['circle', 'triangle', 'rectangle'])
        o_other = orca.DataFrame({'angles': [0, 3, 4]}, index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(p_other, o_other.to_pandas())

        p_index = pd.DataFrame({'angles': [3, 5, 8], 'degrees': [2, 5, 7]}, index=['circle', 'triangle', 'rectangle'])
        o_index = orca.DataFrame({'angles': [3, 5, 8], 'degrees': [2, 5, 7]}, index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(p_index, o_index.to_pandas())

        pre = pdf * p_other
        ore = (odf * o_other).to_pandas()
        assert_frame_equal(pre, ore)

        pre = pdf.mul(p_other)
        ore = odf.mul(o_other).to_pandas()
        assert_frame_equal(pre, ore)

        pre = pdf.mul(p_index)
        ore = odf.mul(o_index).to_pandas()
        assert_frame_equal(pre, ore)

    def test_dataframe_binary_operator_function_div_scalar(self):
        pdf = pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
        odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                             index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(pdf, odf.to_pandas())

        pre = pdf / 2
        ore = (odf / 2).to_pandas()
        assert_frame_equal(pre, ore)

        # pre = pdf.div(2)
        # ore = odf.div(2).to_pandas()
        # assert_frame_equal(pre, ore)

    def test_dataframe_binary_operator_function_rdiv_scalar(self):
        pdf = pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
        odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                             index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(pdf, odf.to_pandas())

        pre = pdf.rdiv(2)
        ore = odf.rdiv(2).to_pandas()
        # 2/0 in pandas equals to inf while in orca equals to nan,
        # thus we replace these values with zeros for correctness assertion
        pre[np.isinf(pre)] = 0
        ore[np.isnan(ore)] = 0
        assert_frame_equal(pre, ore)

    def test_dataframe_binary_operator_function_div_multiIndex_param_level_param_fill_value(self):
        pdf = pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]}, index=['circle', 'triangle', 'rectangle'])
        odf = orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                             index=['circle', 'triangle', 'rectangle'])

        pdf_multi = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6], 'degrees': [360, 180, 360, 360, 540, 720]},
                                 index=[['A', 'A', 'A', 'B', 'B', 'B'],
                                        ['circle', 'triangle', 'rectangle', 'square', 'pentagon', 'hexagon']])
        odf_multi = orca.DataFrame({'angles': [0, 3, 4, 4, 5, 6], 'degrees': [360, 180, 360, 360, 540, 720]},
                                   index=[['A', 'A', 'A', 'B', 'B', 'B'],
                                          ['circle', 'triangle', 'rectangle', 'square', 'pentagon', 'hexagon']])
        assert_frame_equal(pdf_multi, odf_multi.to_pandas())

        # TODO: orca.DataFrame.rdiv(orca.DataFrame, level=1, fill_value=0)
        # pre = pdf.rdiv(pdf_multi, level=1, fill_value=0)
        # ore = odf.rdiv(odf_multi, level=1, fill_value=0).to_pandas()
        # assert_frame_equal(pre, ore)

    def test_dataframe_Function_application_GroupBy_window_apply(self):
        pdf = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
        odf = orca.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
        assert_frame_equal(pdf.apply(np.sqrt), odf.apply(np.sqrt).to_pandas())
        # assert_series_equal(pdf.apply(np.sum, axis=0), odf.apply(np.sum, axis=0).to_pandas())

    def test_dataframe_Function_application_GroupBy_window_applymap(self):
        pass

    def test_dataframe_Function_application_GroupBy_window_pipe(self):
        pass

    def test_dataframe_Function_application_GroupBy_window_agg(self):
        # pdf = pd.DataFrame([[1, 2, 3],[4, 5, 6],[7, 8, 9],[np.nan, np.nan, np.nan]],columns = ['A', 'B', 'C'])
        # odf = orca.DataFrame([[1, 2, 3],[4, 5, 6],[7, 8, 9],[np.nan, np.nan, np.nan]],columns = ['A', 'B', 'C'])
        # pdf.agg(['sum', 'min'])
        # odf.agg(['sum', 'min'])
        pass

    def test_dataframe_Function_application_GroupBy_window_aggregate(self):
        pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]], columns=['A', 'B', 'C'])
        odf = orca.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]], columns=['A', 'B', 'C'])
        assert_frame_equal(pdf.aggregate(['sum', 'min']), odf.aggregate(['sum', 'min']).to_pandas())
        assert_frame_equal(pdf.aggregate({'A': ['sum', 'min'], 'B': ['min', 'max']}),
                           odf.aggregate({'A': ['sum', 'min'], 'B': ['min', 'max']}).to_pandas())
        # assert_frame_equal(pdf.aggregate("mean", axis="columns"), odf.aggregate("mean", axis="columns").to_pandas())

    def test_dataframe_Function_application_GroupBy_window_transform(self):
        pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]], columns=['A', 'B', 'C'])
        odf = orca.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]], columns=['A', 'B', 'C'])
        assert_frame_equal(pdf.transform([np.sqrt, np.exp]), odf.transform([np.sqrt, np.exp]).to_pandas())

    def test_dataframe_Function_application_GroupBy_window_expanding(self):
        # pdf = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
        # odf = orca.DataFrame({'B': [0, 1, 2, np.nan, 4]})
        # assert_frame_equal(pdf.expanding(2).sum(), odf.expanding(2).sum().to_pandas())
        pass

    def test_dataframe_Function_application_GroupBy_window_ewm(self):
        ewmp = self.pdf.ewm(com=0.5)
        ewmo = self.odf.ewm(com=0.5)
        assert_frame_equal(ewmp.mean(), ewmo.mean().to_pandas())
        assert_frame_equal(ewmp.std(), ewmo.std().to_pandas())
        assert_frame_equal(ewmp.var(), ewmo.var().to_pandas())
        # TODO: pairwise
        # assert_frame_equal(ewmp.corr(), ewmo.corr().to_pandas())
        # assert_frame_equal(ewmp.cov(), ewmo.cov().to_pandas())

        ewmp = self.pdf.ewm(span=5)
        ewmo = self.odf.ewm(span=5)
        assert_frame_equal(ewmp.mean(), ewmo.mean().to_pandas())
        assert_frame_equal(ewmp.std(), ewmo.std().to_pandas())
        assert_frame_equal(ewmp.var(), ewmo.var().to_pandas())

        ewmp = self.pdf.ewm(halflife=7)
        ewmo = self.odf.ewm(halflife=7)
        assert_frame_equal(ewmp.mean(), ewmo.mean().to_pandas())
        assert_frame_equal(ewmp.std(), ewmo.std().to_pandas())
        assert_frame_equal(ewmp.var(), ewmo.var().to_pandas())

        ewmp = self.pdf.ewm(alpha=0.2)
        ewmo = self.odf.ewm(alpha=0.2)
        assert_frame_equal(ewmp.mean(), ewmo.mean().to_pandas())
        assert_frame_equal(ewmp.std(), ewmo.std().to_pandas())
        assert_frame_equal(ewmp.var(), ewmo.var().to_pandas())

        ewmp = self.pdf.ewm(alpha=0.7, min_periods=2, adjust=False, ignore_na=True)
        ewmo = self.odf.ewm(alpha=0.7, min_periods=2, adjust=False, ignore_na=True)
        assert_frame_equal(ewmp.mean(), ewmo.mean().to_pandas())
        assert_frame_equal(ewmp.std(), ewmo.std().to_pandas())
        assert_frame_equal(ewmp.var(), ewmo.var().to_pandas())

        csvp = self.pdf_csv.ewm(com=0.5)
        csvo = self.odf_csv.ewm(com=0.5)
        assert_frame_equal(csvp.mean(), csvo.mean().to_pandas())
        assert_frame_equal(csvp.std(), csvo.std().to_pandas())
        assert_frame_equal(csvp.var(), csvo.var().to_pandas())

        # TODO: pairwise
        # assert_frame_equal(ewmp.corr(), ewmo.corr().to_pandas())
        # assert_frame_equal(ewmp.cov(), ewmo.cov().to_pandas())

        csvp = self.pdf_csv.ewm(span=5)
        csvo = self.odf_csv.ewm(span=5)
        assert_frame_equal(csvp.mean(), csvo.mean().to_pandas())
        assert_frame_equal(csvp.std(), csvo.std().to_pandas())
        assert_frame_equal(csvp.var(), csvo.var().to_pandas())

        csvp = self.pdf_csv.ewm(halflife=7)
        csvo = self.odf_csv.ewm(halflife=7)
        assert_frame_equal(csvp.mean(), csvo.mean().to_pandas())
        assert_frame_equal(csvp.std(), csvo.std().to_pandas())
        assert_frame_equal(csvp.var(), csvo.var().to_pandas())

        csvp = self.pdf_csv.ewm(alpha=0.2)
        csvo = self.odf_csv.ewm(alpha=0.2)
        assert_frame_equal(csvp.mean(), csvo.mean().to_pandas())
        assert_frame_equal(csvp.std(), csvo.std().to_pandas())
        assert_frame_equal(csvp.var(), csvo.var().to_pandas())

        csvp = self.pdf_csv.ewm(alpha=0.7, min_periods=2, adjust=False, ignore_na=True)
        csvo = self.odf_csv.ewm(alpha=0.7, min_periods=2, adjust=False, ignore_na=True)
        assert_frame_equal(csvp.mean(), csvo.mean().to_pandas())
        assert_frame_equal(csvp.std(), csvo.std().to_pandas())
        assert_frame_equal(csvp.var(), csvo.var().to_pandas())

    def test_dataframe_computations(self):
        # from csv
        # TODO:orca不支持对string类型做all和any运算 getBoolConst method not supported
        # assert_frame_equal(self.pdf_csv.all(), self.odf_csv.all().to_pandas())
        # assert_frame_equal(self.pdf_csv.any(), self.odf_csv.any().to_pandas())
        assert_frame_equal(self.pdf_csv.corr(), self.odf_csv.corr().to_pandas(), check_dtype=False)
        assert_series_equal(self.pdf_csv.count(), self.odf_csv.count().to_pandas(), check_dtype=False)
        assert_frame_equal(self.pdf_csv.cov(), self.odf_csv.cov().to_pandas())
        # TODO:pandas不支持对string类型做kurt运算 而orca可以
        # assert_series_equal(self.pdf_csv.kurt(), self.odf_csv.kurt().to_pandas())
        # assert_frame_equal(self.pdf_csv.kurtosis(), self.odf_csv.kurtosis().to_pandas())
        assert_series_equal(self.pdf_csv.mean(), self.odf_csv.mean().to_pandas())
        # assert_frame_equal(self.pdf_csv.median(), self.odf_csv.median().to_pandas())
        # assert_series_equal(self.pdf_csv.min(), self.odf_csv.min().to_pandas(), check_dtype=False)
        # assert_series_equal(self.pdf_csv.max(), self.odf_csv.max().to_pandas(), check_dtype=False, check_less_precise=True)
        # assert_series_equal(self.pdf_csv.mode(), self.odf_csv.mode().to_pandas())
        # assert_series_equal(self.pdf_csv.prod(), self.odf_csv.prod().to_pandas(), check_less_precise=1)
        # assert_series_equal(self.pdf_csv.product(), self.odf_csv.product().to_pandas())
        # assert_series_equal(self.pdf_csv.skew(), self.odf_csv.skew().to_pandas())
        # assert_series_equal(self.pdf_csv.sum(), self.odf_csv.sum().to_pandas())
        assert_series_equal(self.pdf_csv.std(), self.odf_csv.std().to_pandas())
        assert_series_equal(self.pdf_csv.var(), self.odf_csv.var().to_pandas())

        # # from construction
        # assert_frame_equal(self.pdf.all(), self.odf.all().to_pandas())
        # assert_frame_equal(self.pdf.any(), self.odf.any().to_pandas())
        # assert_frame_equal(self.pdf.corr(), self.odf.corr().to_pandas())
        # assert_frame_equal(self.pdf.count(), self.odf.count().to_pandas())
        # assert_frame_equal(self.pdf.cov(), self.odf.cov().to_pandas())
        # assert_frame_equal(self.pdf.kurt(), self.odf.kurt().to_pandas())
        # assert_frame_equal(self.pdf.kurtosis(), self.odf.kurtosis().to_pandas())
        # assert_frame_equal(self.pdf.mean(), self.odf.mean().to_pandas())
        # assert_frame_equal(self.pdf.median(), self.odf.median().to_pandas())
        # assert_frame_equal(self.pdf.min(), self.odf.mint().to_pandas())
        # assert_frame_equal(self.pdf.max(), self.odf.max().to_pandas())
        # assert_frame_equal(self.pdf.mode(), self.odf.mode().to_pandas())
        # assert_frame_equal(self.pdf.prod(), self.odf.prod().to_pandas())
        # assert_frame_equal(self.pdf.product(), self.odf.product().to_pandas())
        # assert_frame_equal(self.pdf.skew(), self.odf.skew().to_pandas())
        # assert_frame_equal(self.pdf.sum(), self.odf.sum().to_pandas())
        # assert_frame_equal(self.pdf.std(), self.odf.std().to_pandas())
        # assert_frame_equal(self.pdf.var(), self.odf.var().to_pandas())
        #
        # assert_frame_equal(self.pdf.pct_change(), self.odf.pct_change().to_pandas)
        # assert_frame_equal(self.pdf.mad(), self.odf.mad().to_pandas())
        # assert_frame_equal(self.pdf.sem(), self.odf.sem().to_pandas())
        # assert_frame_equal(self.pdf.cumsum(), self.odf.cumsum().to_pandas())
        # assert_frame_equal(self.pdf.cummax(), self.odf.cummax().to_pandas())
        # assert_frame_equal(self.pdf.cummin(), self.odf.cummin().to_pandas())
        # assert_frame_equal(self.pdf.cumprod(), self.odf.cumprod().to_pandas())

    def test_dataframe_append(self):
        n = 10  # note that n should be a multiple of 10
        re = n / 10
        pdf1 = pd.DataFrame({'id': np.arange(1, n + 1, 1, dtype='int32'),
                             'date': np.repeat(pd.date_range('2019.08.01', periods=10, freq='D'), re),
                             'tsymbol': np.repeat(['a', 'b', 'c', 'd', 'e', 'QWW', 'FEA', 'FFW', 'DER', 'POD'], re),
                             'tbool': np.repeat(np.repeat(np.arange(2, dtype='bool'), 5), re),
                             'tchar': np.repeat(np.arange(1, 11, 1, dtype='int8'), re),
                             'tshort': np.repeat(np.arange(1, 11, 1, dtype='int16'), re),
                             'tint': np.repeat(np.arange(1, 11, 1, dtype='int32'), re),
                             'tlong': np.repeat(np.arange(1, 11, 1, dtype='int64'), re),
                             'tfloat': np.repeat(np.arange(1, 11, 1, dtype='float32'), re),
                             'tdouble': np.repeat(np.arange(1, 11, 1, dtype='float64'), re)
                             })
        n = 20  # note that n should be a multiple of 10
        re = n / 10
        pdf2 = pd.DataFrame({'id': np.arange(1, n + 1, 1, dtype='int32'),
                             'date': np.repeat(pd.date_range('2019.08.01', periods=10, freq='D'), re),
                             'tsymbol': np.repeat(['a', 'b', 'c', 'd', 'e', 'QWW', 'FEA', 'FFW', 'DER', 'POD'], re),
                             'tbool': np.repeat(np.repeat(np.arange(2, dtype='bool'), 5), re),
                             'tchar': np.repeat(np.arange(1, 11, 1, dtype='int8'), re),
                             'tshort': np.repeat(np.arange(1, 11, 1, dtype='int16'), re),
                             'tint': np.repeat(np.arange(1, 11, 1, dtype='int32'), re),
                             'tlong': np.repeat(np.arange(1, 11, 1, dtype='int64'), re),
                             'tfloat': np.repeat(np.arange(1, 11, 1, dtype='float32'), re),
                             'tdouble': np.repeat(np.arange(1, 11, 1, dtype='float64'), re)
                             })

        odf1 = orca.DataFrame(pdf1)
        odf2 = orca.DataFrame(pdf2)
        assert_frame_equal(pdf1.append(pdf2), odf1.append(odf2).to_pandas())

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
        self.assertEqual(repr(df[['a', 'b']]), repr(ddf[['a', 'b']]))

        # TODO：NOT IMPLEMENTED
        # self.assertEqual(repr(ddf.a.notnull().alias("x").name), repr("x"))

        # check orca.DataFrame(os.Series)
        pser = pd.Series([1, 2, 3], name='x')
        kser = orca.Series([1, 2, 3], name='x')
        self.assertEqual(repr(pd.DataFrame(pser)), repr(orca.DataFrame(kser)))

    def test_dataframe_multiindex_columns(self):
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

    def test_dataframe_column_level_name(self):
        column = pd.Index(['A', 'B', 'C'], name='X')
        pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=column)
        odf = orca.DataFrame(pdf)

        self.assertEqual(repr(odf), repr(pdf))
        self.assertEqual(repr(odf.columns.names), repr(pdf.columns.names))
        self.assertEqual(repr(odf.columns.names), repr(pdf.columns.names))

    def test_dataframe_multiindex_names_level(self):
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

    def test_reset_index_with_multiindex_columns(self):
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

    def test_multiindex_column_access(self):
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
