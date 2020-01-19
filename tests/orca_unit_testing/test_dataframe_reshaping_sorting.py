import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class DataFrameReashapingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_dataframe_reshaping_sorting_transposing_droplevel(self):
        pdf = pd.DataFrame([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]
                            ]).set_index([0, 1]).rename_axis(['a', 'b'])

        pdf.columns = pd.MultiIndex.from_tuples([
            ('c', 'e'), ('d', 'f')
        ], names=['level_1', 'level_2'])
        odf = orca.DataFrame(pdf)
        assert_frame_equal(pdf.droplevel('a'), odf.droplevel('a').to_pandas())
        # TODO：NotImplementedError: Orca does not support axis == 1
        # assert_frame_equal(pdf.droplevel('level_2',axis=1), odf.droplevel('level_2',axis=1).to_pandas())

    def test_dataframe_reshaping_sorting_transposing_pivot(self):
        pdf = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                            'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                            'baz': [1, 2, 3, 4, 5, 6],
                            'zoo': [2, 4, 5, 6, 4, 7]})
        odf = orca.DataFrame(pdf)
        ptable = pdf.pivot(index='foo', columns='bar', values='baz')
        otable = odf.pivot(index='foo', columns='bar', values='baz')
        assert_frame_equal(otable.to_pandas(), ptable)

        # TODO:orca不支持多个values
        ptable = pdf.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
        # otable = odf.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
        # assert_frame_equal(otable.to_pandas(), ptable)
        # TODO:orcab不支持下标
        ptable = pdf.pivot(index='foo', columns='bar')['baz']
        # otable = odf.pivot(index='foo', columns='bar')['baz']
        # assert_frame_equal(otable.to_pandas(), ptable)
        # TODO:orca不支持index，columns参数为多个列组成的list

    def test_dataframe_reshaping_sorting_transposing_pivot_table(self):
        pdf = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                            "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                            "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                            "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                            "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
        odf = orca.DataFrame(pdf)
        ptable = pdf.pivot_table(values='D', index='A', columns='C', aggfunc="sum")
        otable = odf.pivot_table(values='D', index='A', columns='C', aggfunc="sum")
        assert_frame_equal(otable.to_pandas(), ptable)

        pdf = pd.DataFrame({"time": pd.date_range("15:00:00", periods=9, freq="30s"),
                            "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                            "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                            "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                            "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                            "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
                            "F": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
        odf = orca.DataFrame(pdf)
        ptable = pdf.pivot_table(values='D', index=pdf.time.dt.minute, columns='A', aggfunc="sum")
        otable = odf.pivot_table(values='D', index=odf.time.dt.minute, columns='A', aggfunc="sum")
        assert_frame_equal(otable.to_pandas(), ptable)
        ptable = pdf.pivot_table(values='D', index='A', columns=pdf.time.dt.minute, aggfunc="sum")
        otable = odf.pivot_table(values='D', index='A', columns=odf.time.dt.minute, aggfunc="sum")
        # TODO:DIFFERENT COLNAME
        # assert_frame_equal(otable.to_pandas(), ptable)

        # TODO:orca不支持index，columns参数为多个列组成的list

    def test_dataframe_reshaping_sorting_transposing_reorder_levels(self):
        arrays = [np.array(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux']),
                  np.array(['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'])]
        pdf = pd.DataFrame(np.random.randn(8, 4), index=arrays)
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.reorder_levels([1, 0], axis=0).to_pandas(), pdf.reorder_levels([1, 0], axis=0))

    def test_dataframe_reshaping_sorting_transposing_stack(self):
        pdf = pd.DataFrame([[0, 1], [2, 3]], index=['cat', 'dog'], columns=['weight', 'height'])
        odf = orca.DataFrame(pdf)
        # TODO: orca的结果不分组
        # assert_series_equal(odf.stack().to_pandas(), pdf.stack())
        pdf = pd.DataFrame([[1, 2], [2, 4]], index=['cat', 'dog'],
                           columns=pd.MultiIndex.from_tuples([('weight', 'kg'), ('weight', 'pounds')]))
        odf = orca.DataFrame(pdf)
        # TODO： orca不支持multi level columns
        # assert_series_equal(odf.stack().to_pandas(), pdf.stack())
        pdf = pd.DataFrame({"time": pd.date_range("15:00:00", periods=9, freq="30s"),
                            "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                            "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                            "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                            "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                            "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
                            "F": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
        odf = orca.DataFrame(pdf)
        ptable = pdf.pivot_table(values='D', index=pdf.time.dt.minute, columns='A', aggfunc="sum")
        otable = odf.pivot_table(values='D', index=odf.time.dt.minute, columns='A', aggfunc="sum")
        # TODO： orca的结果不分组
        # assert_series_equal(otable.stack().to_pandas(), ptable.stack())

    def test_dataframe_reshaping_sorting_transposing_transpose_T(self):
        pdf = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.T.to_pandas(), pdf.T)
        assert_frame_equal(odf.transpose().to_pandas(), pdf.transpose())
        assert_frame_equal(odf.transpose(copy=True).to_pandas(), pdf.transpose(copy=True))

        # TODO：transpose on non-numerical types of matrix is not allowed in Orca
        pdf = pd.DataFrame({'name': ['Alice', 'Bob'], 'grade': ['A', 'B'], 'Gender': ['female', 'male']})
        odf = orca.DataFrame(pdf)

    def test_dataframe_reshaping_sorting_transposing_melt(self):
        pdf = pd.DataFrame({"time": pd.date_range("15:00:00", periods=9, freq="30s"),
                            "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                            "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                            "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                            "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                            "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
                            "F": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
        odf = orca.DataFrame(pdf)
        ptable=pdf.melt(id_vars=['A', 'B'], value_vars=['D', 'E'])
        otable = odf.melt(id_vars=['A', 'B'], value_vars=['D', 'E'])
        assert_frame_equal(otable.to_pandas(), ptable)

        ptable = pdf.melt(id_vars='A', value_vars='D', var_name="myVarname", value_name="myValname")
        otable = odf.melt(id_vars='A', value_vars='D', var_name="myVarname", value_name="myValname")
        assert_frame_equal(otable.to_pandas(), ptable)

        pdf = pd.DataFrame({'A': {0: 101, 1: 102, 2: 103}, 'B': {0: 1, 1: 3, 2: 5}, 'C': {0: 2, 1: 4, 2: 6}})
        odf = orca.DataFrame(pdf)
        ptable = pdf.melt()
        otable = odf.melt()
        assert_frame_equal(otable.to_pandas(), ptable)

        ptable = pdf.melt(id_vars=['A'])
        otable = odf.melt(id_vars=['A'])
        assert_frame_equal(otable.to_pandas(), ptable)

        ptable = pdf.melt(value_vars=['A'])
        otable = odf.melt(value_vars=['A'])
        assert_frame_equal(otable.to_pandas(), ptable)


if __name__ == '__main__':
    unittest.main()
