import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


def _create_odf_csv(datal, datar):
    dfsDatabase = "dfs://testMergeDB"
    s = orca.default_session()
    dolphindb_script = """
                        login('admin', '123456')
                        if(existsDatabase('{dbPath}'))
                           dropDatabase('{dbPath}')
                        db=database('{dbPath}', VALUE, `a`b`c`d`e`f`g)
                        stb1=extractTextSchema('{data1}')
                        update stb1 set type="SYMBOL" where name="type"
                        stb2=extractTextSchema('{data2}')
                        update stb2 set type="SYMBOL" where name="type"
                        loadTextEx(db,`tickers,`type, '{data1}',,stb1)
                        loadTextEx(db,`values,`type, '{data2}',,stb2)
                        """.format(dbPath=dfsDatabase, data1=datal, data2=datar)
    s.run(dolphindb_script)


class Csv:
    odf_csv_left = None
    odf_csv_right = None
    odfs_csv_left = None
    odfs_csv_right = None
    pdf_csv_left = None
    pdf_csv_right = None


class DfsMergeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        left_fileName = 'test_merge_left_table.csv'
        right_fileName = 'test_merge_right_table.csv'
        datal = os.path.join(DATA_DIR, left_fileName)
        datal= datal.replace('\\', '/')
        datar = os.path.join(DATA_DIR, right_fileName)
        datar = datar.replace('\\', '/')
        dfsDatabase = "dfs://testMergeDB"

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")
        _create_odf_csv(datal, datar)

        # import
        Csv.odf_csv_left = orca.read_csv(datal)
        Csv.odfs_csv_left = orca.read_table(dfsDatabase, 'tickers')
        Csv.pdf_csv_left = pd.read_csv(datal, parse_dates=[0,1])
        Csv.odf_csv_right = orca.read_csv(datar)
        Csv.odfs_csv_right = orca.read_table(dfsDatabase, 'values')
        Csv.pdf_csv_right = pd.read_csv(datar)

    @property
    def odfs_csv_left(self):
        return Csv.odfs_csv_left

    @property
    def odfs_csv_right(self):
        return Csv.odfs_csv_right

    @property
    def pdf_csv_left(self):
        return Csv.pdf_csv_left

    @property
    def pdf_csv_right(self):
        return Csv.pdf_csv_right

    @property
    def odfs_csv_left_index(self):
        return Csv.odfs_csv_left.set_index("type")

    @property
    def odfs_csv_right_index(self):
        return Csv.odfs_csv_right.set_index("type")

    @property
    def pdf_csv_left_index(self):
        return Csv.pdf_csv_left.set_index("type")

    @property
    def pdf_csv_right_index(self):
        return Csv.pdf_csv_right.set_index("type")

    # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
    # def test_assert_original_dataframe_equal(self):
    #     assert_frame_equal(self.odfs_csv_left.sort_values("date").to_pandas(), self.pdf_csv_left.sort_values("date"), check_dtype=False, check_like=False)
    #     assert_frame_equal(self.odfs_csv_right.sort_values("date").to_pandas(), self.pdf_csv_right.sort_values("date"), check_dtype=False)
    #     assert_frame_equal(self.odfs_csv_left_index.sort_index().to_pandas(), self.pdf_csv_left_index.sort_index(), check_dtype=False, check_like=False)
    #     assert_frame_equal(self.odfs_csv_right_index.sort_index().to_pandas(), self.pdf_csv_right_index.sort_index(), check_dtype=False, check_like=False)

    def test_merge_from_dfs_param_suffix(self):
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, on="type", suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, on="type", suffixes=('_left', '_right'))
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False)

    def test_merge_from_dfs_param_how(self):
        # how = left
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, how="left", on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="left", on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)
        # how = right
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, how="right", on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="right", on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # default how = inner
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # how = outer
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, how="outer", on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="outer", on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_dfs_param_on(self):
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_dfs_param_leftonrighton(self):
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, left_on="type", right_on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, left_on="type", right_on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_dfs_param_how_param_leftonrighton(self):
        # how = left
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, how="left", left_on="type", right_on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="left", left_on="type", right_on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)
        # how = right
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, how="right", left_on="type", right_on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="right", left_on="type", right_on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # default how = inner
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right,  left_on="type", right_on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right,  left_on="type", right_on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # how = outer
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, how="outer", left_on="type", right_on="type")
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="outer", left_on="type", right_on="type")
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_dfs_param_index(self):
        odfs_merge = self.odfs_csv_left_index.merge(self.odfs_csv_right_index, left_index=True, right_index=True)
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, left_index=True, right_index=True)
        assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False)

    def test_merge_from_dfs_index_param_suffix(self):
        odfs_merge = self.odfs_csv_left_index.merge(self.odfs_csv_right_index, left_index=True, right_index=True, suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, left_index=True, right_index=True, suffixes=('_left', '_right'))
        assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False)

    def test_merge_from_dfs_index_param_on(self):
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right_index, left_on="type", right_index=True)
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right_index, left_on="type", right_index=True)
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        odfs_merge = self.odfs_csv_left_index.merge(self.odfs_csv_right, right_on="type", left_index=True)
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right, right_on="type", left_index=True)
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_dfs_index_param_how(self):
        # how = left
        odfs_merge = self.odfs_csv_left_index.merge(self.odfs_csv_right_index, how="left", left_index=True, right_index=True)
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="left", left_index=True, right_index=True)
        assert_frame_equal(odfs_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False)
        # how = right
        odfs_merge = self.odfs_csv_left_index.merge(self.odfs_csv_right_index, how="right", left_index=True, right_index=True)
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="right", left_index=True, right_index=True)
        assert_frame_equal(odfs_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False)

        # default how = inner
        odf_merge = self.odfs_csv_left_index.merge(self.odfs_csv_right_index, how="inner", left_index=True, right_index=True)
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="inner", left_index=True, right_index=True)
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False)

        # how = outer
        odfs_merge = self.odfs_csv_left_index.merge(self.odfs_csv_right_index, how="outer", left_index=True, right_index=True)
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="outer", left_index=True, right_index=True)
        assert_frame_equal(odfs_merge.sort_index().to_pandas(), pdf_merge, check_dtype=False)

    def test_merge_from_dfs_param_suffix_param_how(self):
        # how = left
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, how="left", on="type", suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="left", on="type", suffixes=('_left', '_right'))
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)
        # how = right
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, how="right", on="type", suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="right", on="type", suffixes=('_left', '_right'))
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # default how = inner
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, on="type", suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, on="type", suffixes=('_left', '_right'))
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # how = outer
        odfs_merge = self.odfs_csv_left.merge(self.odfs_csv_right, how="outer", on="type", suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left.merge(self.pdf_csv_right, how="outer", on="type", suffixes=('_left', '_right'))
        # TODO: PARTITIONED TABLE IS IN RANDOM ORDER
        # assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

    def test_merge_from_dfs_index_param_suffix_param_how(self):
        # how = left
        odfs_merge = self.odfs_csv_left_index.merge(self.odfs_csv_right_index, how="left", left_index=True, right_index=True, suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="left", left_index=True, right_index=True, suffixes=('_left', '_right'))
        assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False , check_like=False)
        # how = right
        odfs_merge = self.odfs_csv_left_index.merge(self.odfs_csv_right_index, how="right", left_index=True, right_index=True, suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="right", left_index=True, right_index=True, suffixes=('_left', '_right'))
        assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # default how = inner
        odfs_merge = self.odfs_csv_left_index.merge(self.odfs_csv_right_index, left_index=True, right_index=True,  suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, left_index=True, right_index=True,  suffixes=('_left', '_right'))
        assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)

        # how = outer
        odfs_merge = self.odfs_csv_left_index.merge(self.odfs_csv_right_index, how="outer", left_index=True, right_index=True,  suffixes=('_left', '_right'))
        pdf_merge = self.pdf_csv_left_index.merge(self.pdf_csv_right_index, how="outer",left_index=True, right_index=True,  suffixes=('_left', '_right'))
        assert_frame_equal(odfs_merge.to_pandas(), pdf_merge, check_dtype=False, check_like=False)


if __name__ == '__main__':
    unittest.main()
