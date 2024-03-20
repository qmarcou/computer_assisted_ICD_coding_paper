from unittest import TestCase
import pandas as pd
import pandas_utils as pu


class Test(TestCase):
    def test_lists_to_rows(self):
        test_df = pd.DataFrame({'A': ['a', 'b'],
                                'B': [['c', 'd'], ['e']]})
        test_df2 = pd.DataFrame({'A': ['a', 'b'],
                                 'B': [['c', 'd'], 'e']})
        expect_df = pd.DataFrame({'A': ['a', 'a', 'b'],
                                  'B': ['c', 'd', 'e']}, index=[0, 0, 1])

        pd.testing.assert_frame_equal(pu.lists_to_rows(test_df, 'B'),
                                      expect_df)
        pd.testing.assert_frame_equal(pu.lists_to_rows(test_df2, 'B'),
                                      expect_df)
        # check that col with no list is left unchanged
        pd.testing.assert_frame_equal(pu.lists_to_rows(test_df, 'A'),
                                      test_df)
        # test results for DF with degenerate index (duplicate index)
        test_df3 = pd.DataFrame({'A': ['a', 'b'],
                                'B': [['c', 'd'], ['e']]}, index=[0, 0])
        expect_df2 = pd.DataFrame({'A': ['a', 'a', 'b'],
                                  'B': ['c', 'd', 'e']}, index=[0, 0, 0])
        pd.testing.assert_frame_equal(pu.lists_to_rows(test_df3, 'B'),
                                      expect_df2)