from unittest import TestCase
import concepts_toolbox as ct
import numpy as np
import pandas as pd


class Test(TestCase):
    def test_relationships_to_adjacency_df(self):
        rel_df = pd.DataFrame({'concept_id_1': [1, 2, 3],
                               'concept_id_2': [2, 3, 4],
                               'relationship_id': "Is a"})
        exp_df = pd.DataFrame(np.array([[0, 1, 0, 0], [0, 0, 1, 0],
                                        [0, 0, 0, 1], [0, 0, 0, 0]]),
                              index=[1, 2, 3, 4],
                              columns=[1, 2, 3, 4],
                              dtype=pd.SparseDtype(dtype=np.uint8,
                                                   fill_value=0))
        pd.testing.assert_frame_equal(
            ct.relationships_to_adjacency_df(rel_df, "Is a"),
            exp_df)
        pd.testing.assert_frame_equal(
            ct.relationships_to_adjacency_df(rel_df, "Is a",
                                             unique_concepts=[1, 2, 3, 4]),
            exp_df)
        # Test automatic ordering specifying unique_concept
        exp_df = pd.DataFrame(np.array([[0, 1, 0, 0], [0, 0, 0, 1],
                                        [0, 0, 0, 0], [0, 0, 1, 0]]),
                              index=[1, 2, 4, 3],
                              columns=[1, 2, 4, 3],
                              dtype=pd.SparseDtype(dtype=np.uint8,
                                                   fill_value=0))
        pd.testing.assert_frame_equal(
            ct.relationships_to_adjacency_df(rel_df, "Is a",
                                             unique_concepts=[1, 2, 4, 3]),
            exp_df)
        # Test result expansion with unique_concept superset
        exp_df = pd.DataFrame(np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                                        [0, 0, 0, 1, 0], [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]]),
                              index=[1, 2, 3, 4, 5],
                              columns=[1, 2, 3, 4, 5],
                              dtype=pd.SparseDtype(dtype=np.uint8,
                                                   fill_value=0))
        pd.testing.assert_frame_equal(
            ct.relationships_to_adjacency_df(rel_df, "Is a",
                                             unique_concepts=[1, 2, 3, 4, 5]),
            exp_df)
        # Test relationship reduction if unique_concept is a subset
        exp_df = pd.DataFrame(np.array([[0, 1, 0], [0, 0, 1],
                                        [0, 0, 0]]),
                              index=[1, 2, 3],
                              columns=[1, 2, 3],
                              dtype=pd.SparseDtype(dtype=np.uint8,
                                                   fill_value=0))
        pd.testing.assert_frame_equal(
            ct.relationships_to_adjacency_df(rel_df, "Is a",
                                             unique_concepts=[1, 2, 3]),
            exp_df)
        # Test multiple relationships
        rel_df = pd.DataFrame({'concept_id_1': [1, 2, 3, 1],
                               'concept_id_2': [2, 3, 4, 3],
                               'relationship_id': ["Is a"]*3+["Is Child"] })
        exp_df = pd.DataFrame(np.array([[0, 1, 1, 0], [0, 0, 1, 0],
                                        [0, 0, 0, 1], [0, 0, 0, 0]]),
                              index=[1, 2, 3, 4],
                              columns=[1, 2, 3, 4],
                              dtype=pd.SparseDtype(dtype=np.uint8,
                                                   fill_value=0))
        pd.testing.assert_frame_equal(
            ct.relationships_to_adjacency_df(rel_df, ["Is a","Is Child"]),
            exp_df)