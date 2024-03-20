from unittest import TestCase
import hot_encoding as oh
import pandas as pd
import numpy as np
from scipy.sparse import isspmatrix


class TestOccurMatrices(TestCase):
    test_df = pd.DataFrame({'A': [0, 0],
                            'B': [1, 1],
                            'C': [1, 0]})

    # TODO: remove use of np.matrix
    def test__oh_df_to_mat(self):
        # Values checks
        np.testing.assert_equal(
            oh._oh_df_to_mat(self.test_df, freq=False,
                             negate=False, sparse=None),
            np.matrix('0 1 1; 0 1 0'))
        np.testing.assert_almost_equal(
            oh._oh_df_to_mat(self.test_df, freq=True,
                             negate=False, sparse=None),
            np.matrix('0 1. 1.; 0 1. 0'))
        np.testing.assert_equal(
            oh._oh_df_to_mat(self.test_df, freq=False,
                             negate=True, sparse=None),
            np.matrix('1 0 0; 1 0 1'))
        # Types Checks
        self.assertIsInstance(oh._oh_df_to_mat(self.test_df, freq=False,
                                               negate=True, sparse=None),
                              np.ndarray)
        self.assertTrue(isspmatrix(oh._oh_df_to_mat(self.test_df, freq=False,
                                                    negate=False,
                                                    sparse=True)))
        self.assertTrue(isspmatrix(oh._oh_df_to_mat(self.test_df, freq=False,
                                                    negate=True,
                                                    sparse=False)))
        self.assertIsInstance(oh._oh_df_to_mat(self.test_df, freq=False,
                                               negate=True, sparse=True),
                              np.ndarray)
        self.assertIsInstance(oh._oh_df_to_mat(self.test_df, freq=False,
                                               negate=False, sparse=False),
                              np.ndarray)
        # dtypes checks
        self.assertEqual(oh._oh_df_to_mat(self.test_df, freq=False,
                                          negate=True, sparse=True).dtype.name,
                         'uint64')
        self.assertEqual(oh._oh_df_to_mat(self.test_df, freq=False,
                                          negate=True, sparse=True,
                                          precision=32).dtype.name,
                         'uint32')
        self.assertEqual(oh._oh_df_to_mat(self.test_df, freq=True,
                                          negate=True, sparse=True,
                                          precision=32).dtype.name,
                         'float32')

    def test_compute_cooccur_mat(self):
        # dense
        np.testing.assert_equal(
            oh.compute_cooccur_mat(self.test_df, freq=False, negate=False),
            np.matrix('0 0 0; 0 2 1; 0 1 1'))
        np.testing.assert_almost_equal(
            oh.compute_cooccur_mat(self.test_df, freq=True, negate=False),
            np.matrix('0 0 0; 0 1 .5; 0 .5 .5'))
        np.testing.assert_equal(
            oh.compute_cooccur_mat(self.test_df, freq=False, negate=True),
            np.matrix('2 0 1; 0 0 0; 1 0 1'))
        np.testing.assert_almost_equal(
            oh.compute_cooccur_mat(self.test_df, freq=True, negate=True),
            np.matrix('1 0 .5; 0 0 0;.5 0 .5'))
        # sparse
        np.testing.assert_equal(
            oh.compute_cooccur_mat(self.test_df, freq=False, negate=False,
                                   sparse=True),
            np.matrix('0 0 0; 0 2 1; 0 1 1'))
        np.testing.assert_almost_equal(
            oh.compute_cooccur_mat(self.test_df, freq=True, negate=False,
                                   sparse=True),
            np.matrix('0 0 0; 0 1 .5; 0 .5 .5'))
        np.testing.assert_equal(
            oh.compute_cooccur_mat(self.test_df, freq=False, negate=True,
                                   sparse=False),
            np.matrix('2 0 1; 0 0 0; 1 0 1'))
        np.testing.assert_almost_equal(
            oh.compute_cooccur_mat(self.test_df, freq=True, negate=True,
                                   sparse=False),
            np.matrix('1 0 .5; 0 0 0;.5 0 .5'))

    def test_compute_discordance_mat(self):
        # dense
        np.testing.assert_equal(
            oh.compute_discordance_mat(self.test_df, freq=False),
            np.matrix('0 0 0; 2 0 1; 1 0 0'))
        np.testing.assert_almost_equal(
            oh.compute_discordance_mat(self.test_df, freq=True),
            np.matrix('0 0 0; 1 0 .5; .5 0 0'))
        # sparse
        np.testing.assert_equal(
            oh.compute_discordance_mat(self.test_df, freq=False, sparse=True),
            np.matrix('0 0 0; 2 0 1; 1 0 0'))
        np.testing.assert_almost_equal(
            oh.compute_discordance_mat(self.test_df, freq=True, sparse=True),
            np.matrix('0 0 0; 1 0 .5; .5 0 0'))
        np.testing.assert_equal(
            oh.compute_discordance_mat(self.test_df, freq=False, sparse=False),
            np.matrix('0 0 0; 2 0 1; 1 0 0'))
        np.testing.assert_almost_equal(
            oh.compute_discordance_mat(self.test_df, freq=True, sparse=False),
            np.matrix('0 0 0; 1 0 .5; .5 0 0'))

    def test_norm_cooccur_mat(self):
        co_occur_mat = oh.compute_cooccur_mat(self.test_df, freq=True,
                                              extracols=None)
        print(co_occur_mat)
        np.testing.assert_almost_equal(
            oh.norm_cooccur_mat(co_occur_mat, correct_diag=True),
            np.array([[0, 0, 0],
                      [0, 1, 1],
                      [0, 1, 1]]))

    def test_norm_discordance_mat(self):
        co_occur_mat = oh.compute_cooccur_mat(self.test_df, freq=True,
                                              extracols=None)
        discordance_mat = oh.compute_discordance_mat(self.test_df, freq=True,
                                                     extracols=None)
        np.testing.assert_almost_equal(
            oh.norm_discordance_mat(discordance_mat, co_occur_mat),
            np.array([[0, 0, 0],
                      [1, 0, 1],
                      [1, 0, 0]]))

    def test_mutual_information_mat(self):
        np.testing.assert_almost_equal(
            oh.mutual_information_mat(self.test_df, extracols=None, logbase=2),
            np.matrix('0 0 0; 0 0 0; 0 0 1'))
        # test sparse
        np.testing.assert_almost_equal(
            oh.mutual_information_mat(self.test_df, extracols=None, logbase=2,
                                      sparse=True),
            np.matrix('0 0 0; 0 0 0; 0 0 1'))
        np.testing.assert_almost_equal(
            oh.mutual_information_mat(self.test_df, extracols=None, logbase=2,
                                      sparse=False),
            np.matrix('0 0 0; 0 0 0; 0 0 1'))
        # test precision
        np.testing.assert_almost_equal(
            oh.mutual_information_mat(self.test_df, extracols=None, logbase=2,
                                      sparse=None, precision=16),
            np.matrix('0 0 0; 0 0 0; 0 0 1'))
        np.testing.assert_almost_equal(
            oh.mutual_information_mat(self.test_df, extracols=None, logbase=2,
                                      sparse=True, precision=16),
            np.matrix('0 0 0; 0 0 0; 0 0 1'))

    def test_uncertainty_coef_mat(self):
        mi_mat = oh.mutual_information_mat(self.test_df,
                                           extracols=None, logbase=2)
        np.testing.assert_almost_equal(
            oh.uncertainty_coef_mat(mi_mat),
            np.array([[np.nan, 0, 0], [0, np.nan, 0], [0, 0, np.nan]]))
        mi_mat = np.array([[2, 1, 0], [1, 1, 0], [0, 0, 0]])
        np.testing.assert_almost_equal(
            oh.uncertainty_coef_mat(mi_mat),
            np.array([[np.nan, .5, 0], [1, np.nan, 0], [0, 0, np.nan]]))
