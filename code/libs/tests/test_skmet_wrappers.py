from unittest import TestCase
import skmet_wrappers
import numpy as np
import sklearn.metrics as skmet


class TestAUPRC(TestCase):
    y_true_almnull = np.zeros((15, 10))
    y_true_almnull[0, -1] = True
    y_pred_rd = np.random.uniform(size=y_true_almnull.shape)

    def test_per_sample_auc_PR(self):
        # Test that samples without positive label return NaN
        s_auprc = skmet_wrappers.per_sample_auc_PR(
            y_true=self.y_true_almnull,
            probas_pred=self.y_pred_rd,
            pos_label=1
        )
        np.testing.assert_array_equal(
            x=np.isnan(s_auprc),
            y=np.asarray([False]+[True]*(self.y_true_almnull.shape[0]-1))
        )

    def test_sample_averaged_auc_PR(self):
        # Test NaN handling policy in case of a label has no positive example
        np.testing.assert_array_equal(
            x=skmet_wrappers.sample_averaged_auc_PR(
                y_true=self.y_true_almnull,
                probas_pred=self.y_pred_rd,
                pos_label=1,
                nan_policy='nan'
            ),
            y=np.nan
        )
        np.testing.assert_array_almost_equal(
            x=skmet_wrappers.sample_averaged_auc_PR(
                y_true=self.y_true_almnull,
                probas_pred=self.y_true_almnull,
                pos_label=1,
                nan_policy='ignore'
            ),
            y=1.0
        )
        with self.assertRaises(RuntimeError):
            skmet_wrappers.sample_averaged_auc_PR(
                y_true=self.y_true_almnull,
                probas_pred=self.y_true_almnull,
                pos_label=1,
                nan_policy='error'
            )     

    def test_per_sample_average_precision(self):
        # Test that samples without positive label return NaN
        s_ap = skmet_wrappers.per_sample_average_precision(
            y_true=self.y_true_almnull,
            y_score=self.y_pred_rd,
            pos_label=1
        )
        np.testing.assert_array_equal(
            x=np.isnan(s_ap),
            y=np.asarray([False]+[True]*(self.y_true_almnull.shape[0]-1))
        )
        
    def test_per_code_auc_PR(self):
        # Test that samples without positive label return NaN
        label_auprc = skmet_wrappers.per_code_auc_PR(
            y_true=self.y_true_almnull,
            probas_pred=self.y_pred_rd,
            pos_label=1
        )
        np.testing.assert_array_equal(
            x=np.isnan(label_auprc),
            y=np.asarray([True]*(self.y_true_almnull.shape[1]-1)+[False])
        )

    def test_macro_averaged_auc_PR(self):
        # Test NaN handling policy in case of a label has no positive example
        np.testing.assert_array_equal(
            x=skmet_wrappers.macro_averaged_auc_PR(
                y_true=self.y_true_almnull,
                probas_pred=self.y_pred_rd,
                pos_label=1,
                nan_policy='nan'
            ),
            y=np.nan
        )
        np.testing.assert_array_almost_equal(
            x=skmet_wrappers.macro_averaged_auc_PR(
                y_true=self.y_true_almnull,
                probas_pred=self.y_true_almnull,
                pos_label=1,
                nan_policy='ignore'
            ),
            y=1.0
        )
        with self.assertRaises(RuntimeError):
            skmet_wrappers.macro_averaged_auc_PR(
                y_true=self.y_true_almnull,
                probas_pred=self.y_true_almnull,
                pos_label=1,
                nan_policy='error'
            )


class TestAUROC(TestCase):
    y_true_almnull = np.zeros((15, 10))
    y_true_almnull[0, -1] = True
    y_pred_rd = np.random.uniform(size=y_true_almnull.shape)

    def test_per_sample_AUROC(self):
        # Test that samples without positive label return NaN
        s_auroc = skmet_wrappers.per_sample_AUROC(
            y_true=self.y_true_almnull,
            y_score=self.y_pred_rd,
            pos_label=1
        )
        np.testing.assert_array_equal(
            x=np.isnan(s_auroc),
            y=np.asarray([False]+[True]*(self.y_true_almnull.shape[0]-1))
        )

    def test_sample_averaged_AUROC(self):
        # Compare the results with with sample averaging from sklearn
        y_true = np.zeros_like(self.y_true_almnull)
        y_true[0, 1:] = 1
        y_true[1:, 0] = 1
        self.assertAlmostEqual(
            skmet_wrappers.sample_averaged_AUROC(
                y_true=y_true,
                probas_pred=self.y_pred_rd,
                pos_label=1,
                nan_policy='error'
            ),
            skmet.roc_auc_score(
                y_true=y_true,
                y_score=self.y_pred_rd,
                average='samples'
            )
        )
        
        # Test NaN handling policy in case of a label has no positive example
        np.testing.assert_array_equal(
            x=skmet_wrappers.sample_averaged_AUROC(
                y_true=self.y_true_almnull,
                probas_pred=self.y_pred_rd,
                pos_label=1,
                nan_policy='nan'
            ),
            y=np.nan
        )
        np.testing.assert_array_almost_equal(
            x=skmet_wrappers.sample_averaged_AUROC(
                y_true=self.y_true_almnull,
                probas_pred=self.y_true_almnull,
                pos_label=1,
                nan_policy='ignore'
            ),
            y=1.0
        )
        with self.assertRaises(RuntimeError):
            skmet_wrappers.sample_averaged_AUROC(
                y_true=self.y_true_almnull,
                probas_pred=self.y_true_almnull,
                pos_label=1,
                nan_policy='error'
            )   

    def test_per_code_AUROC(self):
        # Test that samples without positive label return NaN
        label_auroc = skmet_wrappers.per_code_AUROC(
            y_true=self.y_true_almnull,
            probas_pred=self.y_pred_rd,
            pos_label=1
        )
        np.testing.assert_array_equal(
            x=np.isnan(label_auroc),
            y=np.asarray([True]*(self.y_true_almnull.shape[1]-1)+[False])
        )

    def test_macro_averaged_AUROC(self):
        # Compare the results with with macro averaging from sklearn
        y_true = np.random.binomial(n=1, p=.5, size=(100, 3))
        y_true[0, :] = 1  # Make sure at least 1 positive example per label
        y_true[-1, :] = 1  # Make sure at least 1 negative example per label
        y_pred = np.random.uniform(size=y_true.shape)
        self.assertAlmostEqual(
            skmet_wrappers.macro_averaged_AUROC(
                y_true=y_true,
                probas_pred=y_pred,
                pos_label=1,
                nan_policy='error'
            ),
            skmet.roc_auc_score(
                y_true=y_true,
                y_score=y_pred,
                average='macro'
            )
        )
        # Test NaN handling policy in case of a label has no positive example
        np.testing.assert_array_equal(
            x=skmet_wrappers.macro_averaged_AUROC(
                y_true=self.y_true_almnull,
                probas_pred=self.y_pred_rd,
                pos_label=1,
                nan_policy='nan'
            ),
            y=np.nan
        )
        np.testing.assert_array_almost_equal(
            x=skmet_wrappers.macro_averaged_AUROC(
                y_true=self.y_true_almnull,
                probas_pred=self.y_true_almnull,
                pos_label=1,
                nan_policy='ignore'
            ),
            y=1.0
        )
        