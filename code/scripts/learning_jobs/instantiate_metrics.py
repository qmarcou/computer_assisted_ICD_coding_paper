import keras_utils
from tensorflow import keras
import tensorflow as tf
import numpy as np

# Create subset metrics
learning_metrics = []
eval_only_metrics = []
for metric, met_name in zip([keras_utils.metrics.Coverage,
                             keras_utils.metrics.MicroF1score,
                             keras_utils.metrics.MacroF1score,
                             keras.metrics.Precision,
                             keras.metrics.Recall,
                             keras.metrics.AUC,
                             keras_utils.metrics.NDCGMetric,
                             keras_utils.metrics.RankErrorsAtPercentile],
                            ['coverage', 'micro_F1', 'macro_F1',
                             "precision", "recall", 'aucPR', 'NDCG',
                             "rankErrors@"]):
    wrapper_metric = keras_utils.metrics.subset_metric_builder(metric)
    for code_subset, name in zip([icd9_chapters, icd9_subchapters, icd9_3dig,
                                  icd9_billing],
                                 ['ICD9_chapter', 'ICD9_subchapter',
                                  'ICD9_3dig', 'ICD9_billing']):
        if not code_subset.empty:
            code_subset_indices = np.where(train_oh_diagnoses_df
                                           .columns.isin(code_subset))[0] \
                .astype(dtype='int64')
            if met_name.endswith('F1'):
                learning_metrics.append(wrapper_metric(slicing_func=tf.gather,
                                                       slicing_func_kwargs={
                                                           'indices': code_subset_indices,
                                                           'axis': 1},
                                                       name=name + '_' + met_name,
                                                       num_classes=len(
                                                           code_subset_indices),
                                                       threshold=.5
                                                       ))
            elif met_name == "rankErrors@":
                for percentile in [25.0, 50.0, 75.0, 90.0]:
                    eval_only_metrics.append(wrapper_metric(
                        slicing_func=tf.gather,
                        slicing_func_kwargs={
                            'indices': code_subset_indices,
                            'axis': 1},
                        name=name + '_' + met_name + str(percentile) + "thP",
                        q=percentile,
                        no_true_label_value=1.0,
                        interpolation='linear'
                    ))
            elif met_name == "aucPR":
                for macro in [True, False]:
                    if macro:
                        aucmetric_name = name + '_macro_' + met_name
                    else:
                        aucmetric_name = name + '_micro_' + met_name
                    eval_only_metrics.append(wrapper_metric(slicing_func=tf.gather,
                                                            slicing_func_kwargs={
                                                                'indices': code_subset_indices,
                                                                'axis': 1},
                                                            name=aucmetric_name,
                                                            multi_label=macro,
                                                            num_thresholds=200,
                                                            curve='PR'
                                                            ))
            elif met_name == "NDCG":
                eval_only_metrics.append(
                    wrapper_metric(slicing_func=tf.gather,
                                   slicing_func_kwargs={
                                       'indices': code_subset_indices,
                                       'axis': 1},
                                   name=name + '_' + met_name,
                                   topn=None,
                                   gain_fn=keras_utils.metrics.identity,
                                   rank_discount_fn=keras_utils.metrics.inverse
                                   ))
            else:
                eval_only_metrics.append(wrapper_metric(slicing_func=tf.gather,
                                                        slicing_func_kwargs={
                                                            'indices': code_subset_indices,
                                                            'axis': 1},
                                                        name=name + '_' + met_name
                                                        ))
        # else:
        #     print(name + "code level empty, skipped metrics construction.")

learning_metrics.append(keras.metrics.AUC(curve='PR',
                                          num_thresholds=20000 if target_metric_props["name"]=='overall_micro_aucPR' else 200,
                                          multi_label=False,
                                          name='overall_micro_aucPR'))
learning_metrics.append(keras.metrics.AUC(curve='PR',
                                          num_thresholds=5000 if target_metric_props["name"]=='overall_macro_aucPR' else 200,
                                          multi_label=True,
                                          name='overall_macro_aucPR'))
learning_metrics.append(keras.metrics.AUC(curve='ROC',
                                          num_thresholds=20000 if target_metric_props["name"]=='overall_micro_AUROC' else 200,
                                          multi_label=False,
                                          name='overall_micro_AUROC'))
learning_metrics.append(keras.metrics.AUC(curve='ROC',
                                          num_thresholds=5000 if target_metric_props["name"]=='overall_macro_AUROC' else 200,
                                          multi_label=True,
                                          name='overall_macro_AUROC'))

learning_metrics.append(keras.metrics.Precision(name="overall_precision"))
learning_metrics.append(keras.metrics.Recall(name="overall_recall"))
learning_metrics.append(keras_utils.metrics.MacroF1score(
    name="overall_macro_F1",
    num_classes=len(icd9_allcodes),
    threshold=.5
))
learning_metrics.append(keras_utils.metrics.MicroF1score(
    name="overall_micro_F1",
    num_classes=len(icd9_allcodes),
    threshold=.5
))

learning_metrics.append(keras_utils.metrics.NDCGMetric(
    name="overall_NDCG",
    topn=None,))

learning_metrics.append(keras_utils.metrics.NDCGMetric(
    name="overall_hard_NDCG",
    topn=None,
    gain_fn=keras_utils.metrics.identity,
    rank_discount_fn=keras_utils.metrics.inverse))

evaluation_metrics = learning_metrics + eval_only_metrics


# HOTFIX: monkey patch this function to prevent tensorflow from prefixing
# multiple times the output layer name
def _set_metric_names(self):
    """Sets unique metric names."""
    # For multi-output models, prepend the output name to the metric name.
    # For weighted metrics, prepend "weighted_" if the name would be non-unique.
    # pylint: disable=protected-access
    metric_names = set()
    is_multi_output = len(self._output_names) > 1
    zip_args = (self._output_names, self._metrics, self._weighted_metrics)
    for output_name, output_metrics, weighted_output_metrics in zip(*zip_args):
        for m in output_metrics:
            if m is None:
                continue
            if is_multi_output:
                if not m._name.startswith(output_name + '_'):
                    m._name = output_name + '_' + m._name
            if m._name in metric_names:
                raise ValueError(
                    f'Found two metrics with the same name: {m._name}.'
                    'All the metrics added to the model need to have unique names.')
            metric_names.add(m._name)

        for wm in weighted_output_metrics:
            if wm is None:
                continue
            if is_multi_output:
                if output_name + '_' + wm._name in metric_names:
                    wm._name = output_name + '_weighted_' + wm._name
                else:
                    wm._name = output_name + '_' + wm._name
            elif wm._name in metric_names:
                wm._name = 'weighted_' + wm._name

            if wm._name in metric_names:
                raise ValueError(
                    f'Found two weighted metrics with the same name: {wm._name}.'
                    'All the metrics added to the model need to have unique names.')
            metric_names.add(wm._name)
    # pylint: enable=protected-access


import keras.engine.compile_utils as keras_compile

keras_compile.MetricsContainer._set_metric_names = _set_metric_names
