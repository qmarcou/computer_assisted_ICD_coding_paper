import warnings
import keras_utils.callbacks
from tensorflow import errors as tferror
from keras_utils import errors as kuerror

# Get the correct target metric name in case the model has 2 outputs
model = hypermodel.build(hp)
output_L_str = '_'
if len(model.output_names) > 1:
    output_L_str += model.output_names[0] + "_"
target_metric_name = output_L_str + target_metric_props['name']

# Instantiate the tuner
objective_metric = kt.Objective("dev" + target_metric_name,
                                direction=target_metric_props['direction'])
tuner = kt.Hyperband(
    hypermodel,
    objective=objective_metric,
    directory=models_dir,
    project_name=project_name,
    **tuner_init_kwargs
)

# Instantiate a callback to compute the metrics on a 3rd dev dataset
callbacks_list = []
dev_eval_cb = keras_utils.callbacks.DataEvaluator(
    x=dev_oh_ingredients_df.to_numpy(),
    y=dev_oh_diagnoses_df.to_numpy(),
    eval_prefix="dev_",
    **evaluate_kwargs
)
callbacks_list.append(dev_eval_cb)
if target_metric_props['compound']:
    com_met = keras_utils.callbacks.CompoundMetric(
        metrics_dict=target_metric_props['metrics_dict'],
        met_prefixes=['', 'val_', 'dev_'],
    )
    callbacks_list.append(com_met)

old_t_state = tuner.oracle.get_state().get('tried_so_far')
try:
    if not os.path.isdir(tuner.project_dir + "/best_models/"):
        tuner.search(x=train_oh_ingredients_df.to_numpy(),
                     y=train_oh_diagnoses_df.to_numpy(),
                     validation_split=.05,
                     callbacks=callbacks_list+[
                         # Put other callbacks before tensorboard so that the
                         # updates can be logged
                         keras.callbacks.TensorBoard(
                             log_dir=tuner.project_dir + "/tb_logs",
                             write_graph=False,
                             profile_batch=False
                         )],
                     earlystop_mode="max",
                     earlystop_monitor="val" + target_metric_name,
                     verbose=False,
                     **tuner_search_kwargs
                     )
    else:
        warnings.warn("The best_models directory already exists, skipping "
                      "the tuner search. If this is not the intended behavior "
                      "delete the corresponding directory")
except (tferror.ResourceExhaustedError, tferror.InternalError) as e:
    # Check that at least 1 trial has been completed since last stop
    new_t_state = tuner.oracle.get_state().get('tried_so_far')
    if len(new_t_state) > max(len(old_t_state), 1):
        raise kuerror.KerasTunerSearchOOM(
            node_def=e.node_def,
            op=e.op,
            message=("OOM at trial "
                     + str(tuner.oracle.get_state().get('ongoing_trials'))
                     + ", sending exception to terminate the TF process. "
                       "Trials completed since last tuner.search launch: "
                     + str(len(new_t_state) - len(old_t_state) - 1)),
        )
    else:
        raise kuerror.KerasTunerSearchOOMSameTrial(
            node_def=e.node_def,
            op=e.op,
            message=(e._message + "Exception caught on a previously "
                                  "erroneous trial or first trial of the "
                                  "tuning process. "))
except Exception as e:
    raise e
