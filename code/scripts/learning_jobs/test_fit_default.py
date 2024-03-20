import gc

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

model_size = 'min'
if model_size == 'min':
    hp.values.update(
        {
            "batchnorm": False,
            "units": hp_values['units']["min_value"],
            "num_layers": hp_values["num_layers"]["min_value"],
            "dropout": .5
        }
    )
elif model_size == 'max':
    hp.values.update(
        {
            "batchnorm": True,
            "units": hp_values['units']["max_value"],
            "num_layers": hp_values["num_layers"]["max_value"],
            "dropout": .5
        }
    )
else:
    pass

run_gpu = len(tf.config.list_logical_devices('GPU')) > 0 or \
          len(tf.config.list_physical_devices('GPU')) > 0

for i in range(0, 1):
    if run_gpu:
        tf.config.experimental.reset_memory_stats('GPU:0')
    model = hypermodel.build(hp)

    #model.run_eagerly = True
    hypermodel.fit(
        hp, model,
        epochs=1,
        x=train_oh_ingredients_df.to_numpy(),
        y=train_oh_diagnoses_df.to_numpy(),
        validation_split=.05,
        earlystop_mode="max",
        earlystop_monitor="val_" + target_metric_props['name'],
        callbacks=callbacks_list+[
             # Put other callbacks before tensorboard so that the
             # updates can be logged
            keras.callbacks.TensorBoard(
                log_dir="/tmp/tb_logs",
                write_graph=False,
                profile_batch=[10, 30]
            )],
        verbose=True,
    )
    if run_gpu:
        print(tf.config.experimental.get_memory_info('GPU:0')["peak"])
