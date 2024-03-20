exec(open("syspath_setup.py").read())
import pandas as pd
import concepts_toolbox
import keras_tuner as kt
import keras_utils
from tensorflow import keras
import tensorflow as tf
from proj_config import get_modelsdir
import subprocess
import json
from keras.callbacks import TensorBoard

exec(open("load_mimic_omop_oh.py").read())

filter_Sagi = True
exec(open("filter_Sagi_codes.py").read())

exec(open("join_sexage_ingredients.py").read())

exec(open("read_hypersearch_parms.py").read())

exec(open("instantiate_metrics.py").read())

hp_values['output_kernel_regularizer'] = {'enable': True,
                                          'name': "out_L2",
                                          'min_value': 1e-14,
                                          'max_value': 1e-4,
                                          'sampling': "log",
                                          'default': 1e-8,
                                          }

hp = kt.HyperParameters()
hypermodel = keras_utils.models.SequentialMultilabelHypermodel(
    output_size=train_oh_diagnoses_df.shape[1],
    hp_kwargs=hp_values,
    metrics=learning_metrics,
    input_size=train_oh_ingredients_df.shape[1]
)

# exec(open("test_fit_default.py").read())

project_suffix = "_Sagi" if filter_Sagi else "_AllCodes"
project_name = "drugSexAge_outL2reg" + project_suffix + (
    oom_cli_dir_suffix if "oom_cli_dir_suffix" in globals() else "")

exec(open("perform_hyperband_search.py").read())

exec(open("evaluate_save_best_models.py").read())
