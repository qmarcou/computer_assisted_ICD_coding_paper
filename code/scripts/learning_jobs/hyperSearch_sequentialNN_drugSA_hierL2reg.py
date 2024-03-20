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

hp = kt.HyperParameters()

# Read ICD9 concept hierarchy
icd9_relationships = pd.read_csv(datadir / "icd9_concept_relationships.csv.gz",
                                 usecols=['concept_id_1', 'concept_id_2',
                                          'relationship_id'])

# Cast concept ids to ICD codes
for con_id in ['concept_id_1', 'concept_id_2']:
    icd9_relationships = (pd.merge(icd9_relationships,
                                   icd9_concepts.loc[:,
                                   ['concept_id', "concept_code"]],
                                   left_on=con_id,
                                   right_on="concept_id",
                                   how="left")
                          .drop(columns=["concept_id", con_id])
                          .rename(columns={"concept_code": con_id}))

# Build parent->child DAG adjacency matrix
icd9_adj_mat = concepts_toolbox.relationships_to_adjacency_df(
    icd9_relationships,
    relationship="Subsumes",
    unique_concepts=list(train_oh_diagnoses_df.columns)
).sparse.to_dense().to_numpy()

# Add hp entry for the output Hier Dense L2 Regularized layer
hp_values['outputHierL2Reg'] = {'enable': True,
                                'name': "outhier_lambda",
                                'min_value': 1e-15,
                                'max_value': 5e-6,
                                #'step': 5,
                                'sampling': "log",
                                'default': 1e-8,
                                'adjacency_matrix': icd9_adj_mat,
                                'activation': 'sigmoid'
                                }

hypermodel = keras_utils.models.SequentialMultilabelHypermodel(
    output_size=train_oh_diagnoses_df.shape[1],
    hp_kwargs=hp_values,
    metrics=learning_metrics,
    input_size=train_oh_ingredients_df.shape[1],
    detached_loss=False)


# exec(open("test_fit_default.py").read())

project_suffix = "_Sagi" if filter_Sagi else "_AllCodes"
project_name = "drugSexAge_HierL2" + project_suffix + (
    oom_cli_dir_suffix if "oom_cli_dir_suffix" in globals() else "")

exec(open("perform_hyperband_search.py").read())

exec(open("evaluate_save_best_models.py").read())

