"""
Run the best TF model on the complete dataset for different batch
and record runtime for each on the complete dataset.
"""
import timeit
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os
# print working directory
print(os.getcwd())
sys.path.insert(0, str(Path(os.path.abspath('')).parent))
from proj_config import get_datadir, get_modelsdir, get_project_root


# Define paths
datadir = get_datadir() / "mimic-omop/pre-processed/"
model_dir = (get_modelsdir() /
             "mimic-omop/sequentialNN_3/overall_NDCG/drugSexAge_AllCodes" /
             "best_models/1/model")
tables_dir = get_project_root()/"docs/latex/overleaf/tables/"

# Define compute policy
keras.mixed_precision.set_global_policy('float32')

batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# batch_sizes = [16, 32]
repeats = 3

# Read the complete dataset
oh_ingredients_df = (pd.read_parquet(datadir / "oh_ingredients.parquet")
                     .set_index('visit_occurrence_id'))

age_sex_df = pd.read_csv(datadir / "visits_agesex.csv.gz",
                         usecols=['visit_occurrence_id', 'age', 'sex_F'],
                         dtype={'age': 'float64',
                                'sex': 'uint8',
                                'visit_occurrence_id': 'uint64'},
                         index_col='visit_occurrence_id')

oh_saingredients_df = pd.merge(age_sex_df, oh_ingredients_df,
                               left_index=True,
                               right_index=True,
                               how='right')
oh_saingredients_df = tf.constant(oh_saingredients_df.to_numpy())
model = keras.models.load_model(
    model_dir,
    compile=False  # Needed since metrics serializers are missing
)
model.compile()
model.summary()

# oh_saingredients_df = oh_saingredients_df[0:256, :]

runtimes_df = pd.DataFrame()
for batch_size in batch_sizes:
    # Run and time prediction
    def run_predictions():
        model.predict(
            x=oh_saingredients_df,
            batch_size=batch_size,
        )
    # Pre run the model to remove graph compilation steps from evaluation
    try:
        model.predict(
            x=oh_saingredients_df[0:batch_size, :],
            batch_size=batch_size,
        )
        times = timeit.repeat('run_predictions()', repeat=repeats,
                              number=1, globals=globals())
        runtimes_df = pd.concat([runtimes_df,
                                pd.DataFrame({'Batch size': batch_size,
                                              'Runtime': np.mean(times),
                                              'Std': np.std(times),
                                              'Repeats': repeats,
                                              'sample_size':
                                                  oh_saingredients_df.shape[0]
                                              },
                                             index=[0])],
                                ignore_index=True)
    except Exception as e:
        print(e)
        print(f"Stopping at batch size = {batch_size}")
        break

    runtimes_df.to_csv(tables_dir / "runtimes.csv", index=False)
