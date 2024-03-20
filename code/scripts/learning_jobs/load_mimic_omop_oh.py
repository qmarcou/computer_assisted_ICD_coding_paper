from proj_config import get_datadir
import pandas_utils as pu
import pandas as pd

print("Loading complete dataset...")

# get_modelsdir,

# models_dir = get_modelsdir() / "mimic-omop/sequentialNN/"
datadir = get_datadir() / "mimic-omop/pre-processed/"

split_index = (oom_cli_dir_suffix if "oom_cli_dir_suffix" in globals() else "")

train_visits = pd.read_csv(datadir / ('train_visits_id'+split_index+'.csv.gz')).squeeze()
dev_visits = pd.read_csv(datadir / ('dev_visits_id'+split_index+'.csv''.gz')).squeeze()
test_visits = pd.read_csv(datadir / ('test_visits_id'+split_index+'.csv.gz')).squeeze()

all_visits = pd.concat([train_visits, dev_visits, test_visits])

oh_diagnoses_df = (pd.read_parquet(datadir / "oh_diagnoses_rollup.parquet")
                   .set_index('visit_occurrence_id'))

oh_ingredients_df = (pd.read_parquet(datadir / "oh_ingredients.parquet")
                     .set_index('visit_occurrence_id'))

age_sex_df = pd.read_csv(datadir / "visits_agesex.csv.gz",
                         usecols=['visit_occurrence_id', 'age', 'sex_F'],
                         dtype={'age': 'float64',
                                'sex': 'uint8',
                                'visit_occurrence_id': 'uint64'},
                         index_col='visit_occurrence_id')

train_oh_diagnoses_df = oh_diagnoses_df.loc[train_visits, :]
train_oh_ingredients_df = oh_ingredients_df.loc[train_visits, :]

dev_oh_diagnoses_df = oh_diagnoses_df.loc[dev_visits, :]
dev_oh_ingredients_df = oh_ingredients_df.loc[dev_visits, :]

test_oh_diagnoses_df = oh_diagnoses_df.loc[test_visits, :]
test_oh_ingredients_df = oh_ingredients_df.loc[test_visits, :]

icd9_concepts = pd.read_csv(datadir / "icd9_concepts.csv.gz")
icd9_allcodes = icd9_concepts.concept_code
icd9_chapters = icd9_concepts.query('concept_class_id=="Chapter"').concept_code
icd9_subchapters = (icd9_concepts.query('concept_class_id=="SubChapter"')
                    .concept_code)
icd9_3dig = (icd9_concepts.query('concept_class_id.str.startswith("3-dig",'
                                 'na=False)')
             .concept_code)
icd9_4dig = (icd9_concepts.query('concept_class_id.str.startswith("4-dig",'
                                 'na=False)')
             .concept_code)
icd9_5dig = (icd9_concepts.query('concept_class_id.str.startswith("5-dig",'
                                 'na=False)')
             .concept_code)
icd9_billing = (
    icd9_concepts.query('concept_class_id.str.endswith("-dig billing code",'
                        'na=False)')
    .concept_code)
# Note this leaves out ~ 10 codes labeled as "ICD9CM code" but they
# seem erroneous
print("Dataset loading complete.")
