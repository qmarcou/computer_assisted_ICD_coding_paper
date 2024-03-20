This repository is a companion to the article *[Creating a computer assisted ICD coding system: Performance metric choice and use of the ICD hierarchy](https://doi.org/10.1016/j.jbi.2024.104617)* published in the Journal of Biomedical Informatics in March 2024.

# In brief

This repository contains python/Tensorflow code to reproduce the results from the aforementionned article.
Due to licensing and privacy concerns I cannot release the actual data inside the repository, nor individual model predictions.
The data used are however freely available (see details below) to any researcher. For reproducibility I provide stay IDs for the different cross-validation splits, as these IDs do not convey any personal information.


The python code allows to preprocess the MIMIC-III-OMOP data to enable training on medication data using RxNorm ingredients.
The [keras_extra](https://github.com/qmarcou/keras_extra) companion package implements various custom Tensorflow objects in particular the RE@R metric, hierarchical multilabel learning and label imbalance correction techniques. 

# Repository structure:

```
project
│
└───code  # contains all the python code
│   │   
│   └─── conda_env_files # contains files necessary to generate
│   │                    # reproducible conda environments via conda-lock 
│   └─── libs # contains all the custom python modules for the project and 
│   │     │   # the associated tests
│   │     └─── keras_extra # custom tensorflow/keras extensions 
│   └─── profiling # few profiling tests
│   │
│   └─── scripts # contains the jupyter notebooks and learning scripts
│   
└─── data # contains mostly placeholders to ensure directory structure, and some
|             scripts for postgresql database extraction.
│   
│   
└─── models # directory structure to host models created by
            # the learning jobs
```

# Data

The article relies on the MIMIC-III v1.4 dataset that can be accessed on [Physionet](https://physionet.org/content/mimiciii/1.4/), and mounted as a PostgreSQL database. The data has then been mapped to the OMOP-CDM using [ETL scripts from Paris et al](https://github.com/MIT-LCP/mimic-omop). The resulting tables of interest were [dumped to csv](data/mimic-omop/dump_db.sh).

To further map the resulting OMOP data to RxNorm ingredients I have downloaded ontology data from the default OHDSI Athena vocabularies v5.0 06-DEC-21. As chapter and subchapter information were missing from the vocabulary, they were added using R scripts contained in this repository.

# Python Scripts

I provide fully reproducible python environments for both CPU and GPU. They were generated via `conda-lock` in the `code/conda_env_files` directory (see the corresponding [README](code/conda_env_files/README.md)). 

Data preprocessing and checks is performed via the [mimic-omop_checks_and_preproc](code/scripts/mimic-omop_checks_and_preproc.ipynb) and [mimic-omop_hot_encoding](code/scripts/mimic-omop_hot_encoding.ipynb) jupyter notebooks (to be run in that order). 

Neural networks hyper parameter tuning and training scripts are contained in the [learning_scripts](code/scripts/learning_jobs) folder ( `hyperSearch_sequentialNN_*.py` files). The scripts should be run with that folder as working directory.

Performance analysis and more generally code to produce the figures and tables of the article is contained in the [mimic-omop_sequentialNN](code/scripts/mimic-omop_sequentialNN.ipynb) notebook.
