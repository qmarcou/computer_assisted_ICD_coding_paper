# Backcompatibility issues
With time some breaking changes in conda-lock and mamba have impeded this procedure.
To make sure to get it all working you can install conda-lock version 1.2.1 and mamba version 0.24.0.
The constraint on conda-lock's version is easy to statisfy and now enforced in the lock environment file, you can thus blindly follow the instruction below.
The constraint on mamba is a bit more cubersome, and the easiest workaround is to install an old version of mambaforge. A version statisfying this requirement is [mambaforge 4.13.0-1](https://github.com/conda-forge/miniforge/releases/tag/4.13.0-1).

# Folder content

 This folder contains 3 types of environment files:
 - fully locked platform specific dependency files guaranteeing exactly reproducible environments
 - versioned dependency files mostly reproducible but creating an environment from them will trigger conda/mamba solver and might introduce slight changes to the resulting environments
 - un-versioned direct dependencies files enabling creation of environment to their latest version (`unvers_env_files` folder)

# Reproducible environment install   
Use of conda-lock: using the [pip enabled version of conda-lock](https://conda-incubator.github.io/conda-lock/getting_started/) which to this date can only be installed via pip. Simply create a dedicated environment using the provided environment file:

```
conda env create -f unvers_envs/lock_env.yml
```

Fully reproducible:
 - Install lock using :  `conda-lock install -n YOURENVNAME conda-lock.yml`

or
Almost fully reproducible:
 - Install lock using (see warning below): `conda env create --name YOURENV --file tf_cpu-linux-64.lock.yml`

WARNING: Using environment lock files (*.yml) does NOT guarantee that generated environments will be identical over time, since the dependency resolver is re-run every time and changes in repository metadata or resolver logic may cause variation. Conversely, since the resolver is run every time, the resulting packages ARE guaranteed to be seen by conda as being in a consistent state. This makes them useful when updating existing environments.

See conda-lock [GitHub](https://github.com/conda-incubator/conda-lock) or `conda-lock --help` for reference.

```
conda-lock install --name pmsi_db_analysis db_analysis.conda-lock.yml
conda-lock install --name pmsi_tf_cpu tf_cpu.conda-lock.yml
conda-lock install --name pmsi_tf_gpu tf_gpu.conda-lock.yml
```

# Environment update
```
conda update --file db_analysis-linux-64.lock --update-specs --prune -n pmsi_db_analysis
conda update --file tf_cpu-linux-64.lock --update-specs --prune -n pmsi_tf_cpu
conda update --file tf_gpu-linux-64.lock --update-specs --prune -n pmsi_tf_gpu
```

If LD_LIBRARY_PATH is set on your sytem and pointing to system libraries instead of the conda (e.g for libcudnn) you can set this environment variable for a specific conda environment e.g:
```
conda env config vars set LD_LIBRARY_PATH=/PATH/TO/CONDA/envs/pmsi_tf_gpu/lib -n pmsi_tf_gpu`
```
See the related docs [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables)
