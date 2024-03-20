# /bin/#!/usr/bin/env bash -e
# conda-lock lock --help

echo 'Building env and lock files to use TF GPU'
conda-lock lock -f unvers_envs/tensorflow_gpu.yml -f unvers_envs/sci_compute.yml --virtual-package-spec virtual_packages.yml -p linux-64 --kind lock --mamba
mv conda-lock.yml tf_gpu.conda-lock.yml # must have the full conda-lock.yml extension
# see https://github.com/conda-incubator/conda-lock/issues/154
conda-lock render -k explicit -k env --filename-template "tf_gpu-{platform}.lock" tf_gpu.conda-lock.yml

echo 'Building env and lock files to use TF CPU'
conda-lock lock -f unvers_envs/tensorflow_cpu.yml -f unvers_envs/sci_compute.yml --virtual-package-spec virtual_packages.yml -p linux-64 --kind lock --mamba
mv conda-lock.yml tf_cpu.conda-lock.yml # must have the full conda-lock.yml extension
# see https://github.com/conda-incubator/conda-lock/issues/154
conda-lock render -k explicit -k env --filename-template "tf_cpu-{platform}.lock" tf_cpu.conda-lock.yml

echo 'Building data analysis environment'
conda-lock lock -f unvers_envs/db_analysis.yml -f unvers_envs/sci_compute.yml --virtual-package-spec virtual_packages.yml -p linux-64 --kind lock --mamba
mv conda-lock.yml db_analysis.conda-lock.yml # must have the full conda-lock.yml extension
conda-lock render -k explicit -k env --filename-template "db_analysis-{platform}.lock" db_analysis.conda-lock.yml
