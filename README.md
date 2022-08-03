# Recovering individual emotional states from sparse ratings using collaborative filtering 

This repository contains all the code, data, and analyses supporting the associated manuscript. You may also be interested in the associated toolbox [Neighbors](https://neighbors.cosanlab.com).

## Analysis

1. Run `preprocess_decisions.py` and `preprocess_moth.py`
2. Run `fit_all_models.py` 3 times once for each dataset: `iaps`,`moth`, and`decisions`
3. Run `fit_mice.py` to run MICE across all 3 datasets
4. Examine analysis results in `Model_Comparisons.ipynb` notebook

## Data

All `.csv` files in this repo are handled by [Git LFS](https://git-lfs.github.com/) which needs to be installed before cloning/pulling this repo in order to populate these files with data. You can do this buy downloading `git lfs` from the link above and then typing the following command in a terminal: `git lfs install`. Afterwards, any `git clone` or `git pull` operations should automatically grab the large files from Github's large file storage. At anytime you can see what files are being managed by LFS using: `git lfs ls-files`. 

## Library functions

Additional helper functions used in various scripts are notebooks are located in `lib/lib.py`, and can be imported for use, e.g. `from lib import verify_results`

## Python Environment Setup

The `environment.yml` file in this repo can be used to bootstrap a conda environment for reproducibility:

`conda env create --name collab_filter --file environment.yml`

### Original setup

Project dependencies were managed using `conda` and `pip`. They have been exported using:

`conda env export > environment.yml`.

This environment was originally created using:

`conda create --name collab_filter -c ejolly -c defaults -c conda-forge python=3.8 "r::r-base" pymer4`.

Additional packages were installed manually using `conda install` or `pip install`.

### Parallelization with Ray

`fit_all_models.py` requires a working [Ray](https://ray.io/) installation

1. start a ray "head-node" instance on the multi-core machine you plan to use, e.g. after `ssh`-ing

   `ray start --head -v`

2. Now run `fit_all_models.py` following the documentation in the script
3. Examine live estimation time and log messages at Ray's web dashboard server `localhost:8125` (default). If you `ssh`'d with port-forwarding enable you should be able to see Ray's dashboard through your local machine's browser as well

## Development setup

Development work was performed in Visual Studio Code, which requires it's own configuration to work well with the conda environment. Once your environment is setup, you should appropriately adjust the relevant paths in `.vscode/settings.json` for your system.
