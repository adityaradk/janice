# Dataset Preprocessing

Within this folder are files for processing the dataset prior to using it to train/test/validate the models.

Most of these folders are intentionally left empty as GitHub has a file size restriction of 25 MB.

## Setup

To set up your enviroment, download the `exoTrain` and `exoTest` CSV files from WΔ's [Kaggle post](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data/) (~278 MB) and place the files in the `raw` folder.

Then, proceed to run the scripts in the following order:
1. `to_npy.py`
2. `normalize.py`
3. `partition.py`
4. `oversample.py` (with `--ost` as either `smote` or `adasyn`)  or `augment.py`

Use the `-h` tag for more information.

To enable _perfect_ replication of our results, we've also included our normalized validation set in the `partitioned` folder. If the scripts are rerun, a new randomly chosen validation set will be generated.
