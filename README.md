# About this
this a collection of my ML efforts on perovskites. The idea is to eventually change this in a way that it works without everythin being integrated in my repo.

# Dependencies
pull-requests welcome. This uses a varying number of the usual suspects (`scipy`, `numpy`, `pandas`) and `pytorch` and `pytorch-geometric`. In addition, it probably will depend on `dscribe` and our very own `sne_fingerprints`-library.

# Usage
Add `data_tools` (for data-exchange-objects), `sne_ml_databases` (for loading databases from a folder (see `/mnt/projects/sne_guests/perovskite_db` for exemplary databases.) and `torch_tools` as modulefolder to the pythonpath. I.e. in Python 3 just add the folder of this repo.

See torch_examples for some example code to build surrogate models. Set the environment variable `PEROVSKITE_DBDIR` to point to a folder having the data from the aforementioned location and you should be able to run the things in `basic_gnn_7benchmarks` (modify the script, it's reproducing the stuff from the global property prediction database).

# Contributing
Please only contribute to the aforementioned folders... It's a mess...