# Trajectories-NEAT. Towards a human-like movements generator based on environmental features

Trajectories-NEAT is a Python library implementing the NEAT experiment present in the paper "Towards a human-like movements generator based on environmental features".

## Requirements
 Tested only with `Python 3.6`
* `numpy==1.17.4`
* `haversine==2.1.2`
* `joblib==0.14.1`
* `matplotlib==3.1.1`
* `mlflow==1.5.0`
* `neat-python==0.92`
* `pandas==0.25.3`
* `scikit-learn==0.22`
* `scipy==1.3.1`
* `shapely==1.6.4`
* `tables==3.6.1`
* `tqdm==4.41.0`
* `pyarrow==0.15.1`

## Data
Download the dataset used from [here](https://doi.org/10.5281/zenodo.3964449) and place it in data.
Update path on the setting file accordingly

## Run the experiments
* check if all the path are corrects in the settings of the various scripts
* to generate new trajectories run src/Main.py
* use `--fitness_definition` to select which kind of fitness you want to use. To obtain the result shown in the paper, use "normal_direction_five"
* use `--numb_of_tra` to define how many trajectories to generate. By default the system support multiprocessing
* use `--point_distancel` if you want to modify the behaviour of the fitness based A*. Check `args.py` to see what are the selection for this argument

