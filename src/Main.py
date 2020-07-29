"""
TrajectoriesNEAT. Towards a human-like movements generator based on environmental features
Copyright (C) 2020  Alessandro Zonta (a.zonta@vu.nl)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging
import os
import random

import numpy as np

import tensorflow as tf
import mlflow

from src.Algorithms.neatAlgorithm import neatAlgorithm
from src.Fitness.GeneralFitness import PRE_DEFINED_BEHAVIOURS_ALL
from src.Fitness.ValueGraphFitness import _get_max_fitness_possible
from src.Settings.arguments import args

if __name__ == "__main__":
    p_distance = "point_distance_"
    if args.point_distance is None:
        p_distance = "point_distance_none"
    else:
        if 0 in args.point_distance:
            p_distance += "0_"
        if 1 in args.point_distance:
            p_distance += "1_"
        if 2 in args.point_distance:
            p_distance += "2_"

    end_output = "{}/{}_{}{}/".format(args.output_directory, args.name_experiment, p_distance,
                                      args.fitness_definition)
    if not os.path.isdir(end_output):
        os.mkdir(end_output)

    name_log = end_output + "/" + args.name_experiment

    # -------------- logger
    logger = logging.getLogger(name_log)
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    fh = logging.FileHandler(name_log + '.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    # -------------- logger

    logger.info(args)

    # -------------- mlflow
    mlflow.set_experiment(experiment_name=args.name_experiment)
    mlflow.start_run(run_name=args.run_name)
    # -------------- mlflow

    # seed for reproducibility
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    random.seed(a=args.seed)

    # -------------- mlflow store settings
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("numb_of_tra", args.numb_of_tra)
    mlflow.log_param("checkpoint", args.checkpoint)
    mlflow.log_param("fitness_definition", args.fitness_definition)
    mlflow.log_param("point_distance", args.point_distance)
    mlflow.log_param("random_point_start", args.random_point_start)

    max_fitness_possible = _get_max_fitness_possible(fitness_definition=args.fitness_definition)
    mlflow.log_param("max_fitness_possible", max_fitness_possible)
    logging.debug("Max fitness possible = {}".format(max_fitness_possible))

    if args.numb_of_tra > len(PRE_DEFINED_BEHAVIOURS_ALL):
        raise ValueError("Too many trajectories, not enough behaviours")

    a = neatAlgorithm(mlflow=mlflow, config_file=args.config_path, logger=logger, output=end_output,
                      fitness_definition=args.fitness_definition)
    check = None if args.checkpoint == "" else args.checkpoint
    a.initialise(frequency_checkpoints=args.freq_checkpoints, restore_checkpoint_name=check)

    a.run(generations=args.generations)

    logging.info("End program")

    mlflow.end_run()
