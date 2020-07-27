"""
TLSTM. Turing Learning system to generate trajectories
Copyright (C) 2018  Alessandro Zonta (a.zonta@vu.nl)

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
import glob
import logging
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.Fitness.ValueGraphFitness import convert

if __name__ == '__main__':
    logger = logging.getLogger("LoadTrajectories")
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    logger.info("Starting script")

    # get all the h5 files
    # find all the save files
    path_to_read = "/Users/alessandrozonta/PycharmProjects/NEAT/Output/"
    folders = glob.glob("{}/*/".format(path_to_read))

    # find different experiments
    exp_zero = "direction_normal"
    exp_one = "normal_direction_five"

    # different_fitness
    fit = ["point_distance_0_1_2_", "point_distance_0_1_", "point_distance_0_2_", "point_distance_1_2_",
           "point_distance_1_", "point_distance_2_", "point_distance_0_", "point_distance_non"]
    real_names = ["neat012", "neat01", "neat02", "neat12", "neat1", "neat2", "neat0", "neat"]
    rnm = dict(zip(fit, real_names))

    experiment_one = [el for el in folders if exp_zero in el]
    experiment_two = [el for el in folders if exp_one in el]

    max_to_normalise = 1600


    idx = 0
    gen_here = []
    value_here = []
    type_here = []
    repetition = []
    general_type = []

    for f in fit:
        repetitions = [el for el in experiment_two if f in el]

        for single_element in repetitions:
            try:
                content = pickle.load(open("{}/statistics.pkl".format(single_element), 'rb'))

                best_fitness = [c.fitness for c in content.most_fit_genomes]
                avg_fitness = np.array(content.get_fitness_mean())
                stdev_fitness = np.array(content.get_fitness_stdev())

                for i in range(len(content.most_fit_genomes)):
                    gen_here.append(i)
                    gen_here.append(i)
                    value_here.append(
                        convert(old_max=max_to_normalise, old_min=0, new_max=600, new_min=0, old_value=best_fitness[i]))
                    value_here.append(
                        convert(old_max=max_to_normalise, old_min=0, new_max=600, new_min=0, old_value=avg_fitness[i]))
                    type_here.append("best")
                    type_here.append("avg")
                    repetition.append(idx)
                    repetition.append(idx)
                    general_type.append(rnm[f])
                    general_type.append(rnm[f])
                idx += 1
            except Exception as e:
                logger.debug("file not available")

    data_here_this_exp = {"gen": gen_here, "values": value_here, "metric": type_here, "models": general_type}
    df = pd.DataFrame(data=data_here_this_exp)

    plt.figure(figsize=(20, 10))
    sns.set_context("talk")
    sns.set_style("darkgrid")
    g = sns.lineplot(x="gen", y="values", hue="models", style="metric", data=df)

    # plt.legend(loc='right', bbox_to_anchor=(1.24, 0.5))
    sns.despine(offset=10, trim=True)
    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.show()