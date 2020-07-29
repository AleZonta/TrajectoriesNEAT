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
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import stats

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

    data_exp = []
    # get all the h5 files
    # find all the save files
    path_to_read = "/Users/alessandrozonta/PycharmProjects/NEAT/Output/"
    correct_files = glob.glob("{}/*_save".format(path_to_read))

    # find different experiments
    exp_one = "normal_direction_five"

    # different_fitness
    fit = ["point_distance_0_2_"]
    real_names = ["neat02"]
    experiment_two = [el for el in correct_files if exp_one in el]
    repetitions_neat = [el for el in experiment_two if fit[0] in el]
    dataframes_neat = [pd.read_hdf(el) for el in repetitions_neat]
    single_dataframe = pd.concat(dataframes_neat)
    source = [real_names[0] for _ in range(single_dataframe.shape[0])]
    single_dataframe['source'] = source
    data_exp.append(single_dataframe)

    # get all the h5 files
    # find all the save files
    path_to_read = "/Users/alessandrozonta/Desktop/output_random_walk/"
    correct_files = glob.glob("{}/*_save".format(path_to_read))

    # different_fitness
    fit = ["fitness_no_visited_seed_standard_", "fitness_seed_standard_"]
    real_names = ["RWFBNV", "RWFB"]
    rw = dict(zip(fit, real_names))
    for f in fit:
        repetitions = [el for el in correct_files if f in el]
        logger.debug("{} - {}".format(f, len(repetitions)))
        for el in repetitions:
            correct_files.remove(el)
        dataframes = [pd.read_hdf(el) for el in repetitions]
        single_dataframe = pd.concat(dataframes)
        source = [rw[f] for _ in range(single_dataframe.shape[0])]
        single_dataframe['source'] = source
        data_exp.append(single_dataframe)

    path_to_read = "/Users/alessandrozonta/PycharmProjects/astar/output/"
    astar_attraction = pd.read_hdf("{}/test_normal_data".format(path_to_read))
    list_to_add_astar_attraction = ["aa" for _ in range(astar_attraction.shape[0])]
    astar_attraction['source'] = list_to_add_astar_attraction
    data_exp.append(astar_attraction)

    df = pd.concat(data_exp)
    min_dataframe = abs(df["fitness"].min())
    df["fitness"] = (df["fitness"] + min_dataframe) / (700 + min_dataframe)
    df["direction"] = df["direction"] / 800
    df["no_overlapping"] = df["no_overlapping"] / 200

    small_dataset = df[["direction", "fitness", "no_overlapping", "source"]]
    small_dataset.columns = ['Directions', 'Fitness', "No_Overlapping", "Source"]

    reshaped = small_dataset.melt(id_vars=['Source'], value_vars=['No_Overlapping', 'Fitness', 'Directions'])
    reshaped.columns = ['Versions', 'Measurements', "Score"]
    sns.boxplot(x="Measurements", y="Score", hue="Versions", data=reshaped, showfliers=False)
    sns.despine(offset=10, trim=True)
    # plt.savefig("combined_final_evaluation.pdf")
    plt.show()
    plt.close()

    # real_names = ["RWFBNV", "RWFB", "neat02", "aa"]
    # for name in real_names:
    #     ax = sns.distplot(df[df["source"] == name]["total_length"], label=name, kde=False, rug=True)
    # ax.set(xlabel='total length')
    # plt.legend()
    # sns.despine(offset=10, trim=True)
    # plt.savefig("{}/total_length_neat.pdf".format(path_to_read))
    # plt.show()
    # plt.close()

    to_check = ["fitness", "no_overlapping", "direction"]

    for c in to_check:
        total = []
        total.append(stats.ks_2samp(df[df["source"] == "neat02"][c], df[df["source"] == "aa"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat02"][c], df[df["source"] == "RWFB"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat02"][c], df[df["source"] == "RWFBNV"][c]).pvalue)
        logger.info(total)
        logger.info(np.mean(np.array(total)))
        logger.info(np.std(np.array(total)))

        total = []
        total.append(stats.ks_2samp(df[df["source"] == "aa"][c], df[df["source"] == "RWFB"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "aa"][c], df[df["source"] == "RWFBNV"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "aa"][c], df[df["source"] == "neat02"][c]).pvalue)
        logger.info(total)
        logger.info(np.mean(np.array(total)))
        logger.info(np.std(np.array(total)))

        total = []
        total.append(stats.ks_2samp(df[df["source"] == "RWFB"][c], df[df["source"] == "RWFBNV"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "RWFB"][c], df[df["source"] == "neat02"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "RWFB"][c], df[df["source"] == "aa"][c]).pvalue)
        logger.info(total)
        logger.info(np.mean(np.array(total)))
        logger.info(np.std(np.array(total)))

        total = []
        total.append(stats.ks_2samp(df[df["source"] == "RWFBNV"][c], df[df["source"] == "RWFB"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "RWFBNV"][c], df[df["source"] == "neat02"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "RWFBNV"][c], df[df["source"] == "aa"][c]).pvalue)
        logger.info(total)
        logger.info(np.mean(np.array(total)))
        logger.info(np.std(np.array(total)))
        logger.info("------------------------")