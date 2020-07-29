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

    # get all the h5 files
    # find all the save files
    path_to_read = "/Users/alessandrozonta/PycharmProjects/NEAT/Output/"
    correct_files = glob.glob("{}/*_save".format(path_to_read))

    # find different experiments
    exp_zero = "direction_normal"
    exp_one = "normal_direction_five"

    # different_fitness
    fit = ["point_distance_0_1_2_", "point_distance_0_1_", "point_distance_0_2_", "point_distance_1_2_",
           "point_distance_1_", "point_distance_2_", "point_distance_0_", "point_distance_non"]
    real_names = ["neat012", "neat01", "neat02", "neat12", "neat1", "neat2", "neat0", "neat"]
    rnm = dict(zip(fit, real_names))

    experiment_one = [el for el in correct_files if exp_zero in el]
    experiment_two = [el for el in correct_files if exp_one in el]

    data_exp = []
    for f in fit:
        repetitions = [el for el in experiment_two if f in el]
        logger.debug(repetitions)
        for el in repetitions:
            experiment_two.remove(el)
        dataframes = [pd.read_hdf(el) for el in repetitions]
        single_dataframe = pd.concat(dataframes)
        source = [rnm[f] for _ in range(single_dataframe.shape[0])]
        single_dataframe['source'] = source
        data_exp.append(single_dataframe)
    df = pd.concat(data_exp)

    min_dataframe = abs(df["fitness"].min())
    df["fitness"] = (df["fitness"] + min_dataframe) / (700 + min_dataframe)
    df["direction"] = df["direction"] / 800
    df["no_overlapping"] = df["no_overlapping"] / 200

    small_dataset = df[["direction", "fitness", "no_overlapping", "source"]]
    small_dataset.columns = ['Directions', 'Fitness', "No_Overlapping", "Source"]

    reshaped = small_dataset.melt(id_vars=['Source'], value_vars=['No_Overlapping', 'Fitness', 'Directions'])
    reshaped.columns = ['Versions', 'Measurements', "Score"]
    sns.boxplot(x="Measurements", y="Score", hue="Versions", data=reshaped, showfliers = False)
    sns.despine(offset=10, trim=True)
    plt.savefig("{}/combined_graph_neat.pdf".format(path_to_read))
    # plt.show()
    plt.close()

    #
    # type_to_check = "fitness"
    # # for name in real_names:
    # #     sns.distplot(df[df["source"] == name][type_to_check], label=name, kde=False, rug=True)
    # sns.boxplot(x="source", y="fitness", data=df)
    # # plt.legend()
    # sns.despine(offset=10, trim=True)
    # # plt.show()
    # path_to_read = "/Users/alessandrozonta/PycharmProjects/NEAT/output/"
    # plt.savefig("{}/fitness_neat.pdf".format(path_to_read))
    # plt.close()
    #
    # ax = sns.boxplot(x="source", y="total_length", data=df, showfliers = False)
    # ax.set(xlabel='sources', ylabel='total length')
    # sns.despine(offset=10, trim=True)
    # # plt.show()
    # path_to_read = "/Users/alessandrozonta/PycharmProjects/NEAT/output/"
    # plt.savefig("{}/total_length_neat.pdf".format(path_to_read))
    # plt.close()


    for name in real_names:
        ax = sns.distplot(df[df["source"] == name]["total_length"], label=name, kde=False, rug=True)
    ax.set(xlabel='total length')
    plt.legend()
    sns.despine(offset=10, trim=True)
    plt.savefig("{}/total_length_neat.pdf".format(path_to_read))
    plt.close()

    # ax = sns.catplot(x="source", y="no_overlapping", data=df)
    # ax.set(xlabel='sources', ylabel='overlapping ratio')
    # sns.despine(offset=10, trim=True)
    # # plt.show()
    # # #
    # path_to_read = "/Users/alessandrozonta/PycharmProjects/NEAT/output/"
    # plt.savefig("{}/overlapping_neat.pdf".format(path_to_read))
    # plt.close()

    # ax = sns.boxplot(x="source", y="direction", data=df)
    # ax.set(xlabel='sources', ylabel='directions')
    # sns.despine(offset=10, trim=True)
    # path_to_read = "/Users/alessandrozonta/PycharmProjects/NEAT/output/"
    # plt.savefig("{}/directions_neat.pdf".format(path_to_read))
    # # plt.show()
    # plt.close()

    to_check = ["fitness", "no_overlapping", "direction"]

    for c in to_check:
        total = []
        total.append(stats.ks_2samp(df[df["source"] == "neat2"][c], df[df["source"] == "neat012"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat2"][c], df[df["source"] == "neat01"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat2"][c], df[df["source"] == "neat02"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat2"][c], df[df["source"] == "neat0"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat2"][c], df[df["source"] == "neat12"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat2"][c], df[df["source"] == "neat1"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat2"][c], df[df["source"] == "neat"][c]).pvalue)
        logger.info(total)
        logger.info(np.mean(np.array(total)))
        logger.info(np.std(np.array(total)))

        total = []
        total.append(stats.ks_2samp(df[df["source"] == "neat1"][c], df[df["source"] == "neat012"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat1"][c], df[df["source"] == "neat01"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat1"][c], df[df["source"] == "neat02"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat1"][c], df[df["source"] == "neat0"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat1"][c], df[df["source"] == "neat12"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat1"][c], df[df["source"] == "neat2"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat1"][c], df[df["source"] == "neat"][c]).pvalue)
        logger.info(total)
        logger.info(np.mean(np.array(total)))
        logger.info(np.std(np.array(total)))

        total = []
        total.append(stats.ks_2samp(df[df["source"] == "neat0"][c], df[df["source"] == "neat012"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat0"][c], df[df["source"] == "neat01"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat0"][c], df[df["source"] == "neat02"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat0"][c], df[df["source"] == "neat1"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat0"][c], df[df["source"] == "neat12"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat0"][c], df[df["source"] == "neat2"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat0"][c], df[df["source"] == "neat"][c]).pvalue)
        logger.info(total)
        logger.info(np.mean(np.array(total)))
        logger.info(np.std(np.array(total)))

        total = []
        total.append(
            stats.ks_2samp(df[df["source"] == "neat"][c], df[df["source"] == "neat012"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat"][c], df[df["source"] == "neat01"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat"][c], df[df["source"] == "neat02"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat"][c], df[df["source"] == "neat1"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat"][c], df[df["source"] == "neat12"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat"][c], df[df["source"] == "neat2"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat"][c], df[df["source"] == "neat0"][c]).pvalue)
        logger.info(total)
        logger.info(np.mean(np.array(total)))
        logger.info(np.std(np.array(total)))

        total = []
        total.append(
            stats.ks_2samp(df[df["source"] == "neat01"][c], df[df["source"] == "neat012"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat01"][c], df[df["source"] == "neat0"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat01"][c], df[df["source"] == "neat02"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat01"][c], df[df["source"] == "neat1"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat01"][c], df[df["source"] == "neat12"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat01"][c], df[df["source"] == "neat2"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat01"][c], df[df["source"] == "neat"][c]).pvalue)
        logger.info(total)
        logger.info(np.mean(np.array(total)))
        logger.info(np.std(np.array(total)))

        total = []
        total.append(
            stats.ks_2samp(df[df["source"] == "neat02"][c], df[df["source"] == "neat012"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat02"][c], df[df["source"] == "neat01"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat02"][c], df[df["source"] == "neat0"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat02"][c], df[df["source"] == "neat1"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat02"][c], df[df["source"] == "neat12"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat02"][c], df[df["source"] == "neat2"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat02"][c], df[df["source"] == "neat"][c]).pvalue)
        logger.info(total)
        logger.info(np.mean(np.array(total)))
        logger.info(np.std(np.array(total)))

        total = []
        total.append(
            stats.ks_2samp(df[df["source"] == "neat012"][c], df[df["source"] == "neat0"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat012"][c], df[df["source"] == "neat01"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat012"][c], df[df["source"] == "neat02"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat012"][c], df[df["source"] == "neat1"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat012"][c], df[df["source"] == "neat12"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat012"][c], df[df["source"] == "neat2"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat012"][c], df[df["source"] == "neat"][c]).pvalue)
        logger.info(total)
        logger.info(np.mean(np.array(total)))
        logger.info(np.std(np.array(total)))

        total = []
        total.append(stats.ks_2samp(df[df["source"] == "neat12"][c], df[df["source"] == "neat0"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat12"][c], df[df["source"] == "neat01"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat12"][c], df[df["source"] == "neat02"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat12"][c], df[df["source"] == "neat1"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat12"][c], df[df["source"] == "neat012"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat12"][c], df[df["source"] == "neat2"][c]).pvalue)
        total.append(stats.ks_2samp(df[df["source"] == "neat12"][c], df[df["source"] == "neat"][c]).pvalue)
        logger.info(total)
        logger.info(np.mean(np.array(total)))
        logger.info(np.std(np.array(total)))

        logger.info("------------------------")