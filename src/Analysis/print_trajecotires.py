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
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.Helpers.APF import LoadAPF
from src.Helpers.Funcs import sorted_nicely

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

    apf = True

    loader_apf = LoadAPF(path="/Users/alessandrozonta/PycharmProjects/astar/data/the_right_one_fast", logger=logger)
    loader_apf.load_apf_only_routing_system()


    path_to_read = "/Users/alessandrozonta/Desktop"
    folders = sorted_nicely(glob.glob("{}*/".format(path_to_read)))
    folders = ["/Users/alessandrozonta/Desktop/seed_experiments_v9_point_distance_0_1_direction_normal"]
    for f in folders:
        name_folder = f.split("/")[-1]
        path = "{}/{}".format(path_to_read, name_folder)
        correct_files = glob.glob("{}/test_multipliers_attraction*".format(path))

        path_here = "{}/trajectories_pics/".format(path)
        os.makedirs(path_here, exist_ok=True)

        # need to store all the data from the trajectories
        for el in correct_files:
            content = pickle.load(open(el, 'rb'))
            # content[0] contains setting tested
            # content[1] contains data from the trajectories
            all_trajectories = content[1][2]
            tra_idx = 0
            for tra in all_trajectories:
                x = []
                y = []
                combinations = {}
                for points in tra:
                    x.append(int(points.x))
                    y.append(int(points.y))
                    combinations["{}-{}".format(int(points.x), int(points.y))] = 1
                max_x = max(x) + 1
                min_x = min(x) - 1
                max_y = max(y) + 1
                min_y = min(y) - 1

                max_value_x = ((max_x - min_x) + 10) * 2
                max_value_y = ((max_y - min_y) + 10) * 2

                matrix = np.zeros((max_value_x, max_value_y))

                if apf:
                    difference_to_move_x = int((max_value_x - (max_x - min_x)) / 2)
                    difference_to_move_y = int((max_value_y - (max_y - min_y)) / 2)
                    real_min_x = min_x - difference_to_move_x
                    real_max_x = max_x + difference_to_move_x
                    real_min_y = min_y - difference_to_move_y
                    real_max_y = max_y + difference_to_move_y

                    for j in range(real_min_x, real_max_x):
                        for q in range(real_min_y, real_max_y):
                            value = loader_apf.apf.iloc[j, q]
                            matrix_pos_x = j - real_min_x
                            matrix_pos_y = q - real_min_y
                            if "{}-{}".format(j, q) in combinations:
                                matrix[matrix_pos_x, matrix_pos_y] = 2.0
                            elif value != 0:
                                matrix[matrix_pos_x, matrix_pos_y] = 1.0
                else:
                    mdlx = int(max_value_x / 2)
                    mdly = int(max_value_y / 2)
                    for i in range(1, len(x)):
                        diffx = x[i] - x[i - 1]
                        diffy = y[i] - y[i - 1]
                        mdlx += diffx
                        mdly += diffy
                        matrix[mdlx, mdly] = 2.0

                dataframe_apf = pd.DataFrame.from_records(matrix)
                sns.heatmap(dataframe_apf, fmt="d", vmin=-1.0, vmax=2.0)
                if path_here is None:
                    plt.show()
                else:
                    plt.savefig("{}/trajectory_{}.png".format(path_here, tra_idx), dpi=500)
                plt.close()
                tra_idx += 1


