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

# the way I save trajecotries data is different from astar or random walk, therefore I need to adapt my results
# to be used by the other functions
import glob
import logging
import pickle

import numpy as np
import pandas as pd
from tqdm import trange

from src.Fitness.GeneralFitness import compute_direction
from src.Fitness.ValueGraphFitness import convert, MAX_FITNESS
from src.Helpers.Funcs import sorted_nicely


class NEATFitnessAnalyser(object):
    def __init__(self, log):
        self._log = log

    def read_data(self, path, path_to_save, name_to_save):
        correct_files = glob.glob("{}/test_multipliers_attraction*".format(path))

        # need to store all the data from the trajectories
        data_tra = []
        for el in correct_files:
            content = pickle.load(open(el, 'rb'))
            # content[0] contains setting tested
            # content[1] contains data from the trajectories
            data_tra.append(content[1])

        df = pd.DataFrame(columns=['fitness', 'total_length', 'curliness', "f_d_to_p",
                                   "d_to_m_pt", "d_to_end_p", "direction",
                                   "no_overlapping"])
        for i in trange(len(data_tra[0][0]), desc="reading files"):
            # trajectory i
            all_fitness = []
            all_behaviours = []
            current_analysed = []
            for j in range(len(data_tra)):
                # pass through all the data
                all_fitness.append(data_tra[j][0][i])
                all_behaviours.append(data_tra[j][1][i])

                current_trajectory = data_tra[j][2][i]
                current_analysed.append(current_trajectory)

            # get more info from the trajectories generated
            values_of_same_elements = []
            for ii in range(len(current_analysed)):
                first_tra = [[int(p.x), int(p.y)] for p in current_analysed[ii]]
                for j in range(ii + 1, len(current_analysed)):
                    second_tra = [[int(p.x), int(p.y)] for p in current_analysed[j]]
                    tot = [*first_tra, *second_tra]

                    equality = pd.DataFrame(np.array(tot).T).T.drop_duplicates(keep=False).values.shape[0] / len(
                        tot)
                    # s = min(len(first_tra), len(second_tra))
                    # count = np.count_nonzero(first_tra[:s] == second_tra[:s])
                    # number of similar value over s
                    # equality = 1 - (count / (s * 2))
                    values_of_same_elements.append(equality)
            average_distance = np.mean(np.array(values_of_same_elements))
            # 0 is exactly the same vectors
            # 1 is exactly different vectors
            average_converted_distance = convert(old_max=1, old_min=0, new_max=MAX_FITNESS, new_min=0,
                                                 old_value=average_distance)

            # need to force to go to four directions

            # I have starting point
            # I have ending points
            all_the_directions = [compute_direction(origin=tra[0], destination=tra[-1]) for tra in current_analysed]

            all_result_normal_general = np.mean(np.array(all_fitness))
            all_result_behaviours_general = np.mean(np.array(all_behaviours), axis=0)
            single_direction_value = len(set(all_the_directions))

            df.loc[i] = [all_result_normal_general, all_result_behaviours_general[0],
                         all_result_behaviours_general[1] * 100,
                         all_result_behaviours_general[2], all_result_behaviours_general[3],
                         all_result_behaviours_general[4], single_direction_value * 100, average_converted_distance]
        df.to_hdf("{}/{}_save".format(path_to_save, name_to_save), key='data', mode='w')
        self._log.info("Data Saved!")

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

    a = NEATFitnessAnalyser(log=logger)
    path = "/Volumes/Data/ale/random_walk/output/"
    folders = sorted_nicely(glob.glob("{}*/".format(path)))
    for f in folders:
        name_folder = f.split("/")[-1]
        a.read_data(path="{}/{}".format(path, name_folder), path_to_save="{}".format(path), name_to_save=name_folder)


