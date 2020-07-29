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
import pickle

import neat

from src.Fitness.GeneralFitness import compute_general_fitness
from src.Helpers.APF import LoadAPF
from src.Helpers.Division.ComputeDivision import SubMatrix
from src.Helpers.GenomePhenome import GenomeMeaning
from src.Settings.arguments import args

"""
Multiprocessing function to generate trajectories conditioned from different behaviours
"""
def worker_job_lib_behaviours(starting_point, genome, config, point_distance, multipliers, output_dir, name, i):
    loader_apf = LoadAPF(path="{}/the_right_one_fast".format(args.data_directory),
                         logger=None)
    loader_apf.load_apf_only_routing_system()
    loader_apf.match_index_with_coordinates()

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    loader_genome_meaning = GenomeMeaning(logger=None)
    loader_genome_meaning.load_data()

    sub_matrix = SubMatrix(log=None, apf=loader_apf.apf,
                           list_points=loader_genome_meaning.name_typologies,
                           values_matrix=(loader_apf.x_values, loader_apf.y_values))
    sub_matrix.divide_into_cells()

    starting_point = starting_point[:50]

    fitness_landscape = pickle.load(open("{}/3d_fitness_in_2d_with_limitation.pickle".format(args.data_directory), 'rb'))
    all_result_normal, all_results_behaviours, all_tra_generated, all_variance, average_converted_distance, \
    single_fitness_data, all_the_directions = compute_general_fitness(net=net,
                                                                      number_to_generate=len(starting_point),
                                                                      real_tra=starting_point,
                                                                      apf=loader_apf.apf,
                                                                      sub_matrix=sub_matrix,
                                                                      fitness_landscape=fitness_landscape,
                                                                      random_initial_point=False,
                                                                      point_distance=point_distance,
                                                                      penalty_fitness=args.penalty_behaviours,
                                                                      multiplier=multipliers,
                                                                      single_tra=True)

    res = (multipliers, (all_result_normal, all_results_behaviours, all_tra_generated, all_variance,
                         average_converted_distance, single_fitness_data, all_the_directions))

    with open("{}/test_multipliers_attraction_tot_{}_{}.pkl".format(output_dir, name, i), "wb") as cp_file:
        pickle.dump(res, cp_file, protocol=pickle.HIGHEST_PROTOCOL)

    return res
