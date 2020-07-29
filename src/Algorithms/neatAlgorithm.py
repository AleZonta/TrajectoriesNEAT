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

import json
import multiprocessing
import pickle
import numpy as np

from joblib import Parallel, delayed

import os
import neat
from src.Alg.CheckpointerMine import CheckpointerMine
from src.Alg.GenomeMine import NoveltyGenome
from src.Alg.ParallelEvaluatorMine import ParallelEvaluatorMine
from src.Alg.ReporterMine import MineReporter
from src.Alg.population_mine import PopulationWithNovelty
from src.Algorithms.winning_genome import worker_job_lib_winning_genome
from src.Algorithms.winning_genome_attractions import worker_job_lib_behaviours
from src.Fitness.GeneralFitness import compute_general_fitness, PRE_DEFINED_BEHAVIOURS_ALL
from src.Helpers.APF import LoadAPF
from src.Helpers.Division.ComputeDivision import SubMatrix
from src.Helpers.GenomePhenome import GenomeMeaning

from src.Settings.arguments import args


def eval_genomes(genome, config):
    """
    Function that evaluates the genome
    It loads the train points, it loads the precomputed distances and the fitness landscape definition
    It generates the trajectories and then it evaluates the fitness function
    :param genome: NEAT genome
    :param config: NEAT config
    :return: trajectories information, all the single fitness component and the final fitness function
    """
    number_of_tra_to_generate = args.numb_of_tra

    loader_apf = LoadAPF(path="{}/the_right_one_fast".format(args.data_directory),
                         logger=None)
    loader_apf.load_apf_only_routing_system()
    loader_apf.match_index_with_coordinates()

    real_tra_train = np.load("{}/real_tra_train_starting_points.npy".format(args.data_directory), allow_pickle=True)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    loader_genome_meaning = GenomeMeaning(logger=None)
    loader_genome_meaning.load_data()

    sub_matrix = SubMatrix(log=None, apf=loader_apf.apf,
                           list_points=loader_genome_meaning.name_typologies,
                           values_matrix=(loader_apf.x_values, loader_apf.y_values))
    sub_matrix.divide_into_cells()

    fitness_landscape = pickle.load(open("{}/3d_fitness_in_2d_with_limitation.pickle".format(args.data_directory), 'rb'))

    # fitness_total = []
    # behaviour_total = []
    # variance_total = []
    # direction_total = []
    # all_averaged_converted_distances = []
    # different_total_data_together = []
    # for _ in range(10):
    all_result_normal, all_results_behaviours, all_tra_generated, all_variance, average_converted_distance, \
    single_fitness_data, all_the_directions = compute_general_fitness(net=net,
                                                                      number_to_generate=number_of_tra_to_generate,
                                                                      real_tra=real_tra_train,
                                                                      apf=loader_apf.apf,
                                                                      sub_matrix=sub_matrix,
                                                                      fitness_landscape=fitness_landscape,
                                                                      random_initial_point=args.random_point_start,
                                                                      point_distance=args.point_distance,
                                                                      penalty_fitness=args.penalty_behaviours)

    all_result_normal_general = np.mean(np.array(all_result_normal))
    # fitness_total.append(all_result_normal_general)
    all_result_behaviours_general = np.mean(np.array(all_results_behaviours), axis=0)
    # behaviour_total.append(all_result_behaviours_general)
    variance_general = np.mean(np.array(all_variance), axis=0)
    # variance_total.append(variance_general)
    # all_averaged_converted_distances.append(average_converted_distance)

    helper = np.array(all_the_directions)
    helper = np.sort(helper, kind="heapsort")
    helper_diff = helper[1:]
    helper_diff = np.append(helper_diff, helper[0])
    res = helper_diff - helper
    indexes = np.where(res < 0)
    res[indexes] = res[indexes] + 360
    single_direction_value = np.mean(res)
    # direction_total.append(single_direction_value)

    total_data_together = (all_result_normal, all_results_behaviours, all_tra_generated, all_variance,
                           average_converted_distance, single_fitness_data, all_the_directions)
    # different_total_data_together.append(total_data_together)

    # all_result_normal_general = np.mean(np.array(fitness_total))
    # all_result_behaviours_general = np.mean(np.array(behaviour_total), axis=0)
    # variance_general = np.mean(np.array(variance_total), axis=0)
    # single_direction_value = np.mean(np.array(direction_total))
    # average_converted_distance = np.mean(np.array(all_averaged_converted_distances))

    return all_result_normal_general, all_result_behaviours_general, variance_general, average_converted_distance, \
           single_direction_value, total_data_together


class neatAlgorithm(object):
    def __init__(self, mlflow, config_file, logger, output, fitness_definition, prob_add=0.1):
        self._log = logger
        self._mlflow = mlflow
        # Load configuration.
        self._config = neat.Config(NoveltyGenome, neat.DefaultReproduction,
                                   neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                   config_file)

        self._population = None
        self._stats = None
        self._output_directory = output
        self._fitness_definition = fitness_definition
        self._prob_add = prob_add

    def initialise(self, frequency_checkpoints, restore_checkpoint_name=None):
        if restore_checkpoint_name is not None:
            self._population = CheckpointerMine.restore_checkpoint_with_novelty(filename=restore_checkpoint_name,
                                                                                output_directory=self._output_directory,
                                                                                prob_add=self._prob_add)
        else:
            # Create the population, which is the top-level object for a NEAT run.
            self._population = PopulationWithNovelty(self._config, prob_add=self._prob_add,
                                                     output_directory=self._output_directory)

        # Add a stdout reporter to show progress in the terminal.
        self._population.add_reporter(MineReporter(show_species_detail=True, logger=self._log, mlflow=self._mlflow))
        self._stats = neat.StatisticsReporter()
        self._population.add_reporter(self._stats)
        self._population.add_reporter(CheckpointerMine(generation_interval=frequency_checkpoints,
                                                       filename_prefix="{}/neat-checkpoint-".format(
                                                           self._output_directory)))

    def run(self, generations):
        pe = ParallelEvaluatorMine(num_workers=multiprocessing.cpu_count(), eval_function=eval_genomes,
                                   fitness_definition=self._fitness_definition)
        winner = self._population.run(pe.evaluate, generations)
        # Display the winning genome.
        self._log.info('\nBest genome:\n{!s}'.format(winner))

        self._log.info('Saving file best genome...')
        with open("{}/winning_genome.pkl".format(self._output_directory), "wb") as cp_file:
            pickle.dump(winner, cp_file, protocol=pickle.HIGHEST_PROTOCOL)

        self._log.info('Saving statistics...')
        with open("{}/statistics.pkl".format(self._output_directory), "wb") as cp_file:
            pickle.dump(self._stats, cp_file, protocol=pickle.HIGHEST_PROTOCOL)

        try:
            self.test_winning_genome(individual=winner)
        except Exception as e:
            self._log.debug("Error occurred in testing the winning genome. Operation Aborted")
            self._log.debug(e)
        try:
            self.test_winning_genome_attraction_scales(individual=winner, output_dir=self._output_directory)
        except Exception as e:
            self._log.debug("Error occurred in testing the attraction. Operation Aborted")
            self._log.debug(e)

    def generate_trajectories_from_checkpoint(self, number_of_trajectories_to_generate):
        pe = ParallelEvaluatorMine(num_workers=2, eval_function=eval_genomes,
                                   fitness_definition=self._fitness_definition)
        self._log.info("Re-evalaute again population")

        pop = self._population.get_population(pe.evaluate)

        # find indexes with high fitness
        all_the_fitness = [pop[ind].real_fitness for ind in pop]
        index_max_fitness = sorted(range(len(all_the_fitness)), key=lambda sub: all_the_fitness[sub])[-5:]

        small_pop_list = [list(pop.items())[idx] for idx in index_max_fitness]
        small_pop = {}
        for el in small_pop_list:
            small_pop[el[0]] = el[1]

        self._log.info("Original value generated evolving with 1 tra top 5")
        for ind in small_pop:
            print("{}; {}".format(pop[ind].real_fitness,
                                  pop[ind].behaviour
                                  ))

        for to_generate in number_of_trajectories_to_generate:
            # write new value for testing again
            # number_of_tra_to_generate_old = args.numb_of_tra
            args.numb_of_tra = to_generate

            with open('commandline_args.txt', 'w') as f:
                json.dump(args.__dict__, f, indent=2)

            with open('commandline_args.txt', 'r') as f:
                args.__dict__ = json.load(f)
            pop = self._population.evaluate_small_set_population(fitness_function=pe.evaluate, pop=small_pop)

            self._log.info("Robustness model over {} tra".format(to_generate))
            for ind in pop:
                print("{}; {}".format(pop[ind].real_fitness,
                                      pop[ind].behaviour
                                      ))

    def test_winning_genome(self, individual, output_dir=None, debug=False):
        starting_point_train = np.load("{}/real_tra_test_starting_points.npy".format(args.data_directory), allow_pickle=True)
        # generate 200 trajectories from the first 200 points
        trajectories_to_generate = args.number_of_test_trajectories

        self._log.info("Winning genome is tested to generate {} new trajectories".format(trajectories_to_generate))

        cores = multiprocessing.cpu_count()
        if cores > trajectories_to_generate:
            cores = trajectories_to_generate
        if debug:
            cores = 1
        with Parallel(n_jobs=cores, verbose=30) as parallel:
            res = parallel(delayed(worker_job_lib_winning_genome)(starting_point_train[i], individual, self._config,
                                                   args.point_distance) for i in range(trajectories_to_generate))

        if output_dir is None:
            output_dir = self._output_directory

        with open("{}test_winning_genome_trajectories.pkl".format(output_dir), "wb") as cp_file:
            pickle.dump(res, cp_file, protocol=pickle.HIGHEST_PROTOCOL)

        self._log.info("-----Robustness winning genomes-----")
        all_fitness = [el[0] for el in res]
        all_behaviours = [el[1] for el in res]
        mean_fitness = np.mean(np.array(all_fitness))
        mean_behaviours = np.mean(np.array(all_behaviours), axis=0)
        self._log.info("from: {}; {}".format(individual.real_fitness, individual.behaviour))
        self._log.info("to: {}; {}".format(mean_fitness, mean_behaviours))

    def test_winning_genome_attraction_scales(self, individual, name="exp", output_dir=None, debug=False):
        self._log.info("Winning genome is tested to generate different behaviours")
        self._log.info("{}; {}".format(individual.real_fitness, individual.behaviour))

        starting_point_train = np.load("{}/real_tra_test_starting_points.npy".format(args.data_directory), allow_pickle=True)

        # realize the multiplier_settings
        # 6 positions
        # from 0 to 100
        permutations = PRE_DEFINED_BEHAVIOURS_ALL

        trajectories_to_generate = args.number_of_test_trajectories
        cores = multiprocessing.cpu_count()
        if cores > trajectories_to_generate:
            cores = trajectories_to_generate
        if debug:
            cores = 1
            permutations = permutations[:2]

        with Parallel(n_jobs=cores, verbose=30) as parallel:
            res = parallel(delayed(worker_job_lib_behaviours)(starting_point_train, individual, self._config,
                                                              args.point_distance, permutations[i], output_dir, name, i)
                           for i in
                           range(len(permutations)))
