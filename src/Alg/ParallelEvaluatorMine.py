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

import numpy as np
from neat import ParallelEvaluator

from src.Algorithms.support import compute_matrix_distances, get_novelty_score
from src.Fitness.ValueGraphFitness import convert, MAX_TOTAL_FITNESS, MAX_FITNESS

"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool


class ParallelEvaluatorMine(ParallelEvaluator):
    def __init__(self, num_workers, eval_function, fitness_definition, timeout=None, k=15):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        super().__init__(num_workers, eval_function, timeout)
        self.pool = Pool(processes=num_workers, maxtasksperchild=1)
        self._k = k
        self._fitness_definition = fitness_definition

    def evaluate(self, genomes, config):
        archive = config[1]
        config = config[0]
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        double_fitness = []
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            double_fitness.append(job.get(timeout=self.timeout))

        real_fitness = []
        all_the_behaviours = []
        variances = []
        all_the_data_to_save = []
        added_constraints = []
        value_direction = []
        for fit in double_fitness:
            fit_result = fit
            real_fitness.append(fit_result[0])
            all_the_behaviours.append(fit_result[1])
            variances.append(fit_result[2])
            all_the_data_to_save.append(fit_result[5])
            added_constraints.append(fit_result[3])
            value_direction.append(fit_result[4])

        total_fitness = []
        total_fitness.extend(all_the_behaviours)
        total_fitness.extend([archive[el] for el in range(len(archive))])

        # now what is returned is not a fitness, but a list of numpy vectors with behavioural information
        distances = compute_matrix_distances(data_a=all_the_behaviours, data_b=total_fitness)
        novelty_score = get_novelty_score(data=distances, k=self._k)

        real_fitness_converted = [convert(old_max=MAX_TOTAL_FITNESS, old_min=-300.,
                                          new_max=100, new_min=0, old_value=el if el > -300 else -300) for el in
                                  real_fitness]
        novelty_score_converted = [convert(old_max=1000., old_min=0.0,
                                           new_max=100, new_min=1, old_value=el if el < 1000 else 1000) for el in
                                   novelty_score]
        if self._fitness_definition == "novelty":
            for value, (ignored_genome_id, genome) in zip(novelty_score, genomes):
                genome.fitness = value
        elif self._fitness_definition == "same":
            final_fitness = np.array(real_fitness_converted) * np.array(novelty_score_converted)
            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "moo":
            raise NotImplementedError("moo ont implemented with neat")
        elif self._fitness_definition == "force_length":

            median_length = []
            for el in all_the_data_to_save:
                all_values = [len(sub_el) for sub_el in el[2]]
                median_value = np.mean(np.array(all_values))
                median_length.append(convert(old_max=5000., old_min=0.0, new_max=100, new_min=1,
                                             old_value=median_value if median_value < 5000 else 5000))

            final_fitness = np.array(real_fitness_converted) * np.array(novelty_score_converted) * np.array(median_length)
            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "force_length_sum":
            median_length = []
            for el in all_the_data_to_save:
                all_values = [len(sub_el) for sub_el in el[2]]
                median_value = np.mean(np.array(all_values))
                median_length.append(convert(old_max=5000., old_min=0.0, new_max=100, new_min=1,
                                             old_value=median_value if median_value < 5000 else 5000))

            final_fitness = np.array(real_fitness_converted) + np.array(novelty_score_converted) + np.array(median_length)
            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "variance":
            final_fitness = np.array(real_fitness) + np.array(variances)
            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "normal":
            final_fitness = np.array(real_fitness)
            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "normal_direction":
            value_direction_converted = [convert(old_max=12, old_min=0, new_max=MAX_FITNESS, new_min=1,
                                                 old_value=el_val) for el_val in value_direction]
            final_fitness = np.array(real_fitness) + np.array(value_direction_converted)

            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "normal_direction_five":
            value_direction_converted = [convert(old_max=12, old_min=0, new_max=MAX_FITNESS, new_min=1,

                                                      old_value=el_val) for el_val in value_direction]
            final_fitness = np.array(real_fitness) + np.array(value_direction_converted) * 5

            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "normal_direction_ten":
            value_direction_converted = [convert(old_max=12, old_min=0, new_max=MAX_FITNESS, new_min=1,
                                                 old_value=el_val) for el_val in value_direction]
            final_fitness = np.array(real_fitness) + np.array(value_direction_converted) * 10

            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "normal_direction_fifteen":
            value_direction_converted = [convert(old_max=12, old_min=0, new_max=MAX_FITNESS, new_min=1,
                                                 old_value=el_val) for el_val in value_direction]
            final_fitness = np.array(real_fitness) + np.array(value_direction_converted) * 15

            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "normal_direction_twenty":
            value_direction_converted = [convert(old_max=12, old_min=0, new_max=MAX_FITNESS, new_min=1,
                                                 old_value=el_val) for el_val in value_direction]
            final_fitness = np.array(real_fitness) + np.array(value_direction_converted) * 20

            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "direction_normal":
            value_direction_converted = [convert(old_max=12, old_min=0, new_max=MAX_FITNESS, new_min=1,
                                                 old_value=el_val) for el_val in value_direction]
            final_step_fitness = [value_direction_converted[i] + real_fitness[i] if value_direction_converted[i] > 125 else value_direction_converted[i] for i in range(len(value_direction_converted))]
            final_fitness = np.array(final_step_fitness)

            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "normal_direction_division":
            value_direction_converted = [convert(old_max=12, old_min=0, new_max=MAX_FITNESS, new_min=1,
                                                 old_value=el_val) for el_val in value_direction]
            final_fitness = (np.array(real_fitness) + np.array(value_direction_converted)) / 4

            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "normal_separation":
            final_fitness = np.array(real_fitness) + np.array(added_constraints)

            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        elif self._fitness_definition == "normal_both":
            value_direction_converted = [convert(old_max=8, old_min=1, new_max=MAX_FITNESS, new_min=1,
                                                 old_value=el_val) for el_val in value_direction]
            final_fitness = np.array(real_fitness) + np.array(value_direction_converted) + np.array(added_constraints)

            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value
        else:
            # I have the values for the directions
            # min is 1, max is 8
            #final_fitness = np.array(real_fitness) + np.array(added_constraints)

            # value_direction_converted = [convert(old_max=8, old_min=1, new_max=MAX_FITNESS, new_min=1,
            #                                      old_value=el_val) for el_val in value_direction]
            # value_direction_converted = []
            # for el in value_direction:
            #     if el < 5:
            #         value_direction_converted.append(0.5)
            #     elif el == 5:
            #         value_direction_converted.append(1)
            #     elif el == 6:
            #         value_direction_converted.append(2)
            #     elif el == 7:
            #         value_direction_converted.append(3)
            #     elif el == 8:
            #         value_direction_converted.append(4)
            #
            # final_fitness *= np.array(value_direction_converted)

            # first the four direction
            # part_c = np.array(value_direction)
            # to_visit = -(360 - part_c)
            # okay_indexes_c = (to_visit > -110).astype(np.int)
            #
            # # then the constraint
            # part_b = np.array(added_constraints)
            # part_b_updated = part_b * okay_indexes_c
            # okay_indexes_b = (part_b_updated > 100).astype(np.int)
            #
            # # then the real a part
            # part_a = np.array(real_fitness)
            # part_a_updated = part_a * okay_indexes_b
            #
            # final_fitness = part_a_updated + part_b_updated + to_visit

            # value_direction_converted = np.array([convert(old_max=8, old_min=1, new_max=MAX_FITNESS, new_min=1,
            #                                      old_value=el_val) for el_val in value_direction])
            # final_fitness = np.array(real_fitness) + value_direction_converted + np.array(added_constraints)
            # final_fitness /= 3

            # final_fitness *= np.array(value_direction)
            final_fitness = np.array(real_fitness)
            for value, (ignored_genome_id, genome) in zip(final_fitness, genomes):
                genome.fitness = value

        # assign the fitness back to each genome
        for value, (ignored_genome_id, genome) in zip(variances, genomes):
            genome.variances_good_trajectories = value
        for value, (ignored_genome_id, genome) in zip(real_fitness, genomes):
            genome.real_fitness = value
            # print(value)
        for value, (ignored_genome_id, genome) in zip(novelty_score, genomes):
            genome.novelty_score = value
            # print(value)
        for value, (ignored_genome_id, genome) in zip(all_the_behaviours, genomes):
            genome.behaviour = value
            # print(value)

        for value, (ignored_genome_id, genome) in zip(all_the_data_to_save, genomes):
            genome.all_the_data = value

        for value, (ignored_genome_id, genome) in zip(value_direction, genomes):
            genome.value_direction = value




