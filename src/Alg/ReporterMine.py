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
import time

from neat.math_util import mean, stdev
from neat.reporting import BaseReporter
from neat.six_util import itervalues, iterkeys

"""
Extend the base reporter that comes with the NEAT package to add mlflow and the normal logger to it
"""
class MineReporter(BaseReporter):
    """Uses `mlflow and logger` to output information about the run"""
    def __init__(self, show_species_detail, logger, mlflow):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self._logger = logger
        self._mlflow = mlflow

    def start_generation(self, generation):
        self.generation = generation
        self._logger.info('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            self._logger.info('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            self._logger.info("   ID   age  size  fitness  adj fit  stag")
            self._logger.info("  ====  ===  ====  =======  =======  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                self._logger.info(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))
                try:
                    self._mlflow.log_metric(key=str(sid), value=s.fitness)
                except Exception as e:
                    self._logger.info("Impossible to record mlflow metric")

        else:
            self._logger('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        self._logger.info('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            self._logger.info("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            self._logger.info("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        self._logger.info('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        self._logger.info(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))
        self._mlflow.log_metric(key="best_fitness", value=best_genome.fitness)
        self._mlflow.log_metric(key="best_fitness_species", value=best_species_id)
        self._mlflow.log_metric(key="best_fitness_id", value=best_genome.key)

    def complete_extinction(self):
        self.num_extinctions += 1
        self._logger.info('All species extinct.')

    def found_solution(self, config, generation, best):
        self._logger.info('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            self._logger.info("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        try:
            self._logger.info(msg)
        except Exception as e:
            print(msg)

    def set_logger_none(self):
        self._mlflow = None
        self._logger = None

    def get_loggers(self):
        return self._mlflow, self._logger

    def set_loggers(self, mlflow, logger):
        self._mlflow = mlflow
        self._logger = logger
