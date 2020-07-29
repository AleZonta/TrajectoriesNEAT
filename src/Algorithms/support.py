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
import random
from copy import deepcopy
from math import inf

import numpy as np


class Archive(object):
    """The Archive contains the highest novelty individuals.
    The insertion is ordered following the generation of the evolution.

    A single copy of each individual is kept at all time.

    :param maxsize: The maximum number of individual to keep in the Archive.
                    It default value is infinite
    :param similar: An equivalence operator between two individuals, optional.
                    It defaults to operator :func:`operator.eq`.

    The class :class:`Archive` provides an interface similar to a list
    (without being one completely). It is possible to retrieve its length, to
    iterate on it forward and backward and to get an item or a slice from it.
    """

    def __init__(self, maxsize=inf, prob_add=0.1):
        self._maxsize = maxsize
        self._keys = list()
        self._items = list()
        self._prob_add = prob_add

    def update(self, population):
        """Update the Archive with the *population* by erasing the
        oldest individuals in it and adding by the latest individuals present in
        *population* (if the size of the archive is fixed).
        individual added only with probability *prob_add*

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        v = random.uniform(0, 1)
        if isinstance(population, dict):
            vector_fitness = [population[p].fitness for p in population.keys()]
            index_element = [p for p in population.keys()]
            index = np.argpartition(vector_fitness, len(vector_fitness) - 1)[len(vector_fitness) - 1:]

            current_ind = population[index_element[0]]

        else:
            vector_fitness = [p.fitness for p in population]
            index = np.argpartition(vector_fitness, len(vector_fitness) - 1)[len(vector_fitness) - 1:]

            current_ind = population[index[0]]

        if v < self._prob_add:
            self.insert(current_ind.behaviour)
            if len(self) > self._maxsize:
                self.remove(0)

    def insert(self, item):
        """Insert a new individual in the archive  Inserting a new
        individual in the hall of fame also preserve the hall of fame's order.
        This method **does not** check for the size of the hall of fame, in a
        way that inserting a new individual in a full hall of fame will not
        remove the worst individual to maintain a constant size.

        :param item: The individual with a fitness attribute to insert in the
                     hall of fame.
        """
        item = deepcopy(item)
        self._items.append(item)
        self._keys.append(item)

    def remove(self, index):
        """Remove the specified *index* from the Archive.

        :param index: An integer giving which item to remove.
        """
        del self._keys[index]
        del self._items[index]

    def clear(self):
        """Clear the archive."""
        del self._items[:]
        del self._keys[:]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    def __reversed__(self):
        return reversed(self._items)

    def __str__(self):
        return str(self._items)


def compute_matrix_distances(data_a, data_b):
    matrix = np.zeros((len(data_a), len(data_b)))
    for i in range(len(data_a)):
        for j in range(len(data_b)):
            vector_a = data_a[i][0]
            vector_b = data_b[j][0]
            matrix[i, j] = np.sqrt(np.sum((vector_a - vector_b) ** 2))
    return matrix


def get_novelty_score(data, k):
    if k > data.shape[0]:
        k = data.shape[0] - 1
    values = []
    for i in range(data.shape[0]):
        vector = data[i]
        idx = np.argpartition(vector, k)
        values.append(float(np.mean(vector[idx[:k]])))
    return values
