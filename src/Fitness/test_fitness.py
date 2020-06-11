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
from scipy.spatial import distance


def convert(old_max, old_min, new_max, new_min, old_value):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    return new_value


def _get_distance_to_center(point, internal, new_min=-300):
    centroid = internal.centroid
    d = -distance.euclidean([centroid.x, centroid.y], [point.x, point.y])
    max_value = -5000
    if d < max_value:
        d = max_value
    fitness_value = convert(old_max=0, old_min=max_value, new_max=200, new_min=new_min, old_value=d)
    return fitness_value, d


def _get_fitness_length_curliness(point, external, internal):
    if internal.contains(point):
        return 0
    elif external.contains(point):
        actual_distance = point.distance(internal)
        # need to normalise from 0 and 160, need to find max
        # this distance goes from 0 to random number big as possible.
        # Need to set a max then convert to value from 160 to 0
        # max_value = 5000
        # if actual_distance > max_value:
        #     actual_distance = max_value
        # fitness_value = (((actual_distance - max_value) * (200 - 0)) / (0 - max_value)) + 0
        return actual_distance
    else:
        actual_distance = point.distance(external)
        return -actual_distance
