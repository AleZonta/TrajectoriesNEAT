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
from scipy.spatial import distance
from shapely.geometry import Point

VAL_NO_DATA = -1
LIMIT_TIMESTEPS = 5000
MAX_FITNESS = 200
MAX_TOTAL_FITNESS = MAX_FITNESS * 3


def _get_max_fitness_possible(fitness_definition):
    """
    Based on the fitness defined by settings, return the max possible fitness achievable by the algorithm
    :param fitness_definition: fitness chosen
    :return: int max fitness
    """
    if "mix" in fitness_definition:
        return Exception("Not okay to use this fitness types")
    elif fitness_definition == "novelty":
        return Exception("I do not have a max for this")
    elif fitness_definition == "moo":
        return Exception("I do not have a max for this")
    elif fitness_definition == "mootwo":
        return Exception("I do not have a max for this")
    elif fitness_definition == "same":
        return 100 * 100
    elif fitness_definition == "force_length":
        return 100 * 100 * 100
    elif fitness_definition == "force_length_sum":
        return 100 + 100 + 100
    elif fitness_definition == "variance":
        return Exception("I do not have a max for this")
    else:
        return MAX_TOTAL_FITNESS


def _get_fitness_length_curliness(point, external, internal):
    """
    Return the fitness from the features length and curliness.
    It used the fitness landscape defined and computes the distance to the important areas.
    If inside the center hull returns 0, otherwise return the distance to the border of the hulls.
    :param point: current features selected
    :param external: external hull of the fitness landscape
    :param internal: internal hull of the fitness landscape
    :return: distance from the important area
    """
    if internal.contains(point):
        actual_distance = 0
    elif external.contains(point):
        actual_distance = point.distance(internal)
    else:
        actual_distance = -point.distance(external)
    # if actual_distance < -150:
    #     actual_distance = -150
    return actual_distance


def _get_distance_to_center(point, internal, new_min=-300):
    """
    Return distance from current feature selected to the center of the hull selected.
    It also normalised the value of the distance to new scale defined by global variable MAX_FITNESS
    :param point: value for the features chosen
    :param internal: information about the internal hull of the fitness landscape
    :param new_min: minimum value of the new scale
    :return: normalised and raw distance
    """
    centroid = internal.centroid
    d = -distance.euclidean([centroid.x, centroid.y], [point.x, point.y])
    max_value = -5000
    if d < max_value:
        d = max_value
    fitness_value = convert(old_max=0, old_min=max_value, new_max=MAX_FITNESS, new_min=new_min, old_value=d)
    return fitness_value, d


def _get_combination_two_fitness_curliness_length(point, internal_normal, internal_special, external):
    """
    Return fitness value for the curliness and length features
    Two cases, either the internal hull is considered as a plateau or the distance to the centroid is returned
    :param point: current value for the two features chosen
    :param internal_normal: information of the internal hull
    :param internal_special: information of the internal hull with centroid
    :param external: information about the external hull
    :return: fitness value
    """
    value_from_curliness_length = _get_fitness_length_curliness(point=point,
                                                                external=external,
                                                                internal=internal_normal)
    if value_from_curliness_length == 0:
        value_from_curliness_length, _ = _get_distance_to_center(point=point,
                                                                 internal=internal_special, new_min=100)
    else:
        value_from_curliness_length = convert(old_max=0, old_min=-150, new_max=100, new_min=-300,
                                              old_value=value_from_curliness_length)
    return value_from_curliness_length


def _get_combination_two_fitness_curliness_distance(point, internal_normal, internal_special, external):
    """
    Return fitness value for the curliness and distance features
    Two cases, either the internal hull is considered as a plateau or the distance to the centroid is returned
    :param point: current value for the two features chosen
    :param internal_normal: information of the internal hull
    :param internal_special: information of the internal hull with centroid
    :param external: information about the external hull
    :return: fitness value
    """
    value_from_curliness_distance = _get_fitness_length_curliness(point=point,
                                                                  external=external,
                                                                  internal=internal_normal)
    if value_from_curliness_distance == 0:
        value_from_curliness_distance, _ = _get_distance_to_center(point=point,
                                                                   internal=internal_special, new_min=100)
    else:
        value_from_curliness_distance = convert(old_max=0, old_min=-150, new_max=100, new_min=-300,
                                                old_value=value_from_curliness_distance)
    return value_from_curliness_distance


def _get_combination_two_fitness_length_distance(point, internal_normal, internal_special, external):
    """
    Return fitness value for the distance and length features
    Two cases, either the internal hull is considered as a plateau or the distance to the centroid is returned
    :param point: current value for the two features chosen
    :param internal_normal: information of the internal hull
    :param internal_special: information of the internal hull with centroid
    :param external: information about the external hull
    :return: fitness value
    """
    value_from_distance_length = _get_fitness_length_curliness(point=point, external=external,
                                                               internal=internal_normal)
    if value_from_distance_length == 0:
        value_from_distance_length, _ = _get_distance_to_center(point=point, internal=internal_special,
                                                                new_min=100)
    else:
        value_from_distance_length = convert(old_max=0, old_min=-150, new_max=100, new_min=-300,
                                             old_value=value_from_distance_length)
    return value_from_distance_length


def get_fitness_value(length, curliness, fitness_landscape, further_distance, point_distance):
    """
    Get combined fitness function
    :param length: current length of the trajectory
    :param curliness: curliness of the trajectory
    :param fitness_landscape: data defining the fitness landscape
    :param further_distance: further distance to start of the trajectory
    :param point_distance: modification to the fitness function
    :return:
    """
    point = Point(curliness * 100, length)

    if 0 in point_distance:
        value_from_curliness_length = \
            _get_combination_two_fitness_curliness_length(point=point, internal_normal=fitness_landscape[1],
                                                          external=fitness_landscape[0],
                                                          internal_special=fitness_landscape[6])
    else:
        value_from_curliness_length = _get_fitness_length_curliness(point=point, external=fitness_landscape[0],
                                                                    internal=fitness_landscape[1])
        value_from_curliness_length = convert(old_max=0, old_min=-150, new_max=MAX_FITNESS, new_min=-300,
                                              old_value=value_from_curliness_length)

    point = Point(curliness * 100, further_distance)
    if 1 in point_distance:
        value_from_curliness_distance = \
            _get_combination_two_fitness_curliness_distance(point=point, internal_normal=fitness_landscape[3],
                                                            external=fitness_landscape[2],
                                                            internal_special=fitness_landscape[8])
    else:
        value_from_curliness_distance = _get_fitness_length_curliness(point=point, external=fitness_landscape[2],
                                                                      internal=fitness_landscape[3])
        value_from_curliness_distance = convert(old_max=0, old_min=-150, new_max=MAX_FITNESS, new_min=-300,
                                              old_value=value_from_curliness_distance)

    point = Point(further_distance, length)
    if 2 in point_distance:
        value_from_distance_length = \
            _get_combination_two_fitness_length_distance(point=point, internal_normal=fitness_landscape[5],
                                                         external=fitness_landscape[4],
                                                         internal_special=fitness_landscape[10])
    else:
        value_from_distance_length = _get_fitness_length_curliness(point=point, external=fitness_landscape[4],
                                                                   internal=fitness_landscape[5])
        value_from_distance_length = convert(old_max=0, old_min=-150, new_max=MAX_FITNESS, new_min=-300,
                                              old_value=value_from_distance_length)

    return value_from_distance_length + value_from_curliness_length + value_from_curliness_distance, \
           value_from_distance_length, value_from_curliness_length, value_from_curliness_distance


def convert(old_max, old_min, new_max, new_min, old_value):
    """
    Convert value from range A to range B
    :param old_max: max range A
    :param old_min: min range A
    :param new_max: max new range B
    :param new_min: min new range B
    :param old_value: value to convert
    :return: value converted
    """
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    return new_value


def get_next_point(current_point, direction):
    """
    Given current position and direction to move, it returns the coordinates of the next point
    :param current_point: current location
    :param direction: direction to move
    :return: next point
    """
    x_value = current_point.x
    y_value = current_point.y
    if direction == 0:
        return x_value - 1, y_value + 1
    elif direction == 1:
        return x_value, y_value + 1
    elif direction == 2:
        return x_value + 1, y_value + 1
    elif direction == 3:
        return x_value + 1, y_value
    elif direction == 4:
        return x_value + 1, y_value - 1
    elif direction == 5:
        return x_value, y_value - 1
    elif direction == 6:
        return x_value - 1, y_value - 1
    elif direction == 7:
        return x_value - 1, y_value
    else:
        raise Exception()
