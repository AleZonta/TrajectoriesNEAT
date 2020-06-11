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
import math
import random
import numpy as np
import pandas as pd
from neat.nn import FeedForwardNetwork
from scipy.spatial import distance
import itertools

from src.Fitness.ValueGraphFitness import VAL_NO_DATA, LIMIT_TIMESTEPS, convert, MAX_FITNESS, MAX_TOTAL_FITNESS, \
    get_fitness_value
from src.Helpers.Funcs import list_neighbours
from src.Helpers.Point import Point

PRE_DEFINED_BEHAVIOURS_ALL = list(set(itertools.permutations([-1, -1, -1, -1, -0.011, 1])))
PRE_DEFINED_BEHAVIOURS = {
    0: [1, -1, -1, -1, -1, -1],
    1: [-1, 1, -1, -1, -1, -1],
    2: [-1, -1, 1, -1, -1, -1],
    3: [-1, -1, -1, 1, -1, -1],
    4: [-1, -1, -1, -1, 1, -1],
    5: [-1, -1, -1, -1, -1, 1],
    6: [-0.011, -0.011, -0.011, -0.011, -0.011, -0.0111],
    7: [-1, -1, -1, -1, -0.011, 1],
    8: [-1, -1, -1, -0.011, -1, 1],
    9: [-1, -1, -1, -1, -1, -1]
}
COMPASS_BRACKETS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]


def compute_direction(origin, destination):
    deltaX = destination.x - origin.x
    deltaY = destination.y - origin.y
    degrees_temp = math.atan2(deltaX, deltaY) / math.pi * 180
    if degrees_temp < 0:
        degrees_final = 360 + degrees_temp
    else:
        degrees_final = degrees_temp

    compass_lookup = round(degrees_final / 45)
    return COMPASS_BRACKETS[compass_lookup]


def compute_direction_angle(origin, destination):
    deltaX = destination.x - origin.x
    deltaY = destination.y - origin.y
    degrees_temp = math.atan2(deltaX, deltaY) / math.pi * 180
    if degrees_temp < 0:
        degrees_final = 360 + degrees_temp
    else:
        degrees_final = degrees_temp

    return degrees_final


def compute_general_fitness(net, number_to_generate, real_tra, apf, sub_matrix, fitness_landscape,
                            random_initial_point, point_distance, penalty_fitness, multiplier=None, single_tra=None):
    """
        compute fitness using the combination of distance and curliness loaded from a file
        (prediscovered areas)
        :param number_to_generate: number of tra to generate
        :param real_tra: real trajecotries, to use the starting point as a base
        :param apf: apf, needed to check if in road
        :return: list of all the fitnesses for all the trajectories generated
        """
    all_results_behaviours = []
    all_result_normal = []
    all_tra_generated = []
    all_variance = []
    single_fitness_data = []

    if point_distance is None:
        point_distance = []

    if point_distance is None:
        point_distance = []
    for i in range(number_to_generate):
        idx_tra = random.randint(0, len(real_tra) - 1) if random_initial_point else 0
        if single_tra is not None:
            idx_tra = i
        # get starting point real trajectory
        first_point_str = real_tra[idx_tra].split("-")
        starting_position_x = int(first_point_str[0])
        starting_position_y = int(first_point_str[1])

        # tra generated (for length)
        tra_generated = []
        # direction generated (for curliness)
        output_generated = []

        # condition to stop generation when is not on the road
        is_on_the_road = True
        count_timesteps = 0
        while is_on_the_road:

            current_point = Point(x=starting_position_x, y=starting_position_y)
            tra_generated.append(current_point)

            # generate input space
            # first find the neighbours points
            points = list_neighbours(x_value=starting_position_x, y_value=starting_position_y,
                                     apf=apf.shape)
            points_on_the_street = sub_matrix.keep_only_points_on_street(
                points=points)  # I do not really need this here
            all_the_charges = []
            final_with_additional = []
            for p in points:
                if p in points_on_the_street:
                    attractions = sub_matrix.return_distance_from_point(current_position=current_point)
                    all_the_charges.extend(attractions)
                    final_with_additional.append(1)
                else:
                    all_the_charges.extend([float(VAL_NO_DATA) for _ in range(6)])
                    final_with_additional.append(VAL_NO_DATA)
            all_the_charges.extend(final_with_additional)

            # all_the_charges = []
            # for p in points:
            #     if p in points_on_the_street:
            #         all_the_charges.append(1)
            #     else:
            #         all_the_charges.append(VAL_NO_DATA)

            # add count of timesteps already moved
            all_the_charges.append(convert(old_max=LIMIT_TIMESTEPS, old_min=0,
                                           new_max=1, new_min=0, old_value=count_timesteps))

            # add behaviour
            if multiplier is None:
                if number_to_generate > 10:
                    behavior = PRE_DEFINED_BEHAVIOURS_ALL[i]
                else:
                    behavior = PRE_DEFINED_BEHAVIOURS[i]
                all_the_charges.extend(behavior)
            else:
                all_the_charges.extend(multiplier)

            # input_data = np.reshape(np.array(all_the_charges), (1, 1, 57))
            input_data = np.reshape(np.array(all_the_charges), (1, 1, 63))
            # input_data = np.reshape(np.array(all_the_charges), (1, 1, 9))
            assert (np.max(input_data) <= 1.0)
            assert (np.min(input_data) >= -1.0)

            if isinstance(net, FeedForwardNetwork):
                output_network = net.activate(all_the_charges)
            else:
                output_network = net.predict(input_data).flatten().tolist()
            output_generated.append(output_network)
            # what if output is all the same and therefore no decision can be made?
            if np.sum(output_network[:-1]) == 0:
                # stay on the same spot
                # if I do nothing I lose a lot of time here till I reach count_timesteps
                # just exit
                if not isinstance(net, FeedForwardNetwork):
                    net.reset_states()
                is_on_the_road = False
            else:
                val_max = max(output_network)
                number_of_times = 0
                for el in output_network:
                    if el == val_max:
                        number_of_times += 1
                if number_of_times == 1:
                    direction = output_network.index(val_max)
                else:
                    indexes = [ik for ik in range(len(output_network)) if output_network[ik] == val_max]
                    direction = indexes[random.randint(0, len(indexes) - 1)]
                if direction == 8:
                    # stop generation
                    if not isinstance(net, FeedForwardNetwork):
                        net.reset_states()
                    is_on_the_road = False
                else:
                    starting_position_x, starting_position_y = get_next_point(current_point, direction)
                    if apf.iloc[starting_position_x, starting_position_y] < 40:
                        # reset state network for next trajectory
                        if not isinstance(net, FeedForwardNetwork):
                            net.reset_states()
                        is_on_the_road = False

            count_timesteps += 1
            # set limit trajectory
            if count_timesteps > LIMIT_TIMESTEPS:
                if not isinstance(net, FeedForwardNetwork):
                    net.reset_states()
                is_on_the_road = False

        # now I have trajectory and direction, need to compute the fitness
        # as distance I am using the number of timesteps of the trajectories
        total_length = len(output_generated)

        directions = [_get_direction(current_point=tra_generated[ii - 1], next_point=tra_generated[ii])
                      for ii in range(1, len(tra_generated))]
        # compute the curliness of the tra
        distances = [distance.euclidean(directions[j - 1], directions[j]) for j in range(1, len(directions))]

        if len(distances) > 0:
            curliness = np.mean(np.array(distances))
            if np.isnan(curliness):
                curliness = 0.0
        else:
            curliness = 0.0

        vector_distances = []
        if len(tra_generated) > 0:
            starting_point = [tra_generated[0].x, tra_generated[0].y]
            # compute distance to further point
            vector_distances = [distance.cityblock(starting_point, [tra_generated[idx].x, tra_generated[idx].y]) for idx
                                in range(1, len(tra_generated) - 1)]

        if len(vector_distances) > 0:
            further_distance_to_point = max(vector_distances)
        else:
            further_distance_to_point = 0

        distance_to_middle_point = 0
        distance_to_end_point = 0
        if len(tra_generated) > 0:
            starting_point = [tra_generated[0].x, tra_generated[0].y]
            # compute distance to middle point
            middle_point = tra_generated[int(len(tra_generated) / 2)]
            distance_to_middle_point = distance.cityblock(starting_point, [middle_point.x, middle_point.y])
            # compute distance to end
            end_point = tra_generated[len(tra_generated) - 1]
            distance_to_end_point = distance.cityblock(starting_point, [end_point.x, end_point.y])

        out, f1, f2, f3 = get_fitness_value(length=total_length, curliness=curliness,
                                            fitness_landscape=fitness_landscape,
                                            further_distance=further_distance_to_point,
                                            point_distance=point_distance)

        all_results_behaviours.append(np.array([total_length, curliness, further_distance_to_point,
                                                distance_to_middle_point, distance_to_end_point]))
        all_tra_generated.append(tra_generated)
        all_result_normal.append(out)
        single_fitness_data.append((f1, f2, f3))

        # check variance good trajecotries
        good_values = [all_results_behaviours[i] for i in range(len(all_result_normal))
                       if all_result_normal[i] >= MAX_TOTAL_FITNESS]
        # for i in range(len(all_result_normal)):
        #     if all_result_normal[i] >= MAX_FITNESS:
        #         good_values.append(all_results_behaviours[i])
        if len(good_values) > 1:
            mean_data = np.mean(np.array(all_results_behaviours), axis=0)
            variance_data_good_values = np.sum(mean_data)
        else:
            variance_data_good_values = 0
        all_variance.append(variance_data_good_values)


    total_trajectories_to_check = number_to_generate
    if penalty_fitness:
        values_of_same_elements = []
        for i in range(total_trajectories_to_check):
            first_tra = [p.vect() for p in all_tra_generated[i]]
            for j in range(i + 1, total_trajectories_to_check):
                second_tra = [p.vect() for p in all_tra_generated[j]]
                tot = [*first_tra, *second_tra]

                equality = pd.DataFrame(np.array(tot).T).T.drop_duplicates(keep=False).as_matrix().shape[0] / len(tot)
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
    else:
        average_converted_distance = 0

    # need to force to go to four directions

    # I have starting point
    # I have ending points
    # all_the_directions = [compute_direction(origin=tra[0], destination=tra[-1]) for tra in all_tra_generated]
    all_the_directions = [compute_direction_angle(origin=tra[0], destination=tra[-1]) for tra in
                          all_tra_generated]

    single_fitness_data.append(average_converted_distance)
    return all_result_normal, all_results_behaviours, all_tra_generated, all_variance, average_converted_distance, single_fitness_data, all_the_directions
