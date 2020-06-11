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
import logging
import os
import pickle

from scipy.spatial import distance
from shapely.geometry import Point

VAL_NO_DATA = -1
LIMIT_TIMESTEPS = 5000
MAX_FITNESS = 200
MAX_TOTAL_FITNESS = MAX_FITNESS * 3


def _get_max_fitness_possible(fitness_definition):
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
    centroid = internal.centroid
    d = -distance.euclidean([centroid.x, centroid.y], [point.x, point.y])
    max_value = -5000
    if d < max_value:
        d = max_value
    fitness_value = convert(old_max=0, old_min=max_value, new_max=MAX_FITNESS, new_min=new_min, old_value=d)
    return fitness_value, d


def _get_combination_two_fitness_curliness_length(point, internal_normal, internal_special, external):
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
    point = Point(curliness * 100, length)

    if 0 in point_distance:
        value_from_curliness_length = \
            _get_combination_two_fitness_curliness_length(point=point, internal_normal=fitness_landscape[1],
                                                          external=fitness_landscape[0],
                                                          internal_special=fitness_landscape[6])
        # value_from_curliness_length, _ = _get_distance_to_center(point=point, internal=fitness_landscape[6])
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
        # value_from_curliness_distance, _ = _get_distance_to_center(point=point, internal=fitness_landscape[8])
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
        # value_from_distance_length, _ = _get_distance_to_center(point=point, internal=fitness_landscape[10])
    else:
        value_from_distance_length = _get_fitness_length_curliness(point=point, external=fitness_landscape[4],
                                                                   internal=fitness_landscape[5])
        value_from_distance_length = convert(old_max=0, old_min=-150, new_max=MAX_FITNESS, new_min=-300,
                                              old_value=value_from_distance_length)

    return value_from_distance_length + value_from_curliness_length + value_from_curliness_distance, \
           value_from_distance_length, value_from_curliness_length, value_from_curliness_distance


def convert(old_max, old_min, new_max, new_min, old_value):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    return new_value


def get_next_point(current_point, direction):
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

    root = os.path.dirname(os.path.abspath(__file__))
    path = root.replace("Network/Fitness", "") + "/Data/"
    fitness_landscape = pickle.load(open("{}/3d_fitness_in_2d_with_limitation.pickle".format(path), 'rb'))


    def check(length, curliness, further_distance):
        logger.debug("{} {} {} -> {}".format(length, curliness, further_distance,
                                             get_fitness_value(length=length, curliness=curliness,
                                                               fitness_landscape=fitness_landscape,
                                                               further_distance=further_distance)))


    def check_point(length, curliness, further_distance):
        fitness, distancess = _get_distance_to_center(point=Point(curliness, length), internal=fitness_landscape[6])

        fitness1, distance1 = _get_distance_to_center(point=Point(curliness, further_distance),
                                                      internal=fitness_landscape[8])

        fitness2, distance2 = _get_distance_to_center(point=Point(further_distance, length),
                                                      internal=fitness_landscape[10])
        return fitness, fitness1, fitness2, distancess, distance1, distance2


    #
    # logger.debug("{} -> {}".format(check_point(length=2600, curliness=0, further_distance=1000)[0], check_point(length=2600, curliness=0, further_distance=1000)[3]))
    # # # # print(check_point(length=1000, curliness=77, further_distance=1000))
    # # # # print(check_point(length=1000, curliness=76, further_distance=1000))
    # # # # logger.debug(check_point(length=500, curliness=0, further_distance=1000)[0])
    # # # # logger.debug(check_point(length=500, curliness=78, further_distance=1000)[0])
    # logger.debug("{} -> {}".format(check_point(length=2600, curliness=80, further_distance=1000)[0], check_point(length=2600, curliness=80, further_distance=1000)[3]))

    # curliness_min = 73
    # curliness_max = 85
    # total_length_min = 0
    # total_length_max = 500
    # distance_further_point_min = 0
    # distance_further_point_max = 1000
    #
    # matrix_a = np.zeros((curliness_max, total_length_max))
    # matrix_b = np.zeros((curliness_max, distance_further_point_max))
    # matrix_c = np.zeros((distance_further_point_max, total_length_max))
    #
    # matrix_ad = np.zeros((distance_further_point_max, total_length_max))
    # matrix_bd = np.zeros((distance_further_point_max, total_length_max))
    # matrix_cd = np.zeros((distance_further_point_max, total_length_max))
    #
    # for i in range(curliness_min, curliness_max):
    #     for j in trange(total_length_min, total_length_max, desc="elaboration curliness {}".format(i)):
    #         for k in range(distance_further_point_min, distance_further_point_max):
    #             # fitness0, fitness1, fitness2 = get_fitness_value(length=j, curliness=i, further_distance=k, fitness_landscape=fitness_landscape, point_distance=[])
    #
    #             fitness0, fitness1, fitness2, distance0, distance1, distance2 = check_point(length=j, curliness=i,
    #                                                                                         further_distance=k)
    #
    #             # print("{}, {} -> {}".format(i, j, fitness))
    #             matrix_a[i, j] = fitness0
    #             matrix_b[i, k] = fitness1
    #             matrix_c[k, i] = fitness2
    #
    #             matrix_ad[i, j] = distance0
    #             matrix_bd[i, k] = distance1
    #             matrix_cd[k, i] = distance2
    # np.save("small_matrix_a_dncistae_total_in_one_different_fitness", matrix_a)
    # np.save("small_matrix_b_dncistae_total_in_one_different_fitness", matrix_b)
    # np.save("small_matrix_c_dncistae_total_in_one_different_fitness", matrix_c)
    # np.save("small_a_dncistae_total_in_one_different_fitness", matrix_ad)
    # np.save("small_b_dncistae_total_in_one_different_fitness", matrix_bd)
    # np.save("small_c_dncistae_total_in_one_different_fitness", matrix_cd)

    # fig = plt.figure()
    # plt.subplot(3, 1, 1)

    # files_to_find = "/Users/alessandrozonta/Desktop/matrix_a_*.npy"
    # txts = sorted_nicely(glob.glob(files_to_find))

    # matrix_here = np.zeros((150, 5000))
    # list_of_matrix = []
    # for i in range(len(txts)):
    #     if i == 0:
    #         matrix_here = np.load(txts[i], allow_pickle=True)

    # distance = False
    # if distance:
    #     name = "matrix_normal_fitness_a_"
    # else:
    #     name = "matrix_normal_fitness_a_"
    # #     else:
    # a = np.load("/Users/alessandrozonta/Desktop/{}500.npy".format(name), allow_pickle=True)
    # # a = np.delete(a, np.arange(120, 1000), 0)
    # b = np.load("/Users/alessandrozonta/Desktop/{}1000.npy".format(name), allow_pickle=True)
    # b = np.delete(b, np.arange(0, 500), 1)
    # # b = np.delete(b, np.arange(120, 1000), 0)
    #
    # c = np.load("/Users/alessandrozonta/Desktop/{}1500.npy".format(name), allow_pickle=True)
    # c = np.delete(c, np.arange(0, 1000), 1)
    # # c = np.delete(c, np.arange(120, 1000), 0)
    #
    # d = np.load("/Users/alessandrozonta/Desktop/{}2000.npy".format(name), allow_pickle=True)
    # d = np.delete(d, np.arange(0, 1500), 1)
    # # d = np.delete(d, np.arange(120, 1000), 0)
    #
    # e = np.load("/Users/alessandrozonta/Desktop/{}2500.npy".format(name), allow_pickle=True)
    # e = np.delete(e, np.arange(0, 2000), 1)
    # # e = np.delete(e, np.arange(120, 1000), 0)
    #
    # f = np.load("/Users/alessandrozonta/Desktop/{}3000.npy".format(name), allow_pickle=True)
    # f = np.delete(f, np.arange(0, 2500), 1)
    # # f = np.delete(f, np.arange(120, 1000), 0)
    #
    # # g = np.load("/Users/alessandrozonta/Desktop/{}2900.npy".format(name), allow_pickle=True)
    # # g = np.delete(g, np.arange(0, 2700), 1)
    # # g = np.delete(g, np.arange(120, 1000), 0)
    # #
    # # h = np.load("/Users/alessandrozonta/Desktop/{}3100.npy".format(name), allow_pickle=True)
    # # h = np.delete(h, np.arange(0, 2900), 1)
    # # h = np.delete(h, np.arange(120, 1000), 0)
    # #
    # # i = np.load("/Users/alessandrozonta/Desktop/{}3300.npy".format(name), allow_pickle=True)
    # # i = np.delete(i, np.arange(0, 3100), 1)
    # # i = np.delete(i, np.arange(120, 1000), 0)
    # #
    # # l = np.load("/Users/alessandrozonta/Desktop/{}3500.npy".format(name), allow_pickle=True)
    # # l = np.delete(l, np.arange(0, 3300), 1)
    # # l = np.delete(l, np.arange(120, 1000), 0)
    # #
    # # m = np.load("/Users/alessandrozonta/Desktop/{}3700.npy".format(name), allow_pickle=True)
    # # m = np.delete(m, np.arange(0, 3500), 1)
    # # m = np.delete(m, np.arange(120, 1000), 0)
    # #
    # # n = np.load("/Users/alessandrozonta/Desktop/{}3900.npy".format(name), allow_pickle=True)
    # # n = np.delete(n, np.arange(0, 3700), 1)
    # # n = np.delete(n, np.arange(120, 1000), 0)
    # #
    # # o = np.load("/Users/alessandrozonta/Desktop/{}4100.npy".format(name), allow_pickle=True)
    # # o = np.delete(o, np.arange(0, 3900), 1)
    # # o = np.delete(o, np.arange(120, 1000), 0)
    # #
    # # p = np.load("/Users/alessandrozonta/Desktop/{}4300.npy".format(name), allow_pickle=True)
    # # p = np.delete(p, np.arange(0, 4100), 1)
    # # p = np.delete(p, np.arange(120, 1000), 0)
    # #
    # # q = np.load("/Users/alessandrozonta/Desktop/{}4500.npy".format(name), allow_pickle=True)
    # # q = np.delete(q, np.arange(0, 4300), 1)
    # # q = np.delete(q, np.arange(120, 1000), 0)
    #
    # # print(-distance.euclidean([78.32,2643.44], [0,2600]))
    # # print(-distance.euclidean([78.32,2643.44], [80,2600]))
    # # # # # load_here = np.log(load_here)
    # # # #     # list_of_matrix.append(load_here)
    # # # #     # matrix_here[0:load_here.shape[0], 0: load_here.shape[1]] = load_here
    # # # #
    # # # #     # matrix_here = np.append(matrix_here, load_here, axis=0)
    # # deleted_area_after = np.delete(load_here, np.arange(0, 100), 0)
    # # #
    # # load_here = np.load("matrix_a_values_fitness_values.npy", allow_pickle=True)
    # # deleted_area_center = np.delete(load_here, np.arange(0, 60), 0)
    # # #
    # # load_here = np.load("matrix_a_values_fitness_values_before.npy", allow_pickle=True)
    # # deleted_area_beginning = np.delete(load_here, np.arange(0, 40), 0)
    # # #
    # # #
    # # part_a = np.load("a_dncistae_total_in_one.npy", allow_pickle=True)
    # # # part_b = np.load("/Users/alessandrozonta/Desktop/matrix_a_fitness_value_up_two.npy", allow_pickle=True)
    # # # part_c = np.load("/Users/alessandrozonta/Desktop/matrix_a_value_2.0.npy", allow_pickle=True)
    # # part_a = np.delete(part_a, np.arange(0, 70), 0)
    # # # part_c = np.delete(part_c, np.arange(0, 40), 0)
    # # # part_c = np.delete(part_c, np.arange(0, 1200), 1)
    # # # part_b = np.delete(part_b, np.arange(0, 40), 0)
    # # # part_b = np.delete(part_b, np.arange(0, 2500), 1)
    # # # total = np.concatenate((part_a, part_c, part_b), axis=1)
    # # #
    # # # np.save("total_normal_fitness_no_border", total)
    # #
    # # # deleted_area_up = np.delete(deleted_area_up, np.arange(0, 40), 0)
    # # # deleted_area_up = deleted_area_up.transpose()
    # # #
    # # #
    # # # load_here_a = np.load("matrix_a_up_two.npy", allow_pickle=True)
    # # print("A")
    # # #
    # # # #
    # # # #
    # total = np.concatenate((a, b, c, d, e, f), axis=1)
    # # total = np.transpose(total)
    # # # # #
    # # # # # # print(np.max(deleted_area))
    # # dataframe_apf = pd.DataFrame.from_records(total)
    # # dataframe_apf = dataframe_apf.transpose()
    # # dataframe_apf = dataframe_apf.iloc[2611:2667, 65:95]
    # # sns.heatmap(dataframe_apf, fmt="d")
    # # plt.gca().invert_yaxis()
    # # plt.show()
    #
    # dataframe_apf = pd.DataFrame.from_records(total)
    # dataframe_apf = dataframe_apf.transpose()
    # sns.heatmap(dataframe_apf, fmt="d")
    # plt.gca().invert_yaxis()
    # plt.show()
    # # line_a = dataframe_apf.iloc[2643].tolist()
    # # line_b = dataframe_apf.iloc[2645].tolist()
    # # line_c = dataframe_apf.iloc[2647].tolist()
    # # line_d = dataframe_apf.iloc[2649].tolist()
    # # line_e = dataframe_apf.iloc[2651].tolist()
    # # line_f = dataframe_apf.iloc[2641].tolist()
    # # line_g = dataframe_apf.iloc[2639].tolist()
    # # line_h = dataframe_apf.iloc[2637].tolist()
    # # line_i = dataframe_apf.iloc[2635].tolist()
    # # line_l = dataframe_apf.iloc[2633].tolist()
    # # plt.plot(np.arange(0, len(line_a)), line_a)
    # # plt.plot(np.arange(0, len(line_b)), line_b)
    # # plt.plot(np.arange(0, len(line_c)), line_c)
    # # plt.plot(np.arange(0, len(line_d)), line_d)
    # # plt.plot(np.arange(0, len(line_e)), line_e)
    # # plt.plot(np.arange(0, len(line_f)), line_f)
    # # plt.plot(np.arange(0, len(line_g)), line_g)
    # # plt.plot(np.arange(0, len(line_h)), line_h)
    # # plt.plot(np.arange(0, len(line_i)), line_i)
    # # plt.plot(np.arange(0, len(line_l)), line_l)
    # # plt.legend(["2643", "2645", "2647", "2649", "2651", "2641", "2639", "2637", "2635", "2633"])
    # # plt.show()
    # from mpl_toolkits import mplot3d
    #
    # # fig = plt.figure()
    # # ax = plt.axes(projection='3d')
    # #
    # # x = np.arange(0, total.shape[0])
    # # y = np.arange(0, total.shape[1])
    # #
    # # X, Y = np.meshgrid(y, x)
    # # Z = total
    # #
    # # ax.contour3D(X, Y, Z, 50, cmap='binary')
    # # ax.set_ylabel('curliness')
    # # ax.set_xlabel('total length')
    # # ax.set_zlabel('fitness')
    # #
    # #
    # #
    # # plt.show()
    #
    # # #
    # # # # plt.subplot(3, 1, 2)
    # # #
    # # files_to_find = "/Users/alessandrozonta/Desktop/matrix_b_*.npy"
    # # txts = sorted_nicely(glob.glob(files_to_find))[::-1]
    # #
    # # matrix_here = np.zeros((150, 3000))
    # # for i in range(len(txts)):
    # #     # if i == 0:
    # #     #     matrix_here = np.load(txts[i], allow_pickle=True)
    # #     # else:
    # #     load_here = np.load(txts[i], allow_pickle=True)
    # #     matrix_here[0:load_here.shape[0], 0: load_here.shape[1]] = load_here
    # #
    # #     # matrix_here = np.append(matrix_here, load_here, axis=0)
    # # dataframe_apf = pd.DataFrame.from_records(matrix_here)
    # # dataframe_apf = dataframe_apf.transpose()
    # # sns.heatmap(dataframe_apf, fmt="d")
    # #
    # # plt.subplot(3, 1, 3)
    # # files_to_find = "/Users/alessandrozonta/Desktop/matrix_c_*.npy"
    # # txts = sorted_nicely(glob.glob(files_to_find))[::-1]
    # #
    # # matrix_here = np.zeros((3000, 5000))
    # # for i in range(len(txts)):
    # #     # if i == 0:
    # #     #     matrix_here = np.load(txts[i], allow_pickle=True)
    # #     # else:
    # #     load_here = np.load(txts[i], allow_pickle=True)
    # #     matrix_here[0:load_here.shape[0], 0: load_here.shape[1]] = load_here
    # #
    # #     # matrix_here = np.append(matrix_here, load_here, axis=0)
    # # dataframe_apf = pd.DataFrame.from_records(matrix_here)
    # # dataframe_apf = dataframe_apf.transpose()
    # # sns.heatmap(dataframe_apf, fmt="d")
    #
    # # plt.show()
    #
    # distance = False
    # if distance:
    #     name = "matrix_normal_fitness_b_"
    # else:
    #     name = "matrix_normal_fitness_b_"
    # #     else:
    # a = np.load("/Users/alessandrozonta/Desktop/{}500.npy".format(name), allow_pickle=True)
    #
    #
    # dataframe_apf = pd.DataFrame.from_records(a)
    # dataframe_apf = dataframe_apf.transpose()
    # sns.heatmap(dataframe_apf, fmt="d")
    # plt.gca().invert_yaxis()
    # plt.show()

    # distance = False
    # if distance:
    #     name = "matrix_normal_fitness_c_"
    # else:
    #     name = "matrix_normal_fitness_c_"
    # #     else:
    # a = np.load("/Users/alessandrozonta/Desktop/{}500.npy".format(name), allow_pickle=True)
    # a = np.delete(a, np.arange(120, 1000), 1)
    #
    # dataframe_apf = pd.DataFrame.from_records(a)
    # sns.heatmap(dataframe_apf, fmt="d")
    # plt.gca().invert_yaxis()
    # plt.gca().invert_xaxis()
    # plt.show()

    logger.info("Ending script")
