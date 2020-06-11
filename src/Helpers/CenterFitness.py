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
import matplotlib.pyplot as plt

from shapely import geometry
from shapely.geometry import Point, LineString
from shapely.ops import linemerge, unary_union, polygonize


def __create_polygon(origin):
    small = [origin[0][origin[1].vertices, 0], origin[0][origin[1].vertices, 1]]
    a, b = zip(small)
    r = [[a[0][i], b[0][i]] for i in range(len(a[0]))]
    return geometry.Polygon(r)


def __create_polygon_multiply(origin, multiplier):
    small = [origin[0][origin[1].vertices, 0], origin[0][origin[1].vertices, 1]]
    a, b = zip(small)
    r = [[a[0][i] * multiplier, b[0][i]] for i in range(len(a[0]))]
    return geometry.Polygon(r)


def cut_polygon_by_line(polygon, line):
    merged = linemerge([polygon.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)


def __get_graphs(path, name, multiplier):
    poly_a_graph_big = __create_polygon_multiply(
        origin=pickle.load(open("{}/{}_total.pickle".format(path, name), 'rb')), multiplier=multiplier)
    poly_a_graph_small = __create_polygon_multiply(
        origin=pickle.load(open("{}/{}_small.pickle".format(path, name), 'rb')), multiplier=multiplier)
    return poly_a_graph_big, poly_a_graph_small


def __get_figure(path, name, plot, line, reverse="normal", multiplier=1):
    poly_a_graph_big, poly_a_graph_small = __get_graphs(path=path, name=name, multiplier=multiplier)
    centroid_small_area_original = poly_a_graph_small.centroid

    # cut element
    if type(line) is list:
        result_small = cut_polygon_by_line(poly_a_graph_small, line[0])
        result_big = cut_polygon_by_line(poly_a_graph_big, line[0])
        result_small = cut_polygon_by_line(result_small[1], line[1])
        result_big = cut_polygon_by_line(result_big[0], line[1])
    else:
        result_small = cut_polygon_by_line(poly_a_graph_small, line)
        result_big = cut_polygon_by_line(poly_a_graph_big, line)

    # print("small: \n {} \n big: \n {}".format(result_small[0].exterior.xy, result_big[0].exterior.xy))

    plot.plot(*poly_a_graph_small.exterior.xy, color="black")
    plot.plot(*poly_a_graph_big.exterior.xy, color="black")
    if reverse == "normal":
        centroid_small_area_moved = result_small[0].centroid
        plot.plot(*result_small[0].exterior.xy, color="red")
        plot.plot(*result_big[0].exterior.xy, color="green")
        plot.scatter([centroid_small_area_moved.x], [centroid_small_area_moved.y], s=20)
    elif reverse == "reverse":
        centroid_small_area_moved = result_small[1].centroid
        plot.plot(*result_small[1].exterior.xy, color="red")
        plot.plot(*result_big[1].exterior.xy, color="green")
        plot.scatter([centroid_small_area_moved.x], [centroid_small_area_moved.y], s=20)
    elif reverse == "reverse_v1":
        centroid_small_area_moved = result_small[1].centroid
        plot.plot(*result_small[1].exterior.xy, color="red")
        plot.plot(*result_big[0].exterior.xy, color="green")
        plot.scatter([centroid_small_area_moved.x], [centroid_small_area_moved.y], s=20)
    elif reverse == "reverse_v2":
        centroid_small_area_moved = result_small[0].centroid
        plot.plot(*result_small[0].exterior.xy, color="red")
        plot.plot(*result_big[1].exterior.xy, color="green")
        plot.scatter([centroid_small_area_moved.x], [centroid_small_area_moved.y], s=20)

    plot.scatter([centroid_small_area_original.x], [centroid_small_area_original.y], s=20)


    return poly_a_graph_small, poly_a_graph_big, result_small


def return_polygon_fitness_functions(save=False,
                                     path="/Users/alessandrozonta/PycharmProjects/LSTMGen/src/DistanceMetric/"):

    multiplier = 100

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
    line = LineString([(0 * multiplier, 5000), (1.0 * multiplier, 5000)])
    total_area_a, small_area_a, reduced_area_a = __get_figure(path=path, name="curliness_real_length", plot=axes[0, 0],
                                                              line=line, multiplier=multiplier)
    axes[0, 0].set_xlabel("curliness")
    axes[0, 0].set_ylabel("total length")

    line = LineString([(0 * multiplier, 2133.3462229265715), (1.0 * multiplier, 2133.3462229265715)])
    total_area_b, small_area_b, reduced_area_b = __get_figure(path=path, name="curliness_further_distance",
                                                              plot=axes[0, 1], line=line, reverse="reverse",
                                                              multiplier=multiplier)
    axes[0, 1].set_xlabel("curliness")
    axes[0, 1].set_ylabel("distance further point")

    line = LineString([(0, 5000), (4000, 5000)])
    total_area_c, small_area_c, reduced_area_c = __get_figure(path=path, name="further_distance_total_length",
                                                              plot=axes[0, 2], line=line)
    axes[0, 2].set_xlabel("distance further point")
    axes[0, 2].set_ylabel("total length")

    line = LineString([(0, 5000), (3000, 5000)])
    total_area_d, small_area_d, reduced_area_d = __get_figure(path=path, name="distance_end_total_length",
                                                              plot=axes[0, 3], line=line)
    axes[0, 3].set_xlabel("distance to end")
    axes[0, 3].set_ylabel("total length")

    line = LineString([(0, 1969.9053354463813), (3000, 1969.9053354463813)])
    total_area_e, small_area_e, reduced_area_e = __get_figure(path=path, name="distance_end_further_distance",
                                                              plot=axes[0, 4], line=line, reverse="reverse_v1")
    axes[0, 4].set_xlabel("distance to end")
    axes[0, 4].set_ylabel("distance further point")

    line = LineString([(0 * multiplier, 1969.9053354463813), (1.0 * multiplier, 1969.9053354463813)])
    total_area_f, small_area_f, reduced_area_f = __get_figure(path=path, name="curliness_distance_end",
                                                              plot=axes[1, 0], line=line, reverse="reverse_v2",
                                                              multiplier=multiplier)
    axes[1, 0].set_xlabel("curliness")
    axes[1, 0].set_ylabel("distance to end")

    line = LineString([(0 * multiplier, 1289.3410648392198), (1.00 * multiplier, 1289.3410648392198)])
    total_area_g, small_area_g, reduced_area_g = __get_figure(path=path, name="curliness_middle_distance",
                                                              plot=axes[1, 1], line=line, reverse="reverse_v1",
                                                              multiplier=multiplier)
    axes[1, 1].set_xlabel("curliness")
    axes[1, 1].set_ylabel("distance middle point")

    line = LineString([(0, 1289.3410648392198), (4000, 1289.3410648392198)])
    total_area_h, small_area_h, reduced_area_h = __get_figure(path=path, name="further_distance_middle_distance",
                                                              plot=axes[1, 2], line=line, reverse="reverse_v1")
    axes[1, 2].set_xlabel("distance further point")
    axes[1, 2].set_ylabel("distance middle point")

    lines = [LineString([(0, 1289.3410648392198), (3000, 1289.3410648392198)]),
             LineString([(1969.9053354463813, 0), (1969.9053354463813, 3000)])]
    total_area_i, small_area_i, reduced_area_i = __get_figure(path=path, name="distance_to_end_middle_distance",
                                                              plot=axes[1, 3], line=lines)
    axes[1, 3].set_xlabel("distance to end")
    axes[1, 3].set_ylabel("distance middle point")

    line = LineString([(5000, 0), (5000, 4000)])
    total_area_l, small_area_l, reduced_area_l = __get_figure(path=path, name="real_length_middle_distance",
                                                              plot=axes[1, 4], line=line, reverse="reverse")
    axes[1, 4].set_xlabel("total length")
    axes[1, 4].set_ylabel("distance middle point")
    plt.show()
    plt.close()

    vector = [total_area_a, small_area_a, reduced_area_a,
              total_area_b, small_area_b, reduced_area_b,
              total_area_c, small_area_c, reduced_area_c,
              total_area_a, small_area_d, reduced_area_d,
              total_area_e, small_area_e, reduced_area_e,
              total_area_f, small_area_f, reduced_area_f,
              total_area_g, small_area_g, reduced_area_g,
              total_area_h, small_area_h, reduced_area_h,
              total_area_i, small_area_i, reduced_area_i,
              total_area_l, small_area_l, reduced_area_l]

    if save:
        root = os.path.dirname(os.path.abspath(__file__))
        path = root.replace("Helpers", "") + "/Data/"
        with open('{}/fitness_complete_all_behaviours.pickle'.format(path), 'wb') as handle:
            pickle.dump(vector, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print(centroid_a.wkt)
    # aa = Point(0., 0.)
    # print(centroid_a.distance(aa))
    #
    # aa = Point(0., 5000.)
    # print(centroid_a.distance(aa))
    #
    # aa = Point(2., 5000.)
    # print(centroid_a.distance(aa))
    #
    # aa = Point(2., 0.)
    # print(centroid_a.distance(aa))
    #
    # print("-------")
    #
    # print(centroid_b.wkt)
    # aa = Point(0., 0.)
    # print(centroid_b.distance(aa))
    #
    # aa = Point(0., 2500.)
    # print(centroid_b.distance(aa))
    #
    # aa = Point(2., 2500.)
    # print(centroid_b.distance(aa))
    #
    # aa = Point(2., 0.)
    # print(centroid_b.distance(aa))
    #
    # print("-------")
    #
    # print(centroid_c.wkt)
    # aa = Point(0., 0.)
    # print(centroid_c.distance(aa))
    #
    # aa = Point(0., 5000.)
    # print(centroid_c.distance(aa))
    #
    # aa = Point(2500., 5000.)
    # print(centroid_c.distance(aa))
    #
    # aa = Point(2500., 0.)
    # print(centroid_c.distance(aa))


if __name__ == '__main__':
    logger = logging.getLogger("print_tra")
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    return_polygon_fitness_functions(save=True)
