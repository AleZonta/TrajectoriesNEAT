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
from shapely.geometry import box


class HugeCell(object):
    def __init__(self, min_x, min_y, max_x, max_y, position, number_of_features, min_x_cell, min_y_cell,
                 max_x_cell, max_y_cell):
        self._polygon = box(min_x, min_y, max_x, max_y, ccw=False)
        # self._points = []
        # for i in range(number_of_features):
        #     self._points.append([])
        # self.position = position
        self.id = "{}-{}".format(position[0], position[1])
        self.matrix = {}
        self._min_x_cell = min_x_cell
        self._max_x_cell = max_x_cell
        self._min_y_cell = min_y_cell
        self._max_y_cell = max_y_cell
        self._indexing = None
        self.index = 0

    # def define_indexing(self, apf):
    #     """
    #     index the real coordinate of the cell with the relative coordinate
    #     match only point in a road
    #     :param apf: apf needed to match the strings
    #     :return:
    #     """
    #     self._indexing = {}
    #     real_x = 0
    #
    #     for x in range(self._min_x_cell, self._max_x_cell):
    #         real_y = 0
    #         for y in range(self._min_y_cell, self._max_y_cell):
    #             if apf.iloc[x, y] > 0:
    #                 self._indexing["{}_{}".format(x, y)] = "{}_{}".format(real_x, real_y)
    #             real_y += 1
    #         real_x += 1

    # def return_value_attraction(self, point, index):
    #     """
    #     return the value of the distances from the current location
    #     :param point: current location, vector two position
    #     :param index: position
    #     :return: float value
    #     """
    #     name_in_index = self._indexing.get("{}_{}".format(point[0], point[1]))
    #     two_values = name_in_index.split("_")
    #     a = int(two_values[0])
    #     b = int(two_values[1])
    #     collector_for_value = self.matrix[a, b, index, 1, self.index]
    #     return collector_for_value
    #
    #     # name_in_index = self._indexing.get("{}_{}".format(point[0], point[1]))
    #     # if name_in_index is not None:
    #     #     collector_for_value = self.matrix.get(name_in_index)
    #     #     if collector_for_value is not None:
    #     #         value = collector_for_value["distances_per_tag"][index]["equation_precomputed_value"]
    #     #         return value
    #     #     else:
    #     #         return None
    #     # else:
        #     raise ValueError("Index not present")

    def return_value_min_distance(self, point, index):
        """
        return the value of the minimum distances from the current location
        :param point: current location, vector two position
        :param index: position
        :return: float value
        """
        values = self._indexing[point[0], point[1]]
        a = values[0]
        b = values[1]
        # name_in_index = self._indexing.get("{}_{}".format(point[0], point[1]))
        # two_values = name_in_index.split("_")
        # a = int(two_values[0])
        # b = int(two_values[1])
        collector_for_value = self.matrix[a, b, index, 0, self.index]
        return collector_for_value

        # return self.matrix[self._indexing["{}_{}".format(point[0], point[1])]]["distances_per_tag"][index][
        #     "min_value_distace"]

    def add_value_back_to_matrix(self, data, point):
        name_in_index = self._indexing.get("{}_{}".format(int(point.x), int(point.y)))
        self.matrix[name_in_index] = data

    def add_value_to_matrix(self, data, point):
        name_in_index = "{}_{}".format(int(point.x), int(point.y))
        self.matrix[name_in_index] = data

    # def has_point(self):
    #     """
    #     does the cell have points
    #     :return: true if yes otherwise False
    #     """
    #     if len(self._points) > 0:
    #         return True
    #     else:
    #         return False

    # def is_inside(self, point):
    #     """
    #     check if point is inside the polygon
    #     :param point: point to check
    #     :return:
    #     """
    #     return self._polygon.contains(point)

    def is_inside_cell(self, point):
        """
        Is the point inside the cell (using matrix coordinate)
        :param point: point to check
        :return: True if it is inside, otherwise False
        """
        # if point.x == 3905 and point.y == 3410:
        #     print("WUT")
        if self._min_x_cell <= point.x < self._max_x_cell and self._min_y_cell <= point.y < self._max_y_cell:
            return True
        return False

    # def add_point(self, point, index):
    #     """
    #     add point to the correct collection
    #     :param point: point to add
    #     :param index: index right collection
    #     :return:
    #     """
    #     self._points[index].append(point)

    # def return_number_elements(self):
    #     """
    #     return number of points in this cell
    #     :return: integer number
    #     """
    #     total_len = 0
    #     for sublist in self._points:
    #         total_len += len(sublist)
    #     return total_len

    # def get_polygon_coords(self):
    #     """
    #     return coordinates of the cells
    #     :return: list of coordinates
    #     """
    #     return self._polygon.exterior.coords

    def get_centroid(self):
        """
        get centroid cell
        :return: list coordinate centroid
        """
        return list(self._polygon.centroid.coords)

    def get_list_item(self, index):
        """
        return list of element corresponding to index
        :param index: index of the vector to return
        :return: list of position
        """
        return self._points[index]

    def get_border_matrix(self):
        """
        return list border of the cell
        :return:
        """
        return [self._min_x_cell, self._max_x_cell, self._min_y_cell, self._max_y_cell]

    # def elaborate_points_on_road(self, apf):
    #     """
    #     keep in memory the points that are on a road in order to be checked
    #     :return:
    #     """
    #     for x in range(self._min_x_cell, self._max_x_cell):
    #         for y in range(self._min_y_cell, self._max_y_cell):
    #             if apf.iloc[x, y] > 0:
    #                 self._on_road["{}_{}".format(x, y)] = True

    def check_if_on_a_road(self, point):
        """
        check if point is in a road or not
        :param point:
        :return:
        """
        # values =
        # name_in_index = self._indexing.get("{}_{}".format(point.x, point.y))
        return False if self._indexing[point.x, point.y, 0] == 0 else True
        # return True
        # two_values = name_in_index.split("_")
        # a = int(two_values[0])
        # b = int(two_values[1])
        # collector_for_value = self.matrix[a, b, 0, 0, self.index]
        # if collector_for_value is np.nan:
        #     return False
        # return True
        # if "{}_{}".format(point.x, point.y) in self._on_road.keys():
        #     return True
        # return False

    # def remove_points(self):
    #     self._points = None
