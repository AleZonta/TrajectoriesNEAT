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
import numpy as np
from shapely.geometry import box


class HugeCell(object):
    def __init__(self, min_x, min_y, max_x, max_y, position, number_of_features, min_x_cell, min_y_cell,
                 max_x_cell, max_y_cell):
        self._polygon = box(min_x, min_y, max_x, max_y, ccw=False)
        self.id = "{}-{}".format(position[0], position[1])
        self.matrix = {}
        self._min_x_cell = min_x_cell
        self._max_x_cell = max_x_cell
        self._min_y_cell = min_y_cell
        self._max_y_cell = max_y_cell
        self._indexing = None
        self.index = 0



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
        collector_for_value = self.matrix[a, b, index, 0, self.index]
        return collector_for_value


    def add_value_back_to_matrix(self, data, point):
        name_in_index = self._indexing.get("{}_{}".format(int(point.x), int(point.y)))
        self.matrix[name_in_index] = data

    def add_value_to_matrix(self, data, point):
        name_in_index = "{}_{}".format(int(point.x), int(point.y))
        self.matrix[name_in_index] = data


    def is_inside_cell(self, point):
        """
        Is the point inside the cell (using matrix coordinate)
        :param point: point to check
        :return: True if it is inside, otherwise False
        """
        if self._min_x_cell <= point.x < self._max_x_cell and self._min_y_cell <= point.y < self._max_y_cell:
            return True
        return False

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

    def check_if_on_a_road(self, point):
        """
        check if point is in a road or not
        :param point:
        :return:
        """

        return False if self._indexing[point.x, point.y, 0] == 0 else True
