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
import sys

import os

from haversine import haversine

import numpy as np

from src.Settings.arguments import args

from src.Helpers.Division.CollectionCells import CollectionCells


class SubMatrix(object):
    def __init__(self, log, apf, list_points, values_matrix, save_and_store=True):
        self._log = log
        self._apf = apf
        self._list_points = list_points
        self._list_of_cells = None
        self._match_key_index = None
        self._save_and_store = save_and_store
        self._values_matrix = values_matrix

        name_file = "{}/matrix_id_matrix_mmap.dat".format(args.data_directory)
        self._coordinate_index = np.memmap(name_file, dtype='int8', mode='r', shape=(6159, 6201, 2))

    def divide_into_cells(self, x_division=40, y_division=40, not_normalised=False):
        """
        compute how many cells are required
        matches the id cell with real coordinates
        generate the right number of cell
        adds all the points to the correct cells
        print heatmap showing point per cell
        :param x_division: how many division in x axis
        :param y_division: how many division in y axis
        :return:
        """

        self._match_key_index = {}
        keys = self._list_points
        count_here = 0
        for key in keys:
            self._match_key_index[key] = count_here
            count_here += 1
        # number_of_features = len(keys)

        if self._log is not None:
            self._log.debug("Creation of the cells")
        self._list_of_cells = CollectionCells(matching=self._match_key_index, x_division=x_division,
                                              y_division=y_division, save_and_store=self._save_and_store)

        self._list_of_cells.load_stored_list_cells(name="smaller_version")

        if self._log is not None:
            self._log.debug("Point division loaded from file")

        self._list_of_cells.load_mmap_data(not_normalised=not_normalised)
        #
        # # for performance support
        self._log = None
        self._apf = None
        self._list_points = None

    def get_max_min_matrix(self):
        return self._list_of_cells.max_values, self._list_of_cells.min_values

    def precompute_minimum_distance_and_equation(self, current_position):
        """
        Precompute the distance and the minimum distance from current position using the cell division idea
        :param current_position: position we are now
        :return: float minimum distance to object
        """
        current_cell_id = self._list_of_cells.find_current_cell_from_matrix_coord(point=current_position)

        neighbors_ids = self._list_of_cells.from_id_get_neighbours(current_id=current_cell_id)
        neighbors_ids.append(current_cell_id)

        all_the_distances = []
        for i in range(len(self._match_key_index.keys())):
            all_the_distances.append({"min_value_distace": sys.float_info.max, "equation_precomputed_value": 0})

        for cell in self._list_of_cells.get_all_cells():
            if cell.id in neighbors_ids:
                index_vector = 0
                for key, value in self._match_key_index.items():
                    # get position all the points
                    vector_points = cell.get_list_item(index=value)
                    if len(vector_points) > 0:
                        # compute distances from current poiunt from them all
                        distances = [haversine((self._values_matrix[0][int(current_position.x)],
                                                self._values_matrix[1][int(current_position.y)]),
                                               list(pos.coords)[0]) * 1000
                                     for pos in vector_points]
                        # compute distances squared
                        distances_updated = [1 / (el * el) for el in distances]

                        all_the_distances[index_vector]["min_value_distace"] = \
                            min(all_the_distances[index_vector]["min_value_distace"], min(distances))
                        all_the_distances[index_vector]["equation_precomputed_value"] += sum(distances_updated)
                    index_vector += 1
            else:
                # only one charge per cell
                centroid_cell = cell.get_centroid()[0]
                distance = haversine((self._values_matrix[0][int(current_position.x)],
                                      self._values_matrix[1][int(current_position.y)]), centroid_cell) * 1000
                distance_updated = 1 / (distance * distance)
                index_vector = 0
                for key, value in self._match_key_index.items():
                    number_of_elements = len(cell.get_list_item(index=value))

                    all_the_distances[index_vector]["min_value_distace"] = \
                        min(all_the_distances[index_vector]["min_value_distace"], distance)
                    all_the_distances[index_vector]["equation_precomputed_value"] += (
                            number_of_elements * distance_updated)
                    index_vector += 1
        return {"distances_per_tag": all_the_distances}

    def return_distance_from_point(self, current_position):
        """
        from the current position return the distance to the closest point
        Check the current cell. If there is not such a point, go to the nearest cells. TODO
        :param current_position: current position
        :return: vector of distances per tag
        """
        raw_id_cell = self._coordinate_index[current_position.x][current_position.y]
        id_current_cell = "{}-{}".format(raw_id_cell[0], raw_id_cell[1])
        cell = self._list_of_cells.get_cell_from_id(id=id_current_cell)
        # current_cell_id = self._list_of_cells.find_current_cell_from_matrix_coord(point=current_position)

        # cell = self._list_of_cells.get_cell_from_id(id=current_cell_id)

        vector_distances = [cell.return_value_min_distance(point=[current_position.x, current_position.y],
                                                           index=i) for i in range(len(self._match_key_index.keys()))]
        return vector_distances

    def keep_only_points_on_street(self, points):
        """
        Check if the points provided are on a route
        :param apf: Dataframe describing the routing system
        :param points: list of points to check
        :return: list of points from the input list that are actually on a route
        """
        points_on_street = []
        for p in points:
            # id_current_cell = self._list_of_cells.find_current_cell_from_matrix_coord(point=p)
            # optimised way with mmeap
            raw_id_cell = self._coordinate_index[p.x][p.y]
            id_current_cell = "{}-{}".format(raw_id_cell[0], raw_id_cell[1])
            cell = self._list_of_cells.get_cell_from_id(id=id_current_cell)
            if cell.check_if_on_a_road(point=p):
                points_on_street.append(p)
        return points_on_street

    def verify_if_file_i_have_is_correct(self):
        name_file = "/Users/alessandrozonta/Desktop/cell_data_to_mmap.dat"

        data_input = np.memmap(name_file, dtype='float32', mode='r', shape=(154, 155, 6, 2, 1600))
        data_input = np.nan_to_num(data_input)
        print("before")
        print("max : {}".format(np.max(data_input)))
        print("min : {}".format(np.min(data_input)))

        def scale(X, x_min, x_max):
            nom = (X - X.min()) * (x_max - x_min)
            denom = X.max() - X.min()
            denom = denom + (denom is 0)
            return x_min + nom / denom

        data_input_old = scale(data_input, -1, 1)
        print("after")
        print("max : {}".format(np.max(data_input)))
        print("min : {}".format(np.min(data_input)))

        root = os.path.dirname(os.path.abspath(__file__))
        output_folder = root.replace("Helpers", "Data").replace("Division", "")
        name_file = "{}/cell_data_to_mmap_normalised.dat".format(output_folder)
        data_input_last = np.memmap(name_file, dtype='float32', mode='r', shape=(154, 155, 6, 2, 1600))

        for i in range(100):
            a = np.random.random_integers(0, 153)
            b = np.random.random_integers(0, 154)
            c = np.random.random_integers(0, 5)
            d = np.random.random_integers(0, 1)
            e = np.random.random_integers(0, 1599)
            old = data_input_old[a, b, c, d, e]
            converted = data_input_last[a, b, c, d, e]
            assert old == converted

