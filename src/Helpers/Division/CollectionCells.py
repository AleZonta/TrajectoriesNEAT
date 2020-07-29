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
import warnings

from src.Settings.arguments import args

warnings.simplefilter(action='ignore', category=FutureWarning)
import os

import pickle



class CollectionCells(object):
    def __init__(self, matching, x_division, y_division, save_and_store=True):
        self._list_cells = {}

        self._x_division = x_division
        self._y_division = y_division
        self._save_and_store = save_and_store
        self.max_values = None
        self.min_values = None



    def store_current_list_cells(self, name="division_cell_list"):
        """
        Store current division of cell in order not to load it every experiments
        :param name: name to save
        :return:
        """
        if self._save_and_store:
            # if self._list_cells[0].has_point():
            # i have something in the cells
            root = os.path.dirname(os.path.abspath(__file__))
            output_folder = root.replace("Helpers/Division", "Data")
            name_file = "{}_{}_by_{}".format(name, self._x_division, self._y_division)
            pickle.dump(self._list_cells, open("{}/{}.pickle".format(output_folder, name_file), 'wb'))

    def load_stored_list_cells(self, name="division_cell_list"):
        """
        If the file is present, load the data from store instead that from file
        :param name: name to load
        :return: return True if loaded is complete, otherwise False
        """
        if not self._save_and_store:
            return False
        # output_folder = "/Volumes/Cthulhu/MacDataPycharmProjects/GTEA/Data/Division/"
        name_file = "{}_{}_by_{}_dict_version".format(name, self._x_division, self._y_division)
        if os.path.isfile("{}/{}.pickle".format(args.data_directory, name_file)):
            self._list_cells = pickle.load(open("{}/{}.pickle".format(args.data_directory, name_file), 'rb'))
            return True
        return False


    def get_all_cells(self):
        """
        return list of all the cells
        :return: list of cells
        """
        return self._list_cells


    def find_current_cell_from_matrix_coord(self, point):
        """
        return id cell given point in matrix coordinate
        :param point: point inside cell
        :return: id of the cell
        """
        for cell in self._list_cells:
            if cell.is_inside_cell(point=point):
                return cell.id

    def from_id_get_neighbours(self, current_id):
        """
        return id of the heighbours of the current cell
        :return:
        """
        neighbours_ids = []
        int_i_id = int(current_id.split("-")[0])
        int_j_id = int(current_id.split("-")[1])
        if int_i_id - 1 >= 0:
            neighbours_ids.append("{}-{}".format(int_i_id - 1, int_j_id))
        if int_i_id - 1 >= 0 and int_j_id + 1 <= self._y_division:
            neighbours_ids.append("{}-{}".format(int_i_id - 1, int_j_id + 1))
        if int_j_id + 1 <= self._y_division:
            neighbours_ids.append("{}-{}".format(int_i_id, int_j_id + 1))
        if int_i_id + 1 <= self._x_division and int_j_id + 1 <= self._y_division:
            neighbours_ids.append("{}-{}".format(int_i_id + 1, int_j_id + 1))
        if int_i_id + 1 <= self._x_division:
            neighbours_ids.append("{}-{}".format(int_i_id + 1, int_j_id))
        if int_i_id + 1 <= self._x_division and int_j_id - 1 >= 0:
            neighbours_ids.append("{}-{}".format(int_i_id + 1, int_j_id - 1))
        if int_j_id - 1 >= 0:
            neighbours_ids.append("{}-{}".format(int_i_id, int_j_id - 1))
        if int_i_id - 1 >= 0 and int_j_id - 1 >= 0:
            neighbours_ids.append("{}-{}".format(int_i_id - 1, int_j_id - 1))
        return neighbours_ids

    def get_cell_from_id(self, id):
        """
        return the cell with the right id
        :param id: id to find
        :return: cell
        """
        return self._list_cells[id]
        # for cell in self._list_cells:
        #     if cell.id == id:
        #         return cell

    def load_mmap_data(self, not_normalised=False):
        if not_normalised:
            name_file = "{}/cell_data_to_mmap.dat".format(args.data_directory)

            data_input = np.memmap(name_file, dtype='float32', mode='r', shape=(154, 155, 6, 2, 1600))
            # data_input = np.nan_to_num(data_input)
            self.min_values = 0.0
            self.max_values = 62230.51
            # data_input = scale(data_input, -1, 1)
        else:
            name_file = "{}/cell_data_to_mmap_normalised.dat".format(args.data_directory)
            data_input = np.memmap(name_file, dtype='float32', mode='r', shape=(154, 155, 6, 2, 1600))
        count = 0
        for k, cell in self._list_cells.items():
            cell.matrix = data_input
            cell.index = count
            count += 1

        name_file = "{}/indexing_fast.dat".format(args.data_directory)
        ndexing = np.memmap(name_file, dtype='int16', mode='r', shape=(6159, 6083, 2))
        for k, cell in self._list_cells.items():
            cell._indexing = ndexing
