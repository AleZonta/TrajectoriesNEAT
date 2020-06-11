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
import os

import csv
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import trange


class GenomeMeaning(object):
    """
    Class representing what every position in the genome means

    It loads all the tag and subtag used in the system and it loads also the location of all the objects
    in the map
    """

    def __init__(self, granularity=0, logger=None):
        self._log = logger
        self._types = None
        self.number_typologies = 0
        self.name_typologies = []
        self.name_and_details = {}
        self.name_and_position = None
        self.list_of_names_per_genome = []
        self.link_main_obj_and_small_objs = {}
        self._features = 0
        self._granularity = granularity
        self._order_name_and_position = None
        self.method_mmap = False

    def load_data(self, test, performance=True):
        """
        Load location data

        All the csv present in the Data folder are loaded
        The file in Settings/Phenotype has to contain the sub tag accepted by the system
        The file contains also the name of the files that have to be in the Data folder
        If one file is missing, the system raises an Exception

        If there are different tag in the file that are not in the Phenotype file, the tag is changed into others

        :param test: if test is true, test files are loaded (with fewer elements)
        :return:
        """
        # loading coordinates
        root = os.path.dirname(os.path.abspath(__file__))
        phenotype_file = root.replace("Helpers", "") + "/Data/Phenotype"
        with open(phenotype_file, 'r') as f:
            self._types = json.load(f)
        if self._log is not None:
            self._log.info("Phenotype meaning loaded")

        self.number_typologies = len(self._types["Typology"])

        for el in self._types["Typology"]:
            for k in el.keys():
                self.name_typologies.append(k.lower())
                self.name_and_details.update(
                    {k.lower(): [x.lower() if x.lower() != "others" else "others_" + k.lower() for x in el[k]]})

        if self._log is not None:
            if self._granularity == 0:
                self._log.debug("Typology loaded {}".format(self.name_typologies))
            else:
                self._log.debug("Typology loaded {}".format(self.name_and_details))

            self._log.warning("Code commented for performances, enable it if needed")

        root = os.path.dirname(os.path.abspath(__file__))
        mmap_file = root.replace("Helpers", "") + "Data/name_and_position.dat"
        if os.path.isfile(mmap_file):
            if self._log is not None:
                self._log.debug("mmap file exist, loading it")

            self.name_and_position = np.memmap(mmap_file, dtype='float32', mode='r', shape=(56, 938737, 2))

            with open('{}Data/order_on_mmap.pickle'.format(root.replace("Helpers", "")), 'rb') as handle:
                self._order_name_and_position = pickle.load(handle)

            with open('{}Data/name_typology.pickle'.format(root.replace("Helpers", "")), 'rb') as handle:
                self.name_typologies = pickle.load(handle)
            self.method_mmap = True
        else:
            self.name_and_position = {}
            for name in self.name_typologies:
                lowercase_name = name.lower()
                root = os.path.dirname(os.path.abspath(__file__))
                data_file = root.replace("Loaders", "") + "/Data/"
                if test:
                    name_file = data_file + lowercase_name + "_test.csv"
                else:
                    name_file = data_file + lowercase_name + ".csv"
                my_file = Path(name_file)
                if my_file.is_file():

                    list_word_accepted = self.name_and_details[lowercase_name]
                    list_word_accepted_fixed = []
                    for word in list_word_accepted:
                        if word == "others":
                            word = "others_" + name
                        list_word_accepted_fixed.append(word)
                        self.link_main_obj_and_small_objs.update({word: name})
                    self.list_of_names_per_genome.extend(list_word_accepted_fixed)

                    # file exist
                    with open(name_file) as csvfile:
                        reader = csv.DictReader(csvfile)
                        dic = {}

                        for row in reader:
                            name_element = row["names"].lower()

                            if name_element in list_word_accepted:
                                okay_id = name_element
                            else:
                                okay_id = "others_" + name

                            if row["x"] != "":
                                dic.setdefault(okay_id, []).append((float(row["x"]), float(row["y"])))

                        self.name_and_position.update({lowercase_name: dic})
                else:
                    if self._log is not None:
                        self._log.debug("File {} not present in folder. Please provide it".format(name_file))
                    raise ValueError("File {} not present in folder. Please provide it".format(name_file))

        if self._log is not None:
            self._log.info("Coordinates loaded!")
        self._log = None

        if performance:
            self._types = None
            self.name_and_details = None

    def _save_name_and_position_to_mmap(self):
        # save mmap name_and_position
        array = np.zeros((56, 938737, 2), dtype='float32')
        element_position = 0
        order = {}
        for key, value in self.name_and_position.items():
            for sub_key, sub_values in value.items():
                # vector_positions = self.name_and_position[key][sub_key]
                order.update({element_position: sub_key})
                for i in trange(len(sub_values)):
                    array[element_position, i] = np.array(sub_values[i], dtype='float32')

                element_position += 1

        with open('order_on_mmap.pickle', 'wb') as handle:
            pickle.dump(order, handle, protocol=pickle.HIGHEST_PROTOCOL)

        filename = "name_and_position.dat"
        fp = np.memmap(filename, dtype='float32', mode='w+', shape=(56, 938737, 2))
        fp[:] = array[:]
        del fp

        self._log.info("Saved mmap!")

    def load_fake_data(self):
        """
        Load fake data in order to understand if the APF and A* works
        It loads a new setting with positions from a file
        :return:
        """

        self.name_typologies = ["fake"]
        with open("/Users/alessandrozonta/PycharmProjects/GTEA/Data/coordinates_fake_data.txt") as f:
            lines = f.readlines()
        list_of_points = []
        for line in lines:
            coordinates = line.split(",")
            list_of_points.append((float(coordinates[0]), float(coordinates[1].replace("\n", ""))))
        current_dict = {"others_fake": list_of_points}

        self.name_and_position = {"fake": current_dict}

        # erase all the other staff

    def get_length_genome_only_objects(self):
        """
        I think it is obvious what this method does.
        If it is not,to complain, send an email to me.
        :return: read the name of the function to understand
        """
        if self._granularity == 0:
            return self.number_typologies
        else:
            return len(self.list_of_names_per_genome)

    def from_genome_to_value_describing_object(self, name_object):
        """
        Genome is constructed using the number of elements loaded in this class.

        genome is long len(self.list_of_names_per_genome) + other details.

        This function receives as input the name of the object I need the charge for.
        Iterating among the self.name_and_details, it finds the index corresponding to
        the name and it returns the index for the genome

        :param name_object: object to find the index
        :return: int value representing index in the genome corresponding to the name
        """
        if self._granularity == 0:
            return self.name_typologies.index(name_object)
            # return self.name_typologies.index(self.link_main_obj_and_small_objs[name_object])
        else:
            return self.list_of_names_per_genome.index(name_object)

    def get_number_features(self):
        """
        Return the number of features needed for the network
        With feature now we count the number of total objects presents in the area.
        TODO check if the number is too damn high. Otherwise try some combination
        :return: number of features
        """
        if self._features == 0:
            if self._granularity == 0:
                self._features = self.number_typologies
            else:
                self._features = len(self._order_name_and_position.keys())
                # typology = self.name_and_position.keys()
                # for key in typology:
                #     self._features += len(self.name_and_position[key].keys())
        return self._features

    def get_points_inside_area(self, limits):
        """
        given the limits return the POI inside the area
        :param limits: list of limits
        :return: POIs in the area
        """
        max_xs = limits[0]
        min_xs = limits[1]
        max_ys = limits[2]
        min_ys = limits[3]

        final_result = {}
        for t in self.name_typologies:
            areas = self.name_and_position[t]

            list_of_points_here = []
            for key, value in areas.items():
                for p in value:
                    if min_xs < p[1] < max_xs and min_ys < p[0] < max_ys:
                        list_of_points_here.append(p)
            final_result[t] = list_of_points_here
        return final_result

