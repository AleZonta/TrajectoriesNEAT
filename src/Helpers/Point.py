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
import math


class Point(object):
    """
    Point class, override default Point class
    """
    __slots__ = ['x', 'y']

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def equals(self, p1):
        """
        Check if two points are equal, comparing the coordinates
        :param p1: point to compare with
        :return: True if equal, False if not
        """
        if math.isclose(self.x, p1.x) and math.isclose(self.y, p1.y):
            return True
        return False

    def print(self):
        """
        print to string the point
        :return: string with the point's coordinates
        """
        return "{}, {}".format(self.x, self.y)

    def to_key(self):
        """
        Create a string key with the coordinates of the point
        :return: string key
        """
        return "{}-{}".format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return "{}, {}.".format(self.x, self.y)

    def vect(self):
        """
        Return the point as a vector if its coordinates
        :return: vector coordinates
        """
        return [self.x, self.y]