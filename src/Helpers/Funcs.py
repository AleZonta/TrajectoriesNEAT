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
import re

from src.Helpers.Point import Point


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def list_neighbours(x_value, y_value, apf, list_already_visited=None):
    """
    Return all the neighbours cells.
    If the neighbour is already visited, it is removed from the return list
    :param x_value: x value starting point
    :param y_value: y value starting point
    :param list_already_visited: list of point already visited
    :return: list of neighbours
    """
    xx = [x_value - 1, x_value, x_value + 1, x_value + 1, x_value + 1, x_value, x_value - 1, x_value - 1]
    yy = [y_value + 1, y_value + 1, y_value + 1, y_value, y_value - 1, y_value - 1, y_value - 1, y_value]

    # remove negative values and values outside bounds apf
    to_erase_x = []
    for i in range(len(xx) - 1, -1, -1):
        if xx[i] < 0:
            to_erase_x.append(i)
        if xx[i] >= apf[0]:
            to_erase_x.append(i)
    to_erase_y = []
    for i in range(len(yy) - 1, -1, -1):
        if yy[i] < 0:
            to_erase_y.append(i)
        if yy[i] >= apf[1]:
            to_erase_x.append(i)

    for i in range(len(xx) - 1, -1, -1):
        erase = False
        if i in to_erase_x or i in to_erase_y:
            erase = True
        if erase:
            del xx[i]
            del yy[i]

    points = []
    for i in range(len(xx)):
        p = Point(xx[i], yy[i])
        if not is_in_list(list_of_points=list_already_visited, point_to_check=p):
            points.append(p)
    return points


def is_in_list(list_of_points, point_to_check):
    """
    Check if a point is already in the list.
    If the list of points is empty or not used, the functions returns False

    :param list_of_points: list of points where to check
    :param point_to_check: point to check
    :return: Boolean Value: True if it is in the list, False otherwise
    """
    if list_of_points is None:
        return False
    for el in list_of_points:
        if el.equals(p1=point_to_check):
            return True
    return False


def _get_direction(current_point, next_point):
    """
    From the current point return the vector defining the decision the model follow to move to the next point
    :param current_point: current location
    :param next_point: point just moved on
    :return: vector specifying the decision the model made to go to the next point. Vector 8 positions one-hot encoded
    """
    x_value = int(current_point.x)
    y_value = int(current_point.y)
    x_value_next = int(next_point.x)
    y_value_next = int(next_point.y)

    diff_x = x_value_next - x_value
    diff_y = y_value_next - y_value
    if diff_x == 1 and diff_y == -1:
        return [1, 0, 0, 0, 0, 0, 0, 0]
    elif diff_x == 0 and diff_y == -1:
        return [0, 1, 0, 0, 0, 0, 0, 0]
    elif diff_x == -1 and diff_y == -1:
        return [0, 0, 1, 0, 0, 0, 0, 0]
    elif diff_x == -1 and diff_y == 0:
        return [0, 0, 0, 1, 0, 0, 0, 0]
    elif diff_x == -1 and diff_y == 1:
        return [0, 0, 0, 0, 1, 0, 0, 0]
    elif diff_x == 0 and diff_y == 1:
        return [0, 0, 0, 0, 0, 1, 0, 0]
    elif diff_x == 1 and diff_y == 1:
        return [0, 0, 0, 0, 0, 0, 1, 0]
    elif diff_x == 1 and diff_y == 0:
        return [0, 0, 0, 0, 0, 0, 0, 1]
    else:
        raise Exception("{} and {} not present".format(diff_x, diff_y))
