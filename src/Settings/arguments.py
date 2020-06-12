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
import argparse


def prepare_parser():
    parser = argparse.ArgumentParser(description="Generation Evolution")

    # EA settings
    parser.add_argument("--fitness_definition", "-f", default="normal",
                        choices=["mix_multiplication", "mix_weighted",
                                 "mix_inverse", "novelty", "moo",
                                 "normal", "same", "normal_direction", "normal_direction_five"],
                        help="Select what fitness to use. Choices are: PB polygon borders")
    parser.add_argument("--point_distance", type=eval, help="Select witch term to use for fitness only distance to "
                                                            "central point")
    parser.add_argument("--penalty_behaviours", action='store_true', help="Add penalty for not having different"
                                                                          " trajecotires with different behaviours")
    parser.add_argument("--generations", "-g", type=int, default=10, help="Set number max of generations")
    parser.add_argument("--numb_of_tra", "-n", type=int, default=30, help="Number of trajectories to generate"
                                                                         " with the same model")

    # program settigs
    parser.add_argument("--freq_checkpoints", "-fr", type=int, default=10, help="Every how many generations to "
                                                                                "save a checkpoint")
    parser.add_argument("--output_directory", "-o", type=str,
                        # default="/Users/alessandrozonta/PycharmProjects/deapGeneration/Output/",
                        default="/Users/alessandrozonta/PycharmProjects/NEAT/Output/",
                        help="Directory where to save"
                             " checkpoint evolution")
    parser.add_argument("--name_experiment", "-ne", type=str, default="test", help="Name Experiment")
    parser.add_argument("--data_directory", type=str, default="/Users/alessandrozonta/PycharmProjects/deapGeneration/Data/", help="Directory with the data")
    parser.add_argument("--config_path", type=str, default="/Users/alessandrozonta/PycharmProjects/NEAT/src/Settings/config-neat", help="Directory with the settings "
                                                                                          "for NEAT")
    parser.add_argument("--run_name", "-rn", type=str, default="00000", help="Name Experiment")


    parser.add_argument("--seed", "-s", type=int, default=42, help="Seed value")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="path checkpoint file where to start evolution from")
    parser.add_argument("--random_point_start",  action='store_true', help="Force program to have random different "
                                                                           "initial point per individual")

    parser.add_argument("--number_of_test_trajectories", type=int, default=100, help="how many trajectories to "
                                                                                     "generate at the end to test "
                                                                                     "the winning genome")

    return parser

parser = prepare_parser()
args = parser.parse_args()