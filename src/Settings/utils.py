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
                                 "normal", "same", "variance", "mootwo"],
                        help="Select what fitness to use. Choices are: PB polygon borders")
    parser.add_argument("--point_distance", type=eval, help="Select witch term to use for fitness only distance to "
                                                            "central point")
    parser.add_argument("--penalty_behaviours", action='store_true', help="Add penalty for not having different"
                                                                          " trajecotires with different behaviours")
    parser.add_argument("--population_size", "-p", type=int, default=50, help="Set population size")
    parser.add_argument("--generations", "-g", type=int, default=250, help="Set number max of generations")
    parser.add_argument("--numb_of_tra", "-n", type=int, default=30, help="Number of trajectories to generate"
                                                                         " with the same model")
    parser.add_argument("--number_of_offspring", "-no", type=int, default=50, help="Number of offsprings")
    parser.add_argument("--neat", "-nt", action='store_true', help="Use NEAT as algorithm")

    # program settigs
    parser.add_argument("--freq_checkpoints", "-fr", type=int, default=10, help="Every how many generations to "
                                                                                "save a checkpoint")
    parser.add_argument("--output_directory", "-o", type=str,
                        # default="/Users/alessandrozonta/PycharmProjects/deapGeneration/Output/",
                        default="./Output/",
                        help="Directory where to save"
                             " checkpoint evolution")
    parser.add_argument("--name_experiment", "-ne", type=str, default="test", help="Name Experiment")
    parser.add_argument("--data_directory", type=str, default="./Data/", help="Directory with the data")
    parser.add_argument("--run_name", "-rn", type=str, default="00000", help="Name Experiment")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Seed value")
    parser.add_argument("--model_used", "-m", default="LSTM", choices=["LSTM", "GRU"],
                        help="Select what model to use")
    parser.add_argument("--checkpoint", type=str, default="/Users/alessandrozonta/Desktop/super_long_all_the_way_neat__point_distance_0_1_direction_normal/neat-checkpoint-94",
                        help="path checkpoint file where to start evolution from")
    parser.add_argument("--random_point_start",  action='store_true', help="Forse program to have random different "
                                                                           "initial point per individual")
    parser.add_argument("--two_stages", action="store_true", help="after reached best fitness for at least 20 "
                                                                  "generation, activate random_point_start")
    parser.add_argument("--number_of_test_trajectories", type=int, default=100, help="how many trajectories to "
                                                                                     "generate at the end to test "
                                                                                     "the winning genome")
    # network settings
    parser.add_argument("--hidden_layers", "-hl", type=int, default=2, help="Number of hidden layers in the network")
    parser.add_argument("--hidden_neurones", "-hn", type=int, default=10, help="Number of neurones per hidden "
                                                                               "layer in the network")
    parser.add_argument("--input_neurones", "-in", type=int, default=10, help="Number of neurones in the input layer")
    parser.add_argument("--activation_function_middle", "-afm", type=str, default="tanh",
                        choices=["tanh", "relu", "sigmoid"], help="Fitness function to use in hidden layers")
    parser.add_argument("--activation_function_end", "-afe", type=str, default="sigmoid",
                        choices=["softmax", "sigmoid"], help="Fitness function to use in output")

    # novelty search settings
    parser.add_argument("--k", type=int, default=15, help="k value for k-neighbours computation")
    parser.add_argument("--prob_add", type=float, default=0.1, help="Probability to add individual to archive")

    return parser

parser = prepare_parser()
args = parser.parse_args()