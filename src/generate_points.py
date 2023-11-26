#!/usr/bin/env python3

"""Creates a .txt file containing 2D points"""

import sys
import random
from enum import Enum

class randomType(Enum):
    UNIFORM = 1
    LINE = 2
    GAUSSIAN = 3


def create_points(nb_pts, min_val, max_val, randomTypeGeneration):
    with open("input.txt", 'w') as f:
        for i in range(nb_pts):
            x, y = generate_pt(min_val, max_val, randomTypeGeneration)
            f.write(f"{x} {y}\n")


def generate_pt(min_val, max_val, randomTypeGeneration):
    if randomTypeGeneration == 1:
        return random.uniform(min_val, max_val), random.uniform(min_val, max_val)
    elif randomTypeGeneration == 2:
        mean_val = (min_val+max_val)/2
        return random.uniform(mean_val*4/5, mean_val*6/5), random.uniform(min_val, max_val)
    elif randomTypeGeneration == 3:
        rangeValues = max_val - min_val
        return random.gauss(0, 1)*rangeValues, random.gauss(0, 1)*rangeValues

if __name__=="__main__":
    if len(sys.argv) != 5 or sys.argv[1] in ["-h","--help"]:
        print("Usage :")
        print("./generate_points.py nb_pts min_val max_val random_type_generation")
        exit(1)

    nb_pts, min_val, max_val, randomTypeGeneration = list(map(int,sys.argv[1:]))

    create_points(nb_pts, min_val, max_val, randomTypeGeneration)
