#!/usr/bin/env python3

"""Creates a .txt file containing 2D points"""

import sys
import random

UNIFORM, LINE, GAUSSIAN = 0, 1, 2

def create_points(nb_pts, min_val, max_val, randomTypeGeneration):
    with open("input.txt", 'w') as f:
        for i in range(nb_pts):
            x, y = generate_pt(min_val, max_val, randomTypeGeneration)
            f.write(f"{x} {y}\n")

def generate_pt(min_val, max_val, randomTypeGeneration):
    if randomTypeGeneration == UNIFORM:
        return random.uniform(min_val, max_val), random.uniform(min_val, max_val)

    if randomTypeGeneration == LINE:
        mean_val = (min_val + max_val)/2
        return random.uniform(mean_val*4/5, mean_val*6/5), random.uniform(min_val, max_val)

    if randomTypeGeneration == GAUSSIAN:
        mean_val = (min_val + max_val)/2
        rangeValues = max_val - min_val
        return random.gauss(mean_val, rangeValues/3), random.gauss(mean_val, rangeValues/3)

if __name__=="__main__":
    if len(sys.argv) != 5 or sys.argv[1] in ["-h","--help"]:
        print("Usage :")
        print("./generate_points.py number_points min_val max_val random_type_generation")
        print("random_type_generation :")
        print("\t0 = Uniform")
        print("\t1 = Line")
        print("\t2 = Gaussian")
        exit(1)

    nb_pts, min_val, max_val, randomTypeGeneration = list(map(int,sys.argv[1:]))

    if not 0 <= randomTypeGeneration <= 2:
        print("Wrong type generation")
        exit(2)

    create_points(nb_pts, min_val, max_val, randomTypeGeneration)
