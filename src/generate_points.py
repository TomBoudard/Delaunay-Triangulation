#!/usr/bin/env python3

"""Creates a .txt file containing 2D points"""

import sys
import random

def create_points(nb_pts, min_val, max_val):
    with open("input.txt", 'w') as f:
        for i in range(nb_pts):
            x, y = generate_pt(min_val, max_val)
            f.write(f"{x} {y}\n")

def generate_pt(min_val, max_val):
    return random.uniform(min_val, max_val), random.uniform(min_val, max_val)

if __name__=="__main__":
    if len(sys.argv) != 4 or sys.argv[1] in ["-h","--help"]:
        print("Usage :")
        print("./generate_points.py nb_pts min_val max_val")
        exit(1)

    nb_pts, min_val, max_val = list(map(int,sys.argv[1:]))

    create_points(nb_pts, min_val, max_val)
