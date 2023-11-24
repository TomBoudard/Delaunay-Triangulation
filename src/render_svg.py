#!/usr/bin/env python3

"""Convertit une liste de triangles en une image svg"""

import sys
import os

# TODO modif
largeur = 5
hauteur = 5

def render_svg(coords, trig):

    coords_list = []

    with open(coords, 'r') as c:
        for line in c:
            # Append each point to the list
            coords_list.append(list(map(float, line.split())))

    with open("output.svg", 'w') as o:
        # Beginning of image
        o.write(f"<svg xmlns='http://www.w3.org/2000/svg' version='1.1' width='{largeur}"
           f"' height='{hauteur}'>\n")

        # Write every triangle
        with open(trig, 'r') as t:
            
            for line in t:
                line = list(map(int, line.split())) # line contains the triplet
                o.write(add_trig(coords_list, line))
        
        # End of image
        o.write(f"</svg>\n")

def add_trig(coords_list, line):
    
    for i in range(3):
        line[i] = coords_list[line[i]]

    return f'<polygon points="{line[0][0]},{line[0][1]} {line[1][0]},{line[1][1]} {line[2][0]},{line[2][1]}" style="fill:white;stroke:black;stroke-width:0.1" />\n'

if __name__=="__main__":
    if len(sys.argv) < 3 or sys.argv[1] in ["-h","--help"]:
        print("Usage :")
        print("./render_svg.py coords_list trig_list")
        exit(1)

    coords_file, trig_file = sys.argv[1], sys.argv[2]

    if not os.path.isfile(coords_file):
        print("Error: coords file not found")
        exit(2)

    if not os.path.isfile(trig_file):
        print("Error: trig file not found")
        exit(3)

    render_svg(coords_file, trig_file)