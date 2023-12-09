#!/usr/bin/env python3

"""Renders triangles in a svg file"""

import sys
import os
from math import sqrt

def get_coords_list(coords_file):
    """Reads file containing every points and store them in an array"""
    """Also returns min and max values (not separating axis)"""

    coords_list = []
    min_val, max_val = None, None

    with open(coords_file, 'r') as c:
        for line in c:
            pt = list(map(float, line.split()))
            coords_list.append(pt)

            if min_val:
                min_val = min(min_val, pt[0], pt[1])
                max_val = max(max_val, pt[0], pt[1])
            else:
                min_val = min(pt[0], pt[1])
                max_val = max(pt[0], pt[1])
    
    return coords_list, min_val, max_val

def render_svg(coords, trig):
    """Main function creating a svg file containing every triangle"""

    coords_list, min_val, max_val = get_coords_list(coords)
    width = max_val - min_val

    with open("output.svg", 'w') as o:
        # Beginning of image
        o.write(f"<svg xmlns='http://www.w3.org/2000/svg' version='1.1' width='{width}"
           f"' height='{width}'>\n")

        trig_list = [] # Triangles list

        # Get every triangle
        with open(trig, 'r') as t:
            for line in t:
                line = list(map(int, line.split())) # line contains the triplet of indexes

                # Offset coordinated so the minimum is (0, 0) for svg rendering
                for i in range(3):
                    line[i] = [coords_list[line[i]][j] - min_val for j in range(2)]

                trig_list.append(line)

        for trig in trig_list:
            o.write(add_trig(trig, width/200)) # Adding triangle
        
        for trig in trig_list:
            o.write(add_circle(trig, width/200)) # Adding circle
        
        # End of image
        o.write(f"</svg>\n")

def add_trig(trig, w):
    """Returns svg string for a given triangle"""

    return f'<polygon points="{trig[0][0]},{trig[0][1]} {trig[1][0]},{trig[1][1]} {trig[2][0]},{trig[2][1]}" style="fill:white;stroke:black;stroke-width:{w}" />\n'

def add_circle(trig, w):
    """Returns svg string for the circumscribed circle of a given triangle"""

    vec_p_AB = (trig[1][1] - trig[0][1], trig[0][0] - trig[1][0]) # Perpendicular to AB
    vec_p_AC = (trig[2][1] - trig[0][1], trig[0][0] - trig[2][0]) # Perpendicular to AC

    # If vectors are identical or opposite, no triangle
    det = vec_p_AB[1] * vec_p_AC[0] - vec_p_AB[0] * vec_p_AC[1]
    if det == 0:
        return ""

    mid_AB = ((trig[1][0] + trig[0][0])/2, (trig[1][1] + trig[0][1])/2) 
    mid_AC = ((trig[2][0] + trig[0][0])/2, (trig[2][1] + trig[0][1])/2)

    u = (vec_p_AB[0] * (mid_AC[1] - mid_AB[1]) - vec_p_AB[1] * (mid_AC[0] - mid_AB[0])) / det

    center = (mid_AC[0] + u * vec_p_AC[0], mid_AC[1] + u * vec_p_AC[1])

    r = sqrt((center[0] - trig[0][0])**2 + (center[1] - trig[0][1])**2)

    return f'<circle cx="{center[0]}" cy="{center[1]}" r="{r}" style="fill:none;stroke:black;stroke-width:{w}" />\n'


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