#!/usr/bin/env python3

"""Renders triangles in a svg file"""

# NOTE : a better visualization with circumscribed circles could be done

import sys
import os

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

        # Write every triangle
        with open(trig, 'r') as t:
            for line in t:
                line = list(map(int, line.split())) # line contains the triplet of indexes
                o.write(add_trig(coords_list, line, min_val, width/200))
        
        # End of image
        o.write(f"</svg>\n")

def add_trig(coords_list, line, offset, w):
    """Returns svg string for a given triangle"""

    # Get coordinates and offset them in the square so it's visible on svg
    for i in range(3):
        line[i] = [coords_list[line[i]][j] - offset for j in range(2)]

    return f'<polygon points="{line[0][0]},{line[0][1]} {line[1][0]},{line[1][1]} {line[2][0]},{line[2][1]}" style="fill:white;stroke:black;stroke-width:{w}" />\n'

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