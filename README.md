# Delaunay Triangulation

In this project, we implemented a parallel algorithm on GPU for Delaunay Triangulations based on [this paper](https://membres-ljk.imag.fr/Christophe.Picard/teaching/gp-gpu/References/lee-1997-IEEE.pdf).

The algorithm is done with cuda. It takes an input file containing a list of 2D points and returns a list of triangles (refering to them as their indexes). The files specification can be found in the next section. The folder also contains Python scripts to generate datasets and svg visualizations of the results.

## Files specifications

### Input (List of points)

A text file containing lines of `x y` coordinates (floats).

Example:
```
1.05 4.3
5.0 2.4
3.2 1.3
1.4 2.8
```

### Output (List of triangles)

A text file containing triplets of indexes for each triangle.

Example:
```
0 1 3
2 1 3
```

## Data generation

`./generate_points.py` can be used to generate `input.txt` input file for our algorithm.

Usage can be found with the help argument (`-h` or `--help`).

## Visualization

`./render_svg.py` can be used to generate a `.svg` output for a given dataset and triangles.

Usage : `./render_svg.py coords_list trig_list`

## Triangulation

Create a build folder and execute `cmake ..` to generate files used for the algorithm. Then `make` and run `./main`.