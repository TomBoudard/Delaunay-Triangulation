#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include "point.cu"
#include "sort_array.cu"
#include "projection.cu"

using namespace std::chrono;

//CPU Compare function
bool xCompare (point2D a, point2D b){return a.x < b.x;}
bool yCompare (point2D a, point2D b){return a.y < b.y;}

std::vector<point2D> readFile(std::string nameFile){
    std::vector<point2D> pointsVector;

    std::ifstream inputFile;
    inputFile.open(nameFile);
    
    unsigned int i=0;
    float x,y;

    while(inputFile >> x >> y) {
        pointsVector.push_back({i++, x, y});
    }

    return pointsVector;
    
}

int main(int argc, char *argv[]) {

    // TODO arguments reading and errors (filename, splitting method, ...)
    if (argc < 2) {
        std::cout << "No input file provided" <<std::endl;
        return 1;
    }

    // Read original values
    std::vector<point2D> pointsVector = readFile(argv[1]);

    // Sorting values according to an axis (TODO GPU SORT)
    auto start = high_resolution_clock::now();
    std::sort(pointsVector.begin(), pointsVector.end(), xCompare);
	auto elapse = std::chrono::system_clock::now() - start;
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapse);

    // Get GPU array of projected points
    point2D* proj = projection(pointsVector);

    // point2D* res = sortInputIntoGPU(pointsVector);

    // for (int i = 0; i < pointsVector.size(); i++){
    //     std::cout << "Index :" << pointsVector[i].index << " X : " << pointsVector[i].x << " Y :" << pointsVector[i].y << std::endl;
    // }

    // cudaFree(res);

    return 0;
}