#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "point.cu"
#include "sort_array.cu"

//CPU Compare functions
bool xCompare (point2D a, point2D b){return a.x < b.x;}
bool yCompare (point2D a, point2D b){return a.y < b.y;}

std::vector<point2D> readFile(std::string nameFile){
    std::vector<point2D> pointsVector;

    std::ifstream inputFile;
    inputFile.open(nameFile);
    
    int i=0;
    float x,y;

    while(inputFile >> x >> y) {
        pointsVector.push_back({i++, x, y});
    }

    return pointsVector;
    
}

point2D* sortInputIntoGPU(std::vector<point2D> pointsVector){
    point2D* res;
    int mem = sizeof(point2D)*pointsVector.size();
    cudaMalloc((void**)&res, mem);
    cudaMemcpy(res, &pointsVector, mem, cudaMemcpyHostToDevice);
}

int main(int argc, char *argv[]) {

    // TODO arguments reading and errors (filename, splitting method, ...)

    std::vector<point2D> pointsVector = readFile(argv[1]);

    std::sort(pointsVector.begin(), pointsVector.end(), yCompare);

    for (int i = 0; i < pointsVector.size(); i++){
        std::cout << "Point number : " << pointsVector[i].index << " X : " << pointsVector[i].x << " Y :" << pointsVector[i].y << std::endl;
    }

    sortInputIntoGPU(pointsVector);

    return 0;
}