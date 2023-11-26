#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "point.cu"
#include "sort_array.cu"

//CPU Compare function
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

int main(int argc, char *argv[]) {

    // TODO arguments reading and errors (filename, splitting method, ...)

    std::vector<point2D> pointsVector = readFile(argv[1]);

    std::sort(pointsVector.begin(), pointsVector.end(), xCompare);

    for (int i = 0; i < pointsVector.size(); i++){
        std::cout << "X : " << pointsVector[i].x << " Y :" << pointsVector[i].y << std::endl;
    }

    return 0;
}