#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include "mesh.cu"
#include "sort_array.cu"
#include "projection.cu"

using namespace std::chrono;

//CPU Compare function
bool xCompare (vertex a, vertex b){return a.x < b.x;}
bool yCompare (vertex a, vertex b){return a.y < b.y;}

// Returns unique 64-bits int with 2 32-bits float
long unsigned int hash(float x, float y) {
    // Get binary representation of both numbers
    long unsigned int xInt = * (unsigned int *) &x;
    long unsigned int yInt = * (unsigned int *) &y;
    return (xInt << 32) + yInt;
}

std::vector<vertex> readFile(std::string nameFile){
    std::vector<vertex> pointsVector;

    std::ifstream inputFile;
    inputFile.open(nameFile);
    
    // Used to check if two points are identical
    std::unordered_set<long unsigned int> pointsSet;

    unsigned int i=0;
    float x,y;

    while(inputFile >> x >> y) {
        // Only push if the point is not over another
        long unsigned int hashValue = hash(x, y);
        if (!pointsSet.count(hashValue)) {
            pointsSet.insert(hashValue);
            pointsVector.push_back({i++, x, y});
        }
    }

    std::cout << "Loaded file with " << pointsVector.size() << " distinct points\n" << std::endl;
    inputFile.close();

    return pointsVector;
    
}

int main(int argc, char *argv[]) {

    // TODO arguments reading and errors (filename, splitting method?, ...)
    if (argc < 2) {
        std::cout << "No input file provided" <<std::endl;
        return 1;
    }

    // Read original values
    std::vector<vertex> pointsVector = readFile(argv[1]);

    // Sorting values according to an axis (TODO GPU SORT)
    auto start = high_resolution_clock::now();
    std::sort(pointsVector.begin(), pointsVector.end(), xCompare);
	auto elapse = std::chrono::system_clock::now() - start;
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapse);

    vertex *pointsOnGPU;
    long unsigned int mem = sizeof(vertex) * pointsVector.size();
    cudaMalloc((void**)&pointsOnGPU, mem);
    cudaMemcpy(pointsOnGPU, &pointsVector[0], mem, cudaMemcpyHostToDevice);

    // Get GPU array of projected points
    vertex* proj = projection(pointsOnGPU, pointsVector.size());
    cudaFree(proj);

    // vertex* res = sortInputIntoGPU(pointsVector);

    // for (int i = 0; i < pointsVector.size(); i++){
    //     std::cout << "Index :" << pointsVector[i].index << " X : " << pointsVector[i].x << " Y :" << pointsVector[i].y << std::endl;
    // }

    // cudaFree(res);

    return 0;
}