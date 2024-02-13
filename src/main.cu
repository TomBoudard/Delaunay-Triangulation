#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include "tools.cu"
#include "sort_array.cu"
#include "parDeTri.cu"
#include "findPaths.cu"

using namespace std::chrono;

// CPU Compare function
bool xCompare (float3 a, float3 b){return a.x < b.x;}
bool yCompare (float3 a, float3 b){return a.y < b.y;}

// Returns unique 64-bits int with 2 32-bits float
long unsigned int hash(float x, float y) {
    // Get binary representation of both numbers
    long unsigned int xInt = * (unsigned int *) &x;
    long unsigned int yInt = * (unsigned int *) &y;
    return (xInt << 32) + yInt;
}

std::vector<float3> readFile(std::string nameFile){
    std::vector<float3> pointsVector;

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
            pointsVector.push_back({x, y, * (float *) &i});
            i++;
        }
    }

    std::cout << "Loaded file with " << pointsVector.size() << " distinct points\n" << std::endl;
    inputFile.close();

    return pointsVector;
    
}

int main(int argc, char *argv[]) {

    // arguments reading and errors (filename)
    if (argc < 2) {
        std::cout << "No input file provided" <<std::endl;
        return 1;
    }

    // -- Read original values
    std::vector<float3> pointsVector = readFile(argv[1]);

    // -- Sort

    // CPU Sorting values according to an axis
    std::sort(pointsVector.begin(), pointsVector.end(), xCompare);

    float3 *pointsOnGPU;
    long unsigned int mem = sizeof(float3) * pointsVector.size();
    cudaMalloc((void**)&pointsOnGPU, mem);
    cudaMemcpy(pointsOnGPU, &pointsVector[0], mem, cudaMemcpyHostToDevice);
    // sortArray(&pointsOnGPU, pointsVector.size());

    // // DEBUG (copy back & print)
    // cudaMemcpy(&pointsVector[0], pointsOnGPU, mem, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < pointsVector.size(); i++){
    //     std::cout << "Index :" << * (int *) &(pointsVector[i].z) << " X :" << pointsVector[i].x << " Y :" << pointsVector[i].y << std::endl;
    // } 

    struct edge* paths = createPaths(pointsOnGPU, pointsVector.size());
    cudaFree(paths);

    // for (int i = 0; i < pointsVector.size(); i++){
    //     std::cout << "Index :" << pointsVector[i].index << " X : " << pointsVector[i].x << " Y :" << pointsVector[i].y << std::endl;
    // }

    // cudaFree(res);

    return 0;
}

// CPU time
// auto start = high_resolution_clock::now();
// auto elapse = std::chrono::system_clock::now() - start;
// auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapse);