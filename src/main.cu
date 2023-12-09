#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include "point.cu"
#include "sort_array.cu"

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

#define N 1024 // TODO WHICH VALUE?

__global__ void test(point2D *pts, unsigned int refPointIndex) {
    point2D* localPoint = &pts[(blockIdx.x*N + threadIdx.x)];

    float old_x = localPoint->x;
    float old_y = localPoint->y;
    
    float ref_x = pts[refPointIndex].x;
    float ref_y = pts[refPointIndex].y;

    localPoint->x = ref_y - old_y;
    localPoint->y = (ref_y - old_y) * (ref_y - old_y) + (ref_x - old_x) * (ref_x - old_x);
}

void projection(std::vector<point2D> pointsVector) {
    point2D *res;

    long unsigned int mem = sizeof(point2D) * pointsVector.size();
    cudaMalloc((void**)&res, mem);
    cudaMemcpy(res, &pointsVector[0], mem, cudaMemcpyHostToDevice);

    dim3 dimGrid((pointsVector.size()+N-1)/N, 1);   // Nb of blocks
    dim3 dimBlock(N, 1);
    test<<<dimGrid, dimBlock>>>(res, pointsVector.size()/2);

    cudaDeviceSynchronize();

    point2D projection[pointsVector.size()]; // projection results
    cudaMemcpy(projection, res, mem, cudaMemcpyDeviceToHost);

    for (int i=0; i<pointsVector.size(); i++) {
        std::cout << "Original: " << pointsVector[i].x << " " << pointsVector[i].y << std::endl;
        std::cout << "Projected: " << projection[i].x << " " << projection[i].y << std::endl;
    }

    cudaFree(res);
}

int main(int argc, char *argv[]) {

    // TODO arguments reading and errors (filename, splitting method, ...)
    if (argc < 2) {
        std::cout << "No input file provided" <<std::endl;
        return 1;
    }
    std::vector<point2D> pointsVector = readFile(argv[1]);

    auto start = high_resolution_clock::now();
    std::sort(pointsVector.begin(), pointsVector.end(), xCompare);
	auto elapse = std::chrono::system_clock::now() - start;
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapse);

    projection(pointsVector);

    // point2D* res = sortInputIntoGPU(pointsVector);

    // for (int i = 0; i < pointsVector.size(); i++){
    //     std::cout << "Index :" << pointsVector[i].index << " X : " << pointsVector[i].x << " Y :" << pointsVector[i].y << std::endl;
    // }

    // cudaFree(res);

    return 0;
}