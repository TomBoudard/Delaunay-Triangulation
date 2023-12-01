#include <stdio.h>
#include "point.cu"

__global__ void simple(point2D *r) {
    r[threadIdx.x].x = threadIdx.x;
}

// Sorts the vector into the GPU memory
// Returns pointer to sorted array in GPU
point2D* sortInputIntoGPU(std::vector<point2D> pointsVector) {
    point2D *res;
    int mem = sizeof(point2D) * pointsVector.size();
    cudaMalloc((void**)&res, mem);

    cudaMemcpy(res, &pointsVector, 10, cudaMemcpyHostToDevice);

    dim3 dimGrid(mem, 1);
    dim3 dimBlock(sizeof(point2D), 1);
    simple<<<dimGrid, dimBlock>>>(res);

    cudaMemcpy(&pointsVector, res, 10, cudaMemcpyDeviceToHost);

    for (int i=0; i<10; i++) std::cout<< res[i].x <<" ";
    std::cout<<std::endl;

    return (point2D*) res;
}

