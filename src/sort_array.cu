#define NB_CORES 2
// Number of elements we should sort on one core (threshold)
#define NB_ELEM_SORT_CORE 5

#ifndef SORT_ARRAY
#define SORT_ARRAY

#include <stdio.h>
#include <cmath>
#include <vector>
#include <iostream>
#include "mesh.cu"

// Sorts a part of array
__global__ void sortGPU() {

}

// Sorts requested array stored on GPU. Updates input pointer
// Input:
//  - Pointer of the original array and length
// Output:
//  - Pointer of the sorted array
void sortArray(vertex** input, unsigned int len) {

    // We do a merge sort so we will alternate between two buffers
    // The first one is the provided array, the other is allocated here
    vertex* newBuffer;
    long unsigned int mem = sizeof(vertex) * len;
    cudaMalloc((void**)&newBuffer, mem);

    vertex** buffers[2];
    buffers[0] = input;
    buffers[1] = &newBuffer;

    unsigned int currentInputBuffer = 0;

    // Input points the final buffer and the other one is free'd
    *input = *buffers[1-currentInputBuffer];
    cudaFree(*buffers[currentInputBuffer]);

    return;
}


// __global__ void simple(vertex *r) {
//     r[threadIdx.x].x = threadIdx.x;
// }

// // Sorts the vector into the GPU memory
// // Returns pointer to sorted array in GPU
// vertex* sortInputIntoGPU(std::vector<vertex> pointsVector) {
//     vertex *res;
//     int mem = sizeof(vertex) * pointsVector.size();
//     cudaMalloc((void**)&res, mem);

//     cudaMemcpy(res, &pointsVector, 10, cudaMemcpyHostToDevice);

//     dim3 dimGrid(mem, 1);
//     dim3 dimBlock(sizeof(vertex), 1);
//     simple<<<dimGrid, dimBlock>>>(res);

//     cudaMemcpy(&pointsVector, res, 10, cudaMemcpyDeviceToHost);

//     for (int i=0; i<10; i++) std::cout<< res[i].x <<" ";
//     std::cout<<std::endl;

//     return (vertex*) res;
// }

#endif