#ifndef SORT_ARRAY
#define SORT_ARRAY

#include <stdio.h>
#include <vector>
#include <iostream>
#include "mesh.cu"


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