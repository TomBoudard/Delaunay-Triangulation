#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include "tools.cu"
#include "triangulation.cu"
#include "findPaths.cu"

using namespace std::chrono;

#define THRESHOLD 25 // TODO WHICH VALUE?

// For a subproblem of n points, we put a limit of 10*n triangles generated.
// This is manually fixed because otherwise it would not have any limit and we have limited memory.
#define NB_MAX_TRIANGLES_PER_PT 10

//CPU Compare function
bool xCompare (float3 a, float3 b){return a.x < b.x;}

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
            pointsVector.push_back(make_float3(x, y, * (float *) &i));
            i++;
        }
    }

    std::cout << "Loaded file with " << pointsVector.size() << " distinct points\n" << std::endl;
    inputFile.close();

    return pointsVector;
    
}

void writeFile(int3* triangleList, int lineSize, int nbLines){
    std::ofstream indexFile("indexOutput.txt");

    for (int i = 0; i < nbLines; i++){
        for(int j = 0; j < lineSize; j++){
            if (triangleList[i*lineSize+j].x < 0)
                break;
            indexFile <<  triangleList[i*lineSize+j].x;
            indexFile <<  " ";
            indexFile <<  triangleList[i*lineSize+j].y;
            indexFile <<  " ";
            indexFile <<  triangleList[i*lineSize+j].z;
            indexFile << std::endl;
        }
    }
    indexFile.close();
}

int main(int argc, char *argv[]) {

    // arguments reading and errors (filename)
    if (argc < 2) {
        std::cout << "No input file provided" <<std::endl;
        return 1;
    }

    // -- Read original values
    std::vector<float3> pointsVector = readFile(argv[1]);
    int nbPoints = pointsVector.size();

    if (nbPoints < 1) {
        std::cout << "Input file is empty or does not exist." << std::endl;
        return 1;
    }

    // CPU Sorting values according to an axis
    std::sort(pointsVector.begin(), pointsVector.end(), xCompare);

    auto start = high_resolution_clock::now();

    // Copy points on GPU
    float3 *pointsOnGPU;
    long unsigned int mem = sizeof(float3) * nbPoints;
    cudaMalloc((void**)&pointsOnGPU, mem);
    cudaMemcpy(pointsOnGPU, &pointsVector[0], mem, cudaMemcpyHostToDevice);

    // Find the number of subproblems according to the threshold of the
    // maximum number of points per subproblems. This number is always a power of 2

    // The way we have done our algorithm requires at least one split
    // To split the problem, the threshold is lowered if it is necesary
    int thresholdUsed = min(nbPoints-1, THRESHOLD);

    int nbSubproblems = 1, log2nbSubproblems = 0;
    while ((nbSubproblems * thresholdUsed) < nbPoints) {
        log2nbSubproblems++;
        nbSubproblems <<= 1;
    }

    struct edge* edgePathsList = createPaths(pointsOnGPU, nbPoints, nbSubproblems, log2nbSubproblems);

    int nbMaxTrianglesUsed = thresholdUsed * NB_MAX_TRIANGLES_PER_PT;

    int3* triangleList;
    int3* initTriangleList = new int3[nbSubproblems*nbMaxTrianglesUsed];

    struct edge* globalEdgeList;
    int boundMaxEdgePerSubset = (int)(2*nbPoints/nbSubproblems - 2)*3*3;
    edge* initGlobalEdgeList = new edge[boundMaxEdgePerSubset*nbSubproblems];

    for (int i = 0; i < boundMaxEdgePerSubset*nbSubproblems; i++){
        initGlobalEdgeList[i].usage = UNUSED;
    }

    cudaMalloc((void**)&triangleList, sizeof(int3)*nbSubproblems*nbMaxTrianglesUsed); // FIXME Stored contiguously
    cudaMemcpy(triangleList, initTriangleList, sizeof(int3)*nbSubproblems*nbMaxTrianglesUsed, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&globalEdgeList, sizeof(edge)*boundMaxEdgePerSubset*nbSubproblems);
    cudaMemcpy(globalEdgeList, initGlobalEdgeList, sizeof(edge)*boundMaxEdgePerSubset*nbSubproblems, cudaMemcpyHostToDevice);

    std::cout << "Nb of subproblems: " << nbSubproblems << std::endl;

    parDeTri<<<nbSubproblems, 1>>>(pointsOnGPU, edgePathsList, globalEdgeList, triangleList, nbPoints, nbSubproblems, nbMaxTrianglesUsed);
    cudaDeviceSynchronize();

    cudaMemcpy(initTriangleList, triangleList, sizeof(int3)*nbSubproblems*nbMaxTrianglesUsed, cudaMemcpyDeviceToHost);

    auto elapse = std::chrono::system_clock::now() - start;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapse);
    std::cout << "Total duration : " << duration.count() << std::endl;

    writeFile(initTriangleList, nbMaxTrianglesUsed, nbSubproblems);

    cudaFree(pointsOnGPU);
    cudaFree(edgePathsList);
    cudaFree(triangleList);
    cudaFree(globalEdgeList);
    delete[] initTriangleList;
    delete[] initGlobalEdgeList;

    return 0;
}

// CPU time
// auto start = high_resolution_clock::now();
// auto elapse = std::chrono::system_clock::now() - start;
// auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapse);
// std::cout << duration.count() << std::endl;

// GPU time
// cudaEvent_t myEvent, laterEvent;
// cudaEventCreate(&myEvent);
// cudaEventRecord(myEvent, 0);
// cudaEventSynchronize(myEvent);
// int dimGrid = (nbPts+N-1)/N;   // Nb of blocks
// int dimBlock = N;
// projectPoints<<<dimGrid, dimBlock>>>(pointsOnGPU, pointsProjected, nbPts, true);
// cudaDeviceSynchronize();

// cudaEventCreate(&laterEvent);
// cudaEventRecord(laterEvent, 0);
// cudaEventSynchronize(laterEvent);