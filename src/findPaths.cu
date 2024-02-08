#define THRESHOLD 4 // TODO WHICH VALUE?
#define NB_MAX_THREADS 4 // SHOULD BE POWER OF 2

__global__ void projectSlice(float3 *points, float3 *projected, int3 *paths, int nbPoints, int roundId) {

    // ID used for the reference point around which other points are projected for this block
    int refPoint = (2 * blockIdx.x + 1) * nbPoints / (2 << roundId);
    
    // IDs of the beginning and end of the slice manipulated by the block
    int sliceBlockBeg = (2 * blockIdx.x) * nbPoints / (2 << roundId);
    int sliceBlockEnd = (2 * (blockIdx.x + 1)) * nbPoints / (2 << roundId);

    // IDs of the beginning and end of the slice manipulated by the thread
    float ratio = (float) threadIdx.x / blockDim.x;
    float ratio2 = (float) (threadIdx.x+1) / blockDim.x;
    int sliceThreadBeg = (int) ((float) sliceBlockBeg * (1-ratio) + (float) sliceBlockEnd * ratio);
    int sliceThreadEnd = (int) ((float) sliceBlockBeg * (1-ratio2) + (float) sliceBlockEnd * ratio2);

    // -- Projection around refPoint on the buffer projection array
    float3 refPt = points[refPoint];
    for(int ptId = sliceThreadBeg; ptId < sliceThreadEnd; ptId++) {

        float3 pt = points[ptId];

        float deltaY = refPt.y - pt.y;
        float deltaX = refPt.x - pt.x;
        float dist = deltaY * deltaY + deltaX * deltaX;

        projected[ptId] = make_float3(deltaY, dist, pt.z);
    }

    // -- Sorting points according to the x axis

    // TODO BETTER SORT ALGORITHM
    if (threadIdx.x == 0) {
        for (int r=0; r < sliceBlockEnd - sliceBlockBeg; r++) {
            for (int n=sliceBlockBeg; n < sliceBlockEnd - r; n++) {
                if (projected[n].x > projected[n+1].x) {
                    float3 tmp = projected[n+1];
                    projected[n+1] = projected[n];
                    projected[n] = tmp;
                }
            }
        }
    }
    __syncthreads();

    // -- Lower Convex Hull (Sequential algorithm)
    // TODO
    if (threadIdx.x == 0) {
        paths[sliceThreadBeg] = make_int3(1, 1, 1);
    }
    
    printf("Round %d, Block %d, Th %d\t will project [%d, %d[ on %d\n", roundId, blockIdx.x, threadIdx.x,
        sliceThreadBeg, sliceThreadEnd, refPoint);
}

int3* createPaths(float3 *points, int nbPoints) {

    // Find the number of subproblems according to the threshold of the
    // maximum number of points per subproblems. This number is always a power of 2
    int nbSubproblems = 1, log2nbSubproblems = 0;
    while ((nbSubproblems * THRESHOLD) < nbPoints) {
        log2nbSubproblems++;
        nbSubproblems <<= 1;
    }

    std::cout << "nb & log2 " << nbSubproblems << " " << log2nbSubproblems << std::endl;

    // Create an array to store every paths
    int3* paths;    // log2(nbSubproblems) * nbPoints Array containing every paths.
                    // First line contains 1st path (max length nbPoints)
                    // Second line contains 2nd and 3rd path (each max length nbPoints/2)
                    // ... with every power of two
    cudaMalloc((void **)&paths, nbPoints * log2nbSubproblems * sizeof(int3));
    cudaMemset(paths, -1, nbPoints * log2nbSubproblems * sizeof(int3));

    // Create a buffer array to store projected points. Rewritten at each round
    float3* bufferProjection;
    cudaMalloc((void **)&bufferProjection, nbPoints * sizeof(float3));

    // Recursively find a path and cut into two subproblems
    for(int i=0, nbBlocks=1; nbBlocks < nbSubproblems; i++, nbBlocks<<= 1) {
        int nbThreads = NB_MAX_THREADS/nbBlocks; // Every block can be done in parallel
        if (nbThreads < 1) nbThreads = 1;

        // Project the complete array
        projectSlice<<<nbBlocks, nbThreads>>>(points, bufferProjection, &paths[nbPoints * i], nbPoints, i);
        cudaDeviceSynchronize();

        // DEBUG PRINT PROJECTED ARRAY
        float3 pt[nbPoints];
        cudaMemcpy(pt, bufferProjection, nbPoints * sizeof(float3), cudaMemcpyDeviceToHost);
        for (int i = 0; i < nbPoints; i++){
            std::cout << "Index :" << * (int *) &(pt[i].z) << " X :" << pt[i].x << " Y :" << pt[i].y << std::endl;
        } 
    }

    // DEBUG PRINT PROJECTED ARRAY
    int3 p[nbPoints * log2nbSubproblems];
    cudaMemcpy(p, paths, nbPoints * log2nbSubproblems * sizeof(int3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < log2nbSubproblems; i++) {
        for (int j = 0; j < nbPoints; j++) {
            if (p[i*nbPoints + j].z >= 0) {
                std::cout << "(" << p[i*nbPoints + j].x << "," << p[i*nbPoints + j].y << "," << p[i*nbPoints + j].z << ")";
            } else {
                std::cout << ".";
            }
        }
        std::cout << std::endl;
    }

    cudaFree(bufferProjection);
    return paths;
}