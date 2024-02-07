#define THRESHOLD 16 // TODO WHICH VALUE?
#define NB_MAX_THREADS 4

__global__ void addLowerConvexHull(int3 *paths, int nbPoints, int pathLen, int roundId) {

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

    // TODO project around refPoint on projection array
    printf("Round %d, Block %d, Th %d\t will project [%d, %d[ on %d\n", roundId, blockIdx.x, threadIdx.x,
        sliceThreadBeg, sliceThreadEnd, refPoint);
    
}

void createPaths(int nbPoints) {

    // Find the number of subproblems according to the threshold of the
    // maximum number of points per subproblems. This number is always a power of 2
    int nbSubproblems = 1;
    while ((nbSubproblems * THRESHOLD) < nbPoints) nbSubproblems <<= 1;

    // Create an array to store every paths
    int pathLen = nbPoints/nbSubproblems; // Maximum number of elements in a path of delaunay edges (TODO find good number)
    int nbPaths = nbSubproblems - 1; // Number of paths

    int3* paths;    // Array containing every paths. They start at pathLen * pathID
    cudaMalloc((void **)&paths, pathLen * nbPaths * sizeof(int3));

    // Recursively find a path and cut into two subproblems
    for(int i=0, nbBlocks=1; nbBlocks <= nbSubproblems; i++, nbBlocks<<= 1) {
        int nbThreads = NB_MAX_THREADS/nbBlocks; // Every block can be done in parallel
        if (nbThreads < 1) nbThreads = 1;

        addLowerConvexHull<<<nbBlocks, nbThreads>>>(paths, nbPoints, pathLen, i);
        cudaDeviceSynchronize();
    }

    cudaFree(paths);
}