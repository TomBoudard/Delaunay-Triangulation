#include "tools.cu"

#define THRESHOLD 10 // TODO WHICH VALUE?
#define NB_MAX_THREADS 1024 // SHOULD BE POWER OF 2

// Macro to compare polar angle between (A and ref) and (B and ref)
#define biggerPolarAngle(A, B, ref) atan2(A.x - ref.x, A.y - ref.y) > atan2(B.x - ref.x, B.y - ref.y)
// Macro to test if we turn clockwise or anti-clockwise
#define ccw(A, B, C) (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)>0

__global__ void projectSlice(float3 *points, float3 *buffers, struct edge *paths, int nbPoints, int roundId) {

    // Buffer contains multiple lines, each is a different buffer
    float3 *projected = buffers;
    float3 *pathsAsPoints = &buffers[nbPoints];

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
    // printf("Round %d, Block %d, Th %d\t will project [%d, %d[ on %d\n", roundId, blockIdx.x, threadIdx.x,
    //     sliceThreadBeg, sliceThreadEnd, refPoint);

    // -- Sorting points by polar angle
    // TODO BETTER SORT ALGORITHM
    if (threadIdx.x == 0) {
        // Find the lowest x-coordinate (and highest y-coordinate if necessary)
        float3 rightmostPoint = projected[sliceBlockBeg];

        for (int r=sliceBlockBeg + 1; r < sliceBlockEnd; r++) {
            if ((projected[r].x > rightmostPoint.x) ||
                (projected[r].x == rightmostPoint.x && projected[r].y > rightmostPoint.y)) {
                rightmostPoint = projected[r];
            }
        }

        // Sort points by polar angle with leftmost point
        // TODO "if several points have the same polar angle then only keep the farthest"
        for (int r=0; r < sliceBlockEnd - sliceBlockBeg; r++) {
            for (int n=sliceBlockBeg; n < sliceBlockEnd - r; n++) {
                if (biggerPolarAngle(projected[n], projected[n+1], rightmostPoint)) {
                    float3 tmp = projected[n+1];
                    projected[n+1] = projected[n];
                    projected[n] = tmp;
                }
            }
        }
    }
    __syncthreads();

    // -- Lower Convex Hull (Sequential algorithm - Graham scan)
    // Writing the stack on a buffer then converting into a path of edges
    if (threadIdx.x == 0) {
        // Find the lowest and highest x-coordinate (and highest y-coordinate to handle '==' case)
        // The algorithm will start from the rightmost point and end when the leftmost point is added
        int leftmostPointIndex = sliceBlockBeg;
        int rightmostPointIndex = sliceBlockBeg;

        for (int r=sliceBlockBeg + 1; r < sliceBlockEnd; r++) {
            if ((projected[r].x < projected[leftmostPointIndex].x) ||
                (projected[r].x == projected[leftmostPointIndex].x && projected[r].y > projected[leftmostPointIndex].y)) {
                leftmostPointIndex = r;
            } else if ((projected[r].x > projected[leftmostPointIndex].x) ||
                (projected[r].x == projected[leftmostPointIndex].x && projected[r].y > projected[leftmostPointIndex].y)) {
                rightmostPointIndex = r;
            }
        }

        pathsAsPoints[sliceBlockBeg] = projected[rightmostPointIndex];
        int stackIndex = 1; // Used to track stack. Starts at one because we already filled first value
        int maxStackIndex = 1; // Used to clean stack at the end by rewriting end values if unused

        for (int pt=0; pt < sliceBlockEnd - sliceBlockBeg - 1; pt++) {  // Maximum number of loops is number of points - 1
            // Skip rightmostPointIndex
            if (sliceBlockBeg + pt == rightmostPointIndex) continue;

            // Pop points from stack until we turn clockwise for the next point
            while (stackIndex > 1 && ccw(pathsAsPoints[sliceBlockBeg + stackIndex - 2],
                                         pathsAsPoints[sliceBlockBeg + stackIndex - 1],
                                         projected[sliceBlockBeg + pt])) {
                //printf("Removing point %u from path\n", * (int*) &projected[sliceBlockBeg + stackIndex - 1].z);
                stackIndex--;
            }
            //printf("Adding point %u to path\n", * (int*) &projected[sliceBlockBeg + pt].z);

            // Add new point to path
            pathsAsPoints[sliceBlockBeg + stackIndex] = projected[sliceBlockBeg + pt];
            stackIndex++;
            if (stackIndex > maxStackIndex) maxStackIndex++;

            // End loop if we hit the last point of the path
            if (sliceBlockBeg + pt == leftmostPointIndex) {
                //printf("End of path ! Added point %u\n", * (int*) &projected[leftmostPointIndex].z);
                break;
            }
        }

        printf("Path done ! Length : %u / %u\t (Went to %u)\n",
               stackIndex, sliceBlockEnd - sliceBlockBeg, maxStackIndex);

        // We know for such the path has at least one edges so two points
        float3 prevPoint;
        float3 curPoint = pathsAsPoints[sliceBlockBeg];

        for (int i=sliceBlockBeg; i<sliceBlockBeg+stackIndex-1; i++) {
            prevPoint = curPoint;
            curPoint = pathsAsPoints[i+1];
            paths[i] = {prevPoint, curPoint, UNUSED};
        }

        paths[sliceBlockBeg+stackIndex-1] = {prevPoint, curPoint, INVALID};
    }
    __syncthreads();
}

struct edge* createPaths(float3 *points, int nbPoints) {

    // Find the number of subproblems according to the threshold of the
    // maximum number of points per subproblems. This number is always a power of 2
    int nbSubproblems = 1, log2nbSubproblems = 0;
    while ((nbSubproblems * THRESHOLD) < nbPoints) {
        log2nbSubproblems++;
        nbSubproblems <<= 1;
    }

    std::cout << "nb & log2 " << nbSubproblems << " " << log2nbSubproblems << std::endl;

    // Create an array to store every paths.
    struct edge* paths;   // log2(nbSubproblems) * nbPoints Array containing every edges.
                          // (There are at most x points per line so x-1 edges + a final edge stored to tell it's the end of the path)
                          // First line contains 1st path (max length nbPoints)
                          // Second line contains 2nd and 3rd path (each max length nbPoints/2)
                          // ... with every power of two
    cudaMalloc((void **)&paths, nbPoints * log2nbSubproblems * sizeof(struct edge));

    // Create a buffer array to store different data
    // First line:  projected points (Rewritten at each round)
    // Second line: stacks used to create the list of points used for a path
    float3* buffers;
    cudaMalloc((void **)&buffers, 2 * nbPoints * sizeof(float3));

    // Recursively find a path and cut into two subproblems
    for(int i=0, nbBlocks=1; nbBlocks < nbSubproblems; i++, nbBlocks<<= 1) {
        int nbThreads = NB_MAX_THREADS/nbBlocks; // Every block can be done in parallel
        if (nbThreads < 1) nbThreads = 1;

        // Project the complete array
        projectSlice<<<nbBlocks, nbThreads>>>(points, buffers, &paths[nbPoints * i], nbPoints, i);
        cudaDeviceSynchronize();

        // // DEBUG PRINT PROJECTED ARRAY
        // float3 pt[nbPoints];
        // cudaMemcpy(pt, bufferProjection, nbPoints * sizeof(float3), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < nbPoints; i++){
        //     std::cout << "Index :" << * (int *) &(pt[i].z) << " X :" << pt[i].x << " Y :" << pt[i].y;
        //     std::cout << " Polar angle with last point :" << atan2(pt[i].x - pt[nbPoints-1].x, pt[i].y - pt[nbPoints-1].y) << std::endl;
        // } 
    }

    // DEBUG PRINT PROJECTED ARRAY
    struct edge p[nbPoints * log2nbSubproblems];
    cudaMemcpy(p, paths, nbPoints * log2nbSubproblems * sizeof(struct edge), cudaMemcpyDeviceToHost);
    for (int i = 0; i < log2nbSubproblems; i++) {
        for (int j = 0; j < nbPoints; j++) {
            edge e = p[i*nbPoints + j];
            if (e.usage == UNUSED) {
                std::cout << "(" << *(int *) &(e.x.z) << " " << *(int *) &(e.y.z) << ")";
            } else if (e.usage == INVALID) {
                std::cout << "|";
            } else {
                std::cout << ".";
            }
        }
        std::cout << std::endl;
    }
    cudaFree(buffers);

    return paths;
}
