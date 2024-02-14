#include "tools.cu"

#define THRESHOLD 5 // TODO WHICH VALUE?
#define NB_MAX_THREADS 1024 // SHOULD BE POWER OF 2

// Macro to compare polar angle between (A and ref) and (B and ref)
#define biggerPolarAngle(A, B, ref) atan2(A.x - ref.x, A.y - ref.y) > atan2(B.x - ref.x, B.y - ref.y)
// Macro to test if we turn clockwise or anti-clockwise
#define ccw(A, B, C) (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)>0
// Macro to test if positions are equal
#define samePos(A, B) (A.x == B.x && A.y == B.y)

__global__ void projectSlice(float3 *points, float3 *buffers, struct edge *paths, int nbPoints, int roundId) {

    // Buffer contains multiple lines, each is a different buffer
    float3 *projected = buffers;
    float3 *pathsAsPoints = &buffers[2*nbPoints];

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


    // ---------------------------------------------------------------
    // -- Projection around refPoint on the buffer projection array --
    // ---------------------------------------------------------------
    float3 refPt = points[refPoint];
    for(int ptId = sliceThreadBeg; ptId < sliceThreadEnd; ptId++) {

        float3 pt = points[ptId];

        float deltaY = refPt.y - pt.y;
        float deltaX = refPt.x - pt.x;
        float dist = deltaY * deltaY + deltaX * deltaX;

        projected[ptId] = make_float3(deltaY, dist, pt.z);
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // -- Sorting points by polar angle with the highest x-coordinate point --
    // -----------------------------------------------------------------------

    // ## First step: Find the highest x-coordinate point (and highest y-coordinate if necessary)
    float3 rightmostPoint = projected[sliceBlockBeg];

    for (int r=sliceBlockBeg + 1; r < sliceBlockEnd; r++) {
        if ((projected[r].x > rightmostPoint.x) ||
            (projected[r].x == rightmostPoint.x && projected[r].y > rightmostPoint.y)) {
            rightmostPoint = projected[r];
        }
    }
    __syncthreads();

    // ## Second step: Sort points by polar angle with reference point
    // Done with fusion sort and by alternating between two buffers
    float3 *buff[2] = {buffers, &buffers[nbPoints]};
    int current_buffer = 0; // Starts on projected points buffer

    // Each core sorts its slice of points. Starts by merging arrays of length 1, then 2, 4, ...
    for (int size = 1; size < sliceThreadEnd - sliceThreadBeg; size <<= 1) {
        // For given length, sorts sequentially every array of this length in the slice
        for (int index=sliceThreadBeg; index < sliceThreadEnd; index += 2*size) {
            int firstPointer = index;
            int secondPointer = index + size;

            for (int i=index; i < min(sliceThreadEnd, index + 2 * size); i++) {
                if ((secondPointer == index + 2 * size || secondPointer >= sliceThreadEnd) ||
                    (!(firstPointer == index + size) && biggerPolarAngle(buff[current_buffer][secondPointer],
                                                                         buff[current_buffer][firstPointer],
                                                                         rightmostPoint))) {
                    buff[1-current_buffer][i] = buff[current_buffer][firstPointer];
                    firstPointer++;
                } else {
                    buff[1-current_buffer][i] = buff[current_buffer][secondPointer];
                    secondPointer++;
                }
            }
        }
        current_buffer = 1 - current_buffer;
    }
    __syncthreads();

    // Merging sorted slices (log2(blockDim.x) steps), each time using two times less cores
    for (uint r=2; r<=blockDim.x; r<<=1) {
        if (threadIdx.x%r == 0) {

            // Merges two sorted slices (first one starting at firstPointer, second at secondPointer)
            float ratio3 = (float) (threadIdx.x+(r>>1)) / blockDim.x;
            int firstPointer = sliceThreadBeg;
            const int secondPointerStart = (int) ((float) sliceBlockBeg * (1-ratio3) + (float) sliceBlockEnd * ratio3);
            int secondPointer = secondPointerStart;

            float ratio4 = (float) (threadIdx.x+r) / blockDim.x;
            int sliceEnd = (int) ((float) sliceBlockBeg * (1-ratio4) + (float) sliceBlockEnd * ratio4);

            for (int i=sliceThreadBeg; i < sliceEnd; i++) {
                if ((secondPointer >= sliceEnd) ||
                    (!(firstPointer == secondPointerStart) && biggerPolarAngle(buff[current_buffer][secondPointer],
                                                                        buff[current_buffer][firstPointer],
                                                                        rightmostPoint))) {
                    buff[1-current_buffer][i] = buff[current_buffer][firstPointer];
                    firstPointer++;
                } else {
                    buff[1-current_buffer][i] = buff[current_buffer][secondPointer];
                    secondPointer++;
                }
            }

        }
        __syncthreads();
        current_buffer = 1 - current_buffer;
    }

    // DEBUG PRINT
    if (threadIdx.x == 0) {
        for (int i=sliceBlockBeg; i<sliceBlockEnd; i++) {
            printf("%u %u (%f, %f)   \t(Angle : %f)\n", blockIdx.x,
                * (int*) &buff[current_buffer][i].z, buff[current_buffer][i].x, buff[current_buffer][i].y,
                atan2(buff[current_buffer][i].x - rightmostPoint.x, buff[current_buffer][i].y - rightmostPoint.y));
        }
    }

    // ------------------------------------------------------------
    // -- Lower Convex Hull (Sequential algorithm - Graham scan) --
    // ------------------------------------------------------------

    // (Writing the stack on a buffer then converting into a path of edges)
    if (threadIdx.x == 0) {
        // Find the lowest and highest x-coordinate (and highest y-coordinate to handle '==' case)
        // The algorithm will start from the rightmost point and end when the leftmost point is added
        int leftmostPointIndex = sliceBlockBeg;
        int rightmostPointIndex = sliceBlockBeg;
        float3* buffer_used = buff[current_buffer];

        for (int r=sliceBlockBeg + 1; r < sliceBlockEnd; r++) {
            if ((buffer_used[r].x < buffer_used[leftmostPointIndex].x) ||
                (buffer_used[r].x == buffer_used[leftmostPointIndex].x && buffer_used[r].y > buffer_used[leftmostPointIndex].y)) {
                leftmostPointIndex = r;
            } else if ((buffer_used[r].x > buffer_used[leftmostPointIndex].x) ||
                (buffer_used[r].x == buffer_used[leftmostPointIndex].x && buffer_used[r].y > buffer_used[leftmostPointIndex].y)) {
                rightmostPointIndex = r;
            }
        }

        pathsAsPoints[sliceBlockBeg] = buffer_used[rightmostPointIndex];
        int stackIndex = 1; // Used to track stack. Starts at one because we already filled first value
        int maxStackIndex = 1; // Used to clean stack at the end by rewriting end values if unused

        // Maximum number of loops is number of points - 1
        for (int pt=0; pt < sliceBlockEnd - sliceBlockBeg - 1; pt++) {
            // Skip rightmostPointIndex
            if (sliceBlockBeg + pt == rightmostPointIndex) continue;

            // Pop points from stack until we turn clockwise for the next point
            while (stackIndex > 1 && (ccw(pathsAsPoints[sliceBlockBeg + stackIndex - 2],
                                         pathsAsPoints[sliceBlockBeg + stackIndex - 1],
                                         buffer_used[sliceBlockBeg + pt]) ||
                                     samePos(pathsAsPoints[sliceBlockBeg + stackIndex - 1],
                                             buffer_used[sliceBlockBeg + pt]))) {
                // printf("Removing point %u from path\n", * (int*) &buffer_used[sliceBlockBeg + stackIndex - 1].z);
                stackIndex--;
            }
            // printf("Adding point %u to path\n", * (int*) &buffer_used[sliceBlockBeg + pt].z);

            // Add new point to path
            pathsAsPoints[sliceBlockBeg + stackIndex] = buffer_used[sliceBlockBeg + pt];
            stackIndex++;
            if (stackIndex > maxStackIndex) maxStackIndex++;

            // End loop if we hit the last point of the path
            if (sliceBlockBeg + pt == leftmostPointIndex) {
                //printf("End of path ! Added point %u\n", * (int*) &buffer_used[leftmostPointIndex].z);
                break;
            }
        }

        // printf("Path done ! Length : %u / %u\t (Went to %u)\n",
        //        stackIndex, sliceBlockEnd - sliceBlockBeg, maxStackIndex);

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
    // Buffer 1&2 :  used for projection & sorting (Rewritten at each round)
    // Buffer 3:     stacks used to create the list of points used for a path
    float3* buffers;
    cudaMalloc((void **)&buffers, 3 * nbPoints * sizeof(float3));

    // Recursively find a path and cut into two subproblems
    for(int i=0, nbBlocks=1; nbBlocks < nbSubproblems; i++, nbBlocks<<= 1) {
        int nbThreads = NB_MAX_THREADS/nbBlocks; // Every block can be done in parallel
        if (nbThreads < 1) nbThreads = 1;

        // Project the complete array
        projectSlice<<<nbBlocks, nbThreads>>>(points, buffers, &paths[nbPoints * i], nbPoints, i);
        cudaDeviceSynchronize();
    }

    // // DEBUG PRINT PATHS
    // struct edge p[nbPoints * log2nbSubproblems];
    // cudaMemcpy(p, paths, nbPoints * log2nbSubproblems * sizeof(struct edge), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < log2nbSubproblems; i++) {
    //     for (int j = 0; j < nbPoints; j++) {
    //         edge e = p[i*nbPoints + j];
    //         if (e.usage == UNUSED) {
    //             std::cout << "(" << *(int *) &(e.x.z) << " " << *(int *) &(e.y.z) << ")";
    //         } else if (e.usage == INVALID) {
    //             std::cout << "|";
    //         } else {
    //             std::cout << ".";
    //         }
    //     }
    //     std::cout << std::endl;
    // }

    cudaFree(buffers);
    return paths;
}
