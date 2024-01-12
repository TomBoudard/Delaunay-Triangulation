// #define N 1024 // TODO WHICH VALUE?

#define ccw(A, B, C) (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)>0

// Each thread finds edges for a different projection
// __global__ void projectPoints(vertex *pts, vertex *projected, long unsigned int nbPts, bool onAxisX) {
//     long unsigned int idx = (blockIdx.x*N + threadIdx.x);
//     long unsigned int projectionIdx = idx / ((nbPts+N-1)/N);
//     long unsigned int refIdx = (nbPts * (1 + 2*projectionIdx)) / (2*N);

//     vertex refPoint = pts[refIdx];
//     vertex localPoint = pts[idx];
//     float deltaY = refPoint.y - localPoint.y;
//     float deltaX = refPoint.x - localPoint.x;

//     projected[idx].index = localPoint.index;
//     projected[idx].x = onAxisX ? deltaY : deltaX; // x takes delta between values over an axis
//     projected[idx].y = deltaY * deltaY + deltaX * deltaX; // y takes euclidian distance squared between points

//     printf("%u %f %f\n", projected[idx].index, projected[idx].x, projected[idx].y);
// }

// // Each thread finds edges for a different projection
// __global__ void projectPoints(vertex *pts, vertex *projected, long unsigned int nbPts, bool onAxisX) {
//     long unsigned int idx = (blockIdx.x*N + threadIdx.x);
//     long unsigned int projectionIdx = idx / ((nbPts+N-1)/N);
//     long unsigned int refIdx = (nbPts * (1 + 2*projectionIdx)) / (2*N);

//     vertex refPoint = pts[refIdx];
//     vertex localPoint = pts[idx];
//     float delta_y = refPoint.y - localPoint.y;
//     float delta_x = refPoint.x - localPoint.x;

//     projected[idx].index = localPoint.index;
//     projected[idx].x = onAxisX ? delta_y : delta_x; // x takes delta between values over an axis
//     projected[idx].y = delta_y * delta_y + delta_x * delta_x; // y takes euclidian distance squared between points
// }

// // // Each thread finds edges for a different projection
// // __global__ void lowerConvexHull(vertex *projected, long unsigned int nbPts) {
// //     long unsigned int projectionId = blockIdx.x;
// //     long unsigned int nbPtsPerProjection = (nbPts+N-1) / N;
// //     // TODO SORT FOR LOWER CONVEX HULL
// //     // for (long unsigned int i = projectionId * nbPtsPerProjection; i < (projectionId + 1) * nbPtsPerProjection; i++) {
// //     //     printf("Thread nb %lu has point nb %lu with projection %f\n", projectionId, i, projected[i].y);
// //     // }

// //     // TODO fix structure for ordered pts
// //     vertex* sortedByAngle;
// //     cudaMalloc((void **) &sortedByAngle, sizeof(vertex) * nbPtsPerProjection);

// //     // TODO fix ugly "structure" ?
// //     long unsigned int* stack;
// //     cudaMalloc((void **) &stack, sizeof(long unsigned int) * nbPtsPerProjection);
// //     unsigned int stackLen = 0;
// //     for (long unsigned int pt = 0; pt < nbPtsPerProjection; pt++) {
// //         // We pop if adding the point means a counterclockwise rotation
// //         while (stackLen > 1 && ccw(sortedByAngle[stack[stackLen]], sortedByAngle[stack[stackLen-1]], sortedByAngle[pt])) {
// //             stackLen--;
// //         }
// //         printf("trying to write %lu\n", pt);
// //         stack[stackLen] = pt;
// //         printf("it worked pog\n");
// //         stackLen++;
// //     }
// // }

// vertex* projection(vertex* pointsOnGPU, long unsigned int nbPts) {

//     long unsigned int mem = nbPts * sizeof(vertex);

//     vertex* pointsProjected;
//     cudaMalloc((void **)&pointsProjected, mem);
    
//     // Projection
//     float theTime;

//     cudaEvent_t myEvent, laterEvent;
//     cudaEventCreate(&myEvent);
//     cudaEventRecord(myEvent, 0);
//     cudaEventSynchronize(myEvent);
//     int dimGrid = (nbPts+N-1)/N;   // Nb of blocks
//     int dimBlock = N;
//     projectPoints<<<dimGrid, dimBlock>>>(pointsOnGPU, pointsProjected, nbPts, true);
//     cudaDeviceSynchronize();

//     cudaEventCreate(&laterEvent);
//     cudaEventRecord(laterEvent, 0);
//     cudaEventSynchronize(laterEvent);

//     // Delaunay edges
//     // Call "N" tasks in parallel
//     // lowerConvexHull<<<N, 1>>>(pointsProjected, nbPts);
//     // cudaDeviceSynchronize();

//     cudaEventElapsedTime(&theTime, myEvent, laterEvent);

    // printf("Algorithm took %f\n", theTime);

//     return pointsProjected;
// }