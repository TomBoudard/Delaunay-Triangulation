#define N 1024 // TODO WHICH VALUE?

__global__ void projectPoint(point2D *pts, point2D *projected, long unsigned int refIdx, bool onAxisX) {
    point2D refPoint = pts[refIdx];
    long unsigned int idx = (blockIdx.x*N + threadIdx.x);
    point2D localPoint = pts[idx];
    float delta_y = refPoint.y - localPoint.y;
    float delta_x = refPoint.x - localPoint.x;

    projected[idx].index = localPoint.index;
    // x takes delta between y values
    projected[idx].x = onAxisX ? delta_y : delta_x;
    // y takes euclidian distance squared between points
    projected[idx].y = delta_y * delta_y + delta_x * delta_x;
}

point2D* projection(point2D* pointsOnGPU, long unsigned int nbPts) {

    long unsigned int mem = nbPts * sizeof(point2D);

    point2D* pointsProjected;
    cudaMalloc((void **)&pointsProjected, mem);
    
    dim3 dimGrid((nbPts+N-1)/N, 1);   // Nb of blocks
    dim3 dimBlock(N, 1);
    projectPoint<<<dimGrid, dimBlock>>>(pointsOnGPU, pointsProjected, nbPts/2, true);

    cudaDeviceSynchronize();

    return pointsProjected;
}