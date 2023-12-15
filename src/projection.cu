#define N 1024 // TODO WHICH VALUE?

__global__ void projectPoint(point2D *pts, point2D *projected, long unsigned int nbPts, bool onAxisX) {
    long unsigned int idx = (blockIdx.x*N + threadIdx.x);
    long unsigned int projectionIdx = idx / ((nbPts+N-1)/N);
    long unsigned int refIdx = (nbPts * (1 + 2*projectionIdx)) / (2*N);

    point2D refPoint = pts[refIdx];
    point2D localPoint = pts[idx];
    float delta_y = refPoint.y - localPoint.y;
    float delta_x = refPoint.x - localPoint.x;

    projected[idx].index = localPoint.index;
    projected[idx].x = onAxisX ? delta_y : delta_x; // x takes delta between values over an axis
    projected[idx].y = delta_y * delta_y + delta_x * delta_x; // y takes euclidian distance squared between points
}

point2D* projection(point2D* pointsOnGPU, long unsigned int nbPts) {

    long unsigned int mem = nbPts * sizeof(point2D);

    point2D* pointsProjected;
    cudaMalloc((void **)&pointsProjected, mem);
    
    int dimGrid = (nbPts+N-1)/N;   // Nb of blocks
    int dimBlock = N;
    projectPoint<<<dimGrid, dimBlock>>>(pointsOnGPU, pointsProjected, nbPts, true);

    cudaDeviceSynchronize();

    return pointsProjected;
}