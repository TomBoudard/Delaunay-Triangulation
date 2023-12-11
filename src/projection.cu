#define N 1024 // TODO WHICH VALUE?

__global__ void projectPoint(point2D *pts, float ref_point_x, float ref_point_y, bool onAxisX) {
    point2D* localPoint = &pts[(blockIdx.x*N + threadIdx.x)];

    float old_x = localPoint->x;
    float old_y = localPoint->y;

    float delta_y = ref_point_y - old_y;
    float delta_x = ref_point_x - old_x;

    // x takes delta between y values
    localPoint->x = onAxisX ? delta_y : delta_x;
    // y takes euclidian distance between points
    localPoint->y = delta_y * delta_y + delta_x * delta_x;
}

point2D* projection(std::vector<point2D> pointsVector) {
    point2D *res;

    long unsigned int mem = sizeof(point2D) * pointsVector.size();
    cudaMalloc((void**)&res, mem);
    cudaMemcpy(res, &pointsVector[0], mem, cudaMemcpyHostToDevice);

    point2D midPoint = pointsVector[pointsVector.size()/2];

    dim3 dimGrid((pointsVector.size()+N-1)/N, 1);   // Nb of blocks
    dim3 dimBlock(N, 1);
    projectPoint<<<dimGrid, dimBlock>>>(res, midPoint.x, midPoint.y, true);

    cudaDeviceSynchronize();

    point2D* projection = new point2D[pointsVector.size()]; // projection results
    cudaMemcpy(projection, res, mem, cudaMemcpyDeviceToHost);

    // for (int i=0; i<pointsVector.size(); i++) {
    //     std::cout << "Original: " << pointsVector[i].x << " " << pointsVector[i].y << std::endl;
    //     std::cout << "Projected: " << projection[i].x << " " << projection[i].y << std::endl;
    // }

    return res;
}