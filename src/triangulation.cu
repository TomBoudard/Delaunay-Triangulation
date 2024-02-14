#include "tools.cu"
#include <math_functions.h>

__device__ float delaunayDistance(float3 const& edgeStart, float3 const& edgeEnd, float3 const& point){

    float oppEdgeA = length(point-edgeEnd); 
    float oppEdgeB = length(edgeStart-point); 
    float oppEdgeC = length(edgeEnd-edgeStart);

    float barCircumCenterA = oppEdgeA*(oppEdgeB+oppEdgeC-oppEdgeA);
    float barCircumCenterB = oppEdgeB*(oppEdgeC+oppEdgeA-oppEdgeB);
    float barCircumCenterC = oppEdgeC*(oppEdgeA+oppEdgeB-oppEdgeC);

    float3 circumCenter;
    float sumBarycenters = barCircumCenterA + barCircumCenterB + barCircumCenterC;
    circumCenter.x = (barCircumCenterA*edgeStart.x + barCircumCenterB*edgeEnd.x + barCircumCenterC*point.x)/sumBarycenters;
    circumCenter.y = (barCircumCenterA*edgeStart.y + barCircumCenterB*edgeEnd.y + barCircumCenterC*point.y)/sumBarycenters;

    float radius = sqrt(length(circumCenter-point));

    if (barCircumCenterC < 0){ //Check whether the circumCenter is in the half space of the point or not
        return -radius;
    }
    else{
        return radius;
    }
}

__global__ void parDeTri(float3* points, edge* edgePathList, edge* globalEdgeList, int3* triangleList, int nbPoints, int nbSubproblems, int nbMaxTriangle){ //TODO Verify if possible to use reference //TODO Avoid if/else as much as possible
    
    printf("TEST START !\n");

    // IDs of the beginning and end of the slice manipulated by the block
    int sliceBlockBeg = (blockIdx.x) * nbPoints / (nbSubproblems);
    int sliceBlockEnd = ((blockIdx.x + 1)) * nbPoints / (nbSubproblems);

    int idLeft = ((int)log2((double)nbSubproblems) - (__ffs(blockIdx.x) - 1) - 1)*nbPoints + blockIdx.x%((int)log2((double)nbSubproblems) - (__ffs(blockIdx.x) - 1))*nbPoints/((int)log2((double)nbSubproblems) - (__ffs(blockIdx.x) - 1));
    int idRight = ((int)log2((double)nbSubproblems) - (__ffs(blockIdx.x + 1) - 1) - 1)*nbPoints + (blockIdx.x+1)%((int)log2((double)nbSubproblems) - (__ffs(blockIdx.x + 1) - 1))*nbPoints/((int)log2((double)nbSubproblems) - (__ffs(blockIdx.x + 1) - 1));

    int boundMaxEdgePerSubset = (int)(2*nbPoints/nbSubproblems - 2)*3*3;
    int copyIndex = blockIdx.x*boundMaxEdgePerSubset*nbSubproblems;

    if (blockIdx.x != 0){
        while (edgePathList[idLeft].usage != INVALID){
            globalEdgeList[copyIndex] = edgePathList[idLeft];
            globalEdgeList[copyIndex].usage = UNUSED_LEFT;
            printf("Block N°%d | CHECK VALUE EDGE INDEX LEFT: %d \n",blockIdx.x, idLeft);
            idLeft++;
            copyIndex++;
        }
    }
    if (blockIdx.x != (nbSubproblems - 1)){
        while (edgePathList[idRight].usage != INVALID){
            globalEdgeList[copyIndex] = edgePathList[idRight];
            globalEdgeList[copyIndex].usage = UNUSED_RIGHT;
            printf("Block N°%d | CHECK VALUE EDGE INDEX RIGHT: %d \n",blockIdx.x, idRight);
            idRight++;
            copyIndex++;
        }
    }

    int triangleIndex = (blockIdx.x)*nbMaxTriangle;

    int startEdgeIndex = blockIdx.x*boundMaxEdgePerSubset*nbSubproblems;
    int endEdgeIndex = startEdgeIndex + copyIndex;

    edge currentEdge;
    float3 bestThirdPoint;
    int bestThirdPointSide;

    printf("TEST TRIANGULATION BEGIN !\n");

    //Triangulation
    while (startEdgeIndex < endEdgeIndex){

        currentEdge = globalEdgeList[startEdgeIndex];

        printf("Block N°%d | Current edge used: %d -> %d \n",blockIdx.x, *(int*)& currentEdge.x.z, *(int*)& currentEdge.y.z);

        float bestRadius = INFINITY;
        bool triangleFound = false;
        for (int i = sliceBlockBeg; i<sliceBlockEnd; i++){
            float3 firstVector = currentEdge.y-currentEdge.x;
            float3 secondVector = points[i]-currentEdge.x;
            float zVectorialProduct = firstVector.x*secondVector.y - firstVector.y*secondVector.x;
            int pointSide = zVectorialProduct/fabs(zVectorialProduct);

            if(points[i].z != currentEdge.x.z && points[i].z != currentEdge.y.z && pointSide != currentEdge.usage && currentEdge.usage != FULL && !(currentEdge.usage == UNUSED_LEFT && pointSide != -1) && !(currentEdge.usage == UNUSED_RIGHT && pointSide != 1)){
                float radius = delaunayDistance(currentEdge.x, currentEdge.y, points[i]);                
                if (radius < bestRadius){
                    int3 currentTriangle = make_int3(currentEdge.x.z, currentEdge.y.z, points[i].z);
                    bool alreadyExisting = false;
                    for (int j = blockIdx.x * nbMaxTriangle; j<triangleIndex; j++){
                        if (currentTriangle == triangleList[j]){
                            alreadyExisting = true;
                        }
                    }
                    if (!alreadyExisting){
                        bestRadius = radius;
                        bestThirdPoint = points[i];
                        bestThirdPointSide = pointSide;
                        triangleFound = true;
                    }
                }
            }
        }

        if (triangleFound){
            
            printf("TEST TRIANGLE FOUND : CHECK VALIDITY !\n");
            printf("Block N°%d | THIRD POINT: %d \n", blockIdx.x, *(int*)& bestThirdPoint.z);
            printf("AHAHAAH: %d \n", *(int*)& currentEdge.x.z);

            bool validTriangle = true;
            if (bestThirdPointSide == -1){//Means that the current edge is being used from y to x to be used in a direct repere
                float3 temp = currentEdge.x;
                currentEdge.x = currentEdge.y;
                currentEdge.y = temp;
            }
            edge secondEdge = {currentEdge.y, bestThirdPoint, 0};
            edge thirdEdge = {bestThirdPoint, currentEdge.x, 0};
            bool secondEdgeNew = true;
            bool thirdEdgeNew = true;
            int indexSecondEdge = -1; //TODO To optimize
            int indexThirdEdge = -1;
            for (int k = 0; k<=endEdgeIndex; k++){ // TODO To optimize if possible
                if (secondEdge == globalEdgeList[k]){
                    secondEdgeNew = false;
                    secondEdge = globalEdgeList[k];
                    indexSecondEdge = k;
                }
                if (thirdEdge == globalEdgeList[k]){
                    thirdEdgeNew = false;
                    thirdEdge = globalEdgeList[k];
                    indexThirdEdge = k;
                }
            }

            //The case where the edge has two points on the same side is not possible here otherwise the bestThirdPoint wouldn't have been choosen

            //Add the two side edges only if they are new (the first one is always preexisting)
            if (secondEdgeNew){
                secondEdge.usage = USED; //New edges are created according to a direct repere 
                globalEdgeList[endEdgeIndex] = secondEdge;
                endEdgeIndex++;
            }
            else{
                float3 firstVectorSecondEdge = secondEdge.y-secondEdge.x;
                float3 secondVectorSecondEdge = currentEdge.x-currentEdge.y;
                float zVectorialProductSecondEdge = firstVectorSecondEdge.x*secondVectorSecondEdge.y - firstVectorSecondEdge.y*secondVectorSecondEdge.x;
                if (zVectorialProductSecondEdge == 1 || secondEdge.usage == FULL){ //If used in the same way as created or already used twice
                    validTriangle = false;
                }
                if (secondEdge.usage == UNUSED_LEFT || secondEdge.usage == UNUSED_RIGHT){ // Special case happening only for the edges from the path
                    secondEdge.usage = FULL; //Because only one side can be used by the subset
                    globalEdgeList[indexThirdEdge].usage = secondEdge.usage;
                }
                else{
                    secondEdge.usage = FULL;
                    globalEdgeList[indexSecondEdge].usage = secondEdge.usage;
                }
            }

            if (thirdEdgeNew){
                thirdEdge.usage = USED; //New edges are created according to a direct repere 
                globalEdgeList[endEdgeIndex] = thirdEdge;
                endEdgeIndex++;
            }
            else{
                float3 firstVectorThirdEdge = thirdEdge.y-thirdEdge.x;
                float3 secondVectorThirdEdge = currentEdge.y-bestThirdPoint;
                float zVectorialProductThirdEdge = firstVectorThirdEdge.x*secondVectorThirdEdge.y - firstVectorThirdEdge.y*secondVectorThirdEdge.x;
                if (zVectorialProductThirdEdge == 1 || thirdEdge.usage == FULL){ //If they have the same sign
                    validTriangle = false;
                }
                if (thirdEdge.usage == UNUSED_LEFT || thirdEdge.usage == UNUSED_RIGHT){ // Special case happening only for the edges from the path
                    thirdEdge.usage = FULL;
                    globalEdgeList[indexThirdEdge].usage = thirdEdge.usage;
                }
                else{
                    thirdEdge.usage = FULL;
                    globalEdgeList[indexThirdEdge].usage = thirdEdge.usage;
                }
            }

            if(validTriangle){
                // std::cout << "THE TRIANGLE IS VALID" << std::endl;
                triangleList[triangleIndex].x = *(int*)& currentEdge.x.z;
                triangleList[triangleIndex].y = *(int*)& currentEdge.y.z;
                triangleList[triangleIndex].z = *(int*)& bestThirdPoint.z; //TODO STORE ONLY DIRECT TIRANGLE
                printf("STORE TRIANGLE : %d %d %d\n", triangleList[triangleIndex].x, triangleList[triangleIndex].y, triangleList[triangleIndex].z);
                triangleIndex++;

                if (triangleIndex > (blockIdx.x+1)*nbMaxTriangle){
                    printf("/!\\ STOP : MAXIMUM AMOUNT OF TRIANGLE PER SUBSET EXCEEDED /!\\ \n");
                }
            }
        }
        // std::cout << "======================" << std::endl;
        startEdgeIndex++;
    }
}