#include "tools.cu"
#include <cuda_runtime.h>

#define sub(a, b)make_float3(a.x - b.x, a.y - b.y, 0)

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

__global__ void parDeTri(float3* points, edge* edgePathList, edge* globalEdgeList, int3* triangleList, int nbPoints, int nbSubproblems, int nbMaxTriangle){
    
    // IDs of the beginning and end of the slice manipulated by the block
    int sliceBlockBeg = (blockIdx.x) * nbPoints / (nbSubproblems);
    int sliceBlockEnd = ((blockIdx.x + 1)) * nbPoints / (nbSubproblems);

    float mostLeftX = points[sliceBlockBeg].x;
    float mostRightX = points[sliceBlockEnd].x;

    int nbLeftNeighbours = 0;
    int nbRightNeighbours = 0;

    int log2nbSubproblems = __ffs(nbSubproblems) - 1;

    // Compute the ids of the (at most) two path(s)
    int idLeftRow = (log2nbSubproblems - (__ffs(blockIdx.x)));
    int idLeftCol = (blockIdx.x - (1 << (log2nbSubproblems - idLeftRow-1))) / (1 << (log2nbSubproblems - idLeftRow)) * nbPoints / (1 << idLeftRow);
    int idRightRow = (log2nbSubproblems - (__ffs(blockIdx.x + 1)));
    int idRightCol = (blockIdx.x+1 - (1 << (log2nbSubproblems - idRightRow-1))) / (1 << (log2nbSubproblems - idRightRow)) * nbPoints / (1 << idRightRow);

    int idLeft = idLeftRow * nbPoints + idLeftCol;
    int idRight = idRightRow * nbPoints + idRightCol;

    int boundMaxEdgePerSubset = (int)(2*nbPoints/nbSubproblems - 2)*3*3;
    int copyIndex = blockIdx.x*boundMaxEdgePerSubset*nbSubproblems;

    if (blockIdx.x != 0){
        while (edgePathList[idLeft].usage != INVALID){
            globalEdgeList[copyIndex] = edgePathList[idLeft];
            globalEdgeList[copyIndex].usage = UNUSED_LEFT;
            idLeft++;
            copyIndex++;
        }
    }
    if (blockIdx.x != (nbSubproblems - 1)){
        while (edgePathList[idRight].usage != INVALID){
            globalEdgeList[copyIndex] = edgePathList[idRight];
            globalEdgeList[copyIndex].usage = UNUSED_RIGHT;
            idRight++;
            copyIndex++;
        }
    }

    int triangleIndex = (blockIdx.x)*nbMaxTriangle;

    int initialStartEdgeIndex = blockIdx.x*boundMaxEdgePerSubset*nbSubproblems;
    int startEdgeIndex = initialStartEdgeIndex;
    int endEdgeIndex = copyIndex;

    edge currentEdge;
    float3 bestThirdPoint;
    int bestThirdPointSide;

    //Triangulation
    while (startEdgeIndex < endEdgeIndex){ //Sliding window

        currentEdge = globalEdgeList[startEdgeIndex];

        float bestRadius = INFINITY;
        bool triangleFound = false;
        for (int i = sliceBlockBeg - nbLeftNeighbours; i<sliceBlockEnd + nbRightNeighbours; i++){
            float3 firstVector = currentEdge.y - currentEdge.x;
            float3 secondVector = points[i] - currentEdge.x;
            float zVectorialProduct = firstVector.x*secondVector.y - firstVector.y*secondVector.x;
            int pointSide = 0;
            if (zVectorialProduct > 0){
                pointSide = 1;
            }
            else if(zVectorialProduct < 0){
                pointSide = -1;
            }

            if(points[i].z != currentEdge.x.z && points[i].z != currentEdge.y.z && pointSide != 0
                && pointSide != currentEdge.usage && currentEdge.usage != FULL
                && !(currentEdge.usage == UNUSED_LEFT && pointSide == 1)
                && !(currentEdge.usage == UNUSED_RIGHT && pointSide == -1)){
                    
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
            
            bool validTriangle = true;
            if (bestThirdPointSide == -1){ //Means that the current edge must be used from y to x to be used in a direct repere
                float3 temp = currentEdge.x;
                currentEdge.x = currentEdge.y;
                currentEdge.y = temp;
            }
            edge secondEdge = {currentEdge.y, bestThirdPoint, 0};
            edge thirdEdge = {bestThirdPoint, currentEdge.x, 0};
            bool secondEdgeNew = true;
            bool thirdEdgeNew = true;
            int indexSecondEdge = -1;
            int indexThirdEdge = -1;
            for (int k = initialStartEdgeIndex; k<=endEdgeIndex; k++){
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
                else{
                    secondEdge.usage = FULL; // The use of a preexisting edge necessarily make it used twice (on both sides) or an edge of the path can be used only once in a subproblem
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
                if (zVectorialProductThirdEdge == 1 || thirdEdge.usage == FULL){ //If used in the same way as created or already used twice
                    validTriangle = false;
                }
                else{
                    thirdEdge.usage = FULL;  // The use of a preexisting edge necessarily make it used twice (on both sides) or an edge of the path can be used only once in a subproblem
                    globalEdgeList[indexThirdEdge].usage = thirdEdge.usage;
                }
            }

            if(validTriangle){
                currentEdge.usage = FULL; // The use of a preexisting edge necessarily make it used twice (on both sides) or an edge of the path can be used only once in a subproblem
                globalEdgeList[startEdgeIndex].usage = currentEdge.usage;

                triangleList[triangleIndex].x = *(int*)& currentEdge.x.z;
                triangleList[triangleIndex].y = *(int*)& currentEdge.y.z;
                triangleList[triangleIndex].z = *(int*)& bestThirdPoint.z;

                triangleIndex++;
                triangleList[triangleIndex].x = -1;

                if (triangleIndex > (blockIdx.x+1)*nbMaxTriangle){
                    printf("/!\\ STOP : MAXIMUM AMOUNT OF TRIANGLE PER SUBSET EXCEEDED /!\\ \n");
                }
            }
        }
        startEdgeIndex++;
    }
}