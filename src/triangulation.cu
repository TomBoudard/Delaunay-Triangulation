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

__global__ void parDeTri(float3* points, edge* edgePathList, edge* globalEdgeList, int3* triangleList, int nbPoints, int nbSubproblems, int nbMaxTriangle, int fakeBlockIdx){
    
    printf("TEST START !\n");

    // IDs of the beginning and end of the slice manipulated by the block
    int sliceBlockBeg = (fakeBlockIdx) * nbPoints / (nbSubproblems);
    int sliceBlockEnd = ((fakeBlockIdx + 1)) * nbPoints / (nbSubproblems);

    float mostLeftX = points[sliceBlockBeg].x;
    float mostRightX = points[sliceBlockEnd].x;

    int idLeft = ((int)log2((double)nbSubproblems) - (__ffs(fakeBlockIdx) - 1) - 1)*nbPoints;
    idLeft += fakeBlockIdx%((int)log2((double)nbSubproblems) - (__ffs(fakeBlockIdx) - 1))*nbPoints/(1<<((int)log2((double)nbSubproblems) - (__ffs(fakeBlockIdx) - 1) - 1));
    int idRight = ((int)log2((double)nbSubproblems) - (__ffs(fakeBlockIdx + 1) - 1) - 1)*nbPoints;
    idRight += fakeBlockIdx%((int)log2((double)nbSubproblems) - (__ffs(fakeBlockIdx + 1) - 1))*nbPoints/(1<<((int)log2((double)nbSubproblems) - (__ffs(fakeBlockIdx + 1) - 1) - 1));
    
    printf("TEST ID LEFT = %d \n", ((int)log2((double)nbSubproblems) - (__ffs(fakeBlockIdx) - 1) - 1));
    printf("TEST ID Right = %d\n", ((int)log2((double)nbSubproblems) - (__ffs(fakeBlockIdx + 1) - 1) - 1));
    printf("%d \n",(fakeBlockIdx+1)%((int)log2((double)nbSubproblems) - (__ffs(fakeBlockIdx + 1) - 1)));
    printf("%d \n",((int)log2((double)nbSubproblems) - (__ffs(fakeBlockIdx + 1) - 1)));


    int boundMaxEdgePerSubset = (int)(2*nbPoints/nbSubproblems - 2)*3*3;
    int copyIndex = fakeBlockIdx*boundMaxEdgePerSubset*nbSubproblems;

    if (fakeBlockIdx != 0){
        while (edgePathList[idLeft].usage != INVALID){
            globalEdgeList[copyIndex] = edgePathList[idLeft];
            globalEdgeList[copyIndex].usage = UNUSED_LEFT;
            printf("PATH POINT %f %f %d -> %f %f %d \n", edgePathList[idLeft].x.x, edgePathList[idLeft].x.y, *(int*)& edgePathList[idLeft].x.z, edgePathList[idLeft].y.x, edgePathList[idLeft].y.y, *(int*)& edgePathList[idLeft].y.z);
            idLeft++;
            copyIndex++;
        }
    }
    if (fakeBlockIdx != (nbSubproblems - 1)){
        while (edgePathList[idRight].usage != INVALID){
            globalEdgeList[copyIndex] = edgePathList[idRight];
            globalEdgeList[copyIndex].usage = UNUSED_RIGHT;
            printf("PATH POINT %f %f %d -> %f %f %d \n", edgePathList[idRight].x.x, edgePathList[idRight].x.y, *(int*)& edgePathList[idRight].x.z, edgePathList[idRight].y.x, edgePathList[idRight].y.y, *(int*)& edgePathList[idRight].y.z);
            idRight++;
            copyIndex++;
        }
    }

    // printf("Block N°%d | THIS BLOCK HAS A NUMBER OF LEFT AND RIGHT NEIGHBOURS OF: %d and %d\n",fakeBlockIdx, nbLeftNeighbours, nbRightNeighbours);


    int triangleIndex = (fakeBlockIdx)*nbMaxTriangle;

    int initialStartEdgeIndex = fakeBlockIdx*boundMaxEdgePerSubset*nbSubproblems;
    int startEdgeIndex = initialStartEdgeIndex;
    int endEdgeIndex = copyIndex;

    // printf("Block N°%d | INDEX PRINT: Start Index : %d End Index : %d Delta Index : %d \n", fakeBlockIdx, startEdgeIndex, endEdgeIndex, copyIndex);

    edge currentEdge;
    float3 bestThirdPoint;
    int bestThirdPointSide;

    // printf("TEST TRIANGULATION BEGIN !\n");

    //Triangulation
    while (startEdgeIndex < endEdgeIndex){

        currentEdge = globalEdgeList[startEdgeIndex];

        printf("Block N°%d | Current edge used: %d -> %d \n",fakeBlockIdx, *(int*)& currentEdge.x.z, *(int*)& currentEdge.y.z);

        float bestRadius = INFINITY;
        bool triangleFound = false;
        for (int i = sliceBlockBeg; i<sliceBlockEnd+copyIndex-initialStartEdgeIndex+1; i++){
            float3 testPoint;
            if (i < sliceBlockEnd){
                testPoint = points[i];
            }
            else if (i == sliceBlockEnd+copyIndex-initialStartEdgeIndex){
                testPoint = globalEdgeList[copyIndex-1].y;
            }
            else{
                testPoint = globalEdgeList[i-sliceBlockEnd+initialStartEdgeIndex].x;
            }
            float3 firstVector = currentEdge.y - currentEdge.x;
            float3 secondVector = testPoint - currentEdge.x;
            float zVectorialProduct = firstVector.x*secondVector.y - firstVector.y*secondVector.x;
            int pointSide = 0;
            if (zVectorialProduct > 0){
                pointSide = 1;
            }
            else if(zVectorialProduct < 0){
                pointSide = -1;
            }

            if(testPoint.z != currentEdge.x.z && testPoint.z != currentEdge.y.z && pointSide != 0
                && pointSide != currentEdge.usage && currentEdge.usage != FULL
                && !(currentEdge.usage == UNUSED_LEFT && pointSide == 1)
                && !(currentEdge.usage == UNUSED_RIGHT && pointSide == -1)){
                    
                float radius = delaunayDistance(currentEdge.x, currentEdge.y, testPoint);
                if (radius < bestRadius){
                    int3 currentTriangle = make_int3(currentEdge.x.z, currentEdge.y.z, points[i].z);
                    bool alreadyExisting = false;
                    for (int j = fakeBlockIdx * nbMaxTriangle; j<triangleIndex; j++){
                        if (currentTriangle == triangleList[j]){
                            alreadyExisting = true;
                        }
                    }
                    if (!alreadyExisting){
                        bestRadius = radius;
                        bestThirdPoint = testPoint;
                        bestThirdPointSide = pointSide;
                        triangleFound = true;
                    }
                }
            }
        }

        if (triangleFound){
            
            printf("Block N°%d | TEST TRIANGLE FOUND : EDGE (%d %d (usage = %d)) with POINT (%d) !\n", fakeBlockIdx, *(int*)& currentEdge.x.z, *(int*)& currentEdge.y.z, *(int*)& currentEdge.usage, *(int*)& bestThirdPoint.z);

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
            for (int k = initialStartEdgeIndex; k<=endEdgeIndex; k++){ // TODO To optimize if possible
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

            // printf("Block N°%d | TEST TEST \n", fakeBlockIdx);

            //The case where the edge has two points on the same side is not possible here otherwise the bestThirdPoint wouldn't have been choosen

            //Add the two side edges only if they are new (the first one is always preexisting)
            if (secondEdgeNew){
                secondEdge.usage = USED; //New edges are created according to a direct repere 
                globalEdgeList[endEdgeIndex] = secondEdge;
                endEdgeIndex++;
                // printf("Block N°%d | NEW SECOND EDGE : %d -> %d \n", fakeBlockIdx, *(int*)& secondEdge.x.z, *(int*)& secondEdge.y.z);
            }
            else{
                float3 firstVectorSecondEdge = secondEdge.y-secondEdge.x;
                float3 secondVectorSecondEdge = currentEdge.x-currentEdge.y;
                float zVectorialProductSecondEdge = firstVectorSecondEdge.x*secondVectorSecondEdge.y - firstVectorSecondEdge.y*secondVectorSecondEdge.x;
                if (zVectorialProductSecondEdge == 1 || secondEdge.usage == FULL){ //If used in the same way as created or already used twice
                    validTriangle = false;
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
                // printf("Block N°%d | NEW THIRD EDGE : %d -> %d \n", fakeBlockIdx, *(int*)& thirdEdge.x.z, *(int*)& thirdEdge.y.z);
            }
            else{
                float3 firstVectorThirdEdge = thirdEdge.y-thirdEdge.x;
                float3 secondVectorThirdEdge = currentEdge.y-bestThirdPoint;
                float zVectorialProductThirdEdge = firstVectorThirdEdge.x*secondVectorThirdEdge.y - firstVectorThirdEdge.y*secondVectorThirdEdge.x;
                if (zVectorialProductThirdEdge == 1 || thirdEdge.usage == FULL){ //If they have the same sign
                    validTriangle = false;
                }
                else{
                    thirdEdge.usage = FULL;
                    globalEdgeList[indexThirdEdge].usage = thirdEdge.usage;
                }
            }

            if(validTriangle){
                currentEdge.usage = FULL;
                globalEdgeList[startEdgeIndex].usage = currentEdge.usage;

                triangleList[triangleIndex].x = *(int*)& currentEdge.x.z;
                triangleList[triangleIndex].y = *(int*)& currentEdge.y.z;
                triangleList[triangleIndex].z = *(int*)& bestThirdPoint.z;
                printf("Block N°%d | STORE TRIANGLE : %d %d %d\n", fakeBlockIdx, triangleList[triangleIndex].x, triangleList[triangleIndex].y, triangleList[triangleIndex].z);
                triangleIndex++;
                triangleList[triangleIndex].x = -1;

                if (triangleIndex > (fakeBlockIdx+1)*nbMaxTriangle){
                    printf("/!\\ STOP : MAXIMUM AMOUNT OF TRIANGLE PER SUBSET EXCEEDED /!\\ \n");
                }
            }
        }
        startEdgeIndex++;
    }
}