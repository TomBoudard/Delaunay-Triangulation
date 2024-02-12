#include "tools.cu"

float delaunayDistance(float3 const& edgeStart, float3 const& edgeEnd, float3 const& point){

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

int parDeTri(float3* points, edge* edgePathList, int3* triangleList, int nbPoints, int nbSubproblems){ //TODO Verify if possible to use reference //TODO Avoid if/else as much as possible
    
    edge currentEdge;
    float3 bestThirdPoint;
    int bestThirdPointSide;
    int startEdgeIndex = 0;
    int triangleIndex = 0;

    edge globalEdgeList[6*nbPoints - 6];

    // IDs of the beginning and end of the slice manipulated by the block
    int sliceBlockBeg = (blockIdx.x) * nbPoints / (nbSubproblems);
    int sliceBlockEnd = ((blockIdx.x + 1)) * nbPoints / (nbSubproblems);

    int idLeft = (log2(nbSubproblems) - __ffs(blockIdx.x) - 1)*nbPoints + blockIdx.x%(log2(nbSubproblems) - __ffs(blockIdx.x))*nbPoints/(log2(nbSubproblems) - __ffs(blockIdx.x))
    int idRight = (log2(nbSubproblems) - __ffs(blockIdx.x+1) - 1)*nbPoints + (blockIdx.x+1)%(log2(nbSubproblems) - __ffs(blockIdx.x+1))*nbPoints/(log2(nbSubproblems) - __ffs(blockIdx.x+1))

    int copyIndex = 0;

    if (blockIdx.x != 0){
        for(int indexLeft = idLeft; indexLeft < nbPoints/(log2(nbSubproblems) - __ffs(blockIdx.x)); indexLeft++){
            globalEdgeList[copyIndex] = edgePathList[indexLeft];
            copyIndex++;
        }
    }
    if (blockIdx.x != (nbSubproblems - 1)){
        for(int indexRight = idRight; indexRight < nbPoints/(log2(nbSubproblems) - __ffs(blockIdx.x+1)); indexRight++){
            globalEdgeList[copyIndex] = edgePathList[indexRight];
            copyIndex++;
        }
    }

    int endEdgeIndex = copyIndex;

    //Triangulation
    while (startEdgeIndex < endEdgeIndex){

        currentEdge = globalEdgeList[startEdgeIndex];

        float bestRadius = INFINITY;
        bool triangleFound = false;
        for (int i = sliceBlockBeg; i<sliceBlockEnd; i++){
            float3 firstVector = currentEdge.y-currentEdge.x;
            float3 secondVector = points[i]-currentEdge.x;
            float zVectorialProduct = firstVector.x*secondVector.y - firstVector.y*secondVector.x;
            int pointSide = zVectorialProduct/fabs(zVectorialProduct);
            if(points[i].z != currentEdge.x.z && points[i].z != currentEdge.y.z && pointSide != currentEdge.side && currentEdge.side != 2){ //If pointSide==currentEdge.z==0 it is skipped but we don't care about this case
                float radius = delaunayDistance(currentEdge.x, currentEdge.y, points[i]);
                if (radius < bestRadius){
                    int3 currentTriangle = make_int3(currentEdge.x.z, currentEdge.y.z, points[i].z);
                    bool alreadyExisting = false;
                    for (int j = 0; j<triangleIndex; j++){
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
                if (secondEdge.usage == UNUSED){ // Special case happening only for the edges from the path
                    secondEdge.usage = USED;
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
                if (thirdEdge.usage == UNUSED){ // Special case happening only for the edges from the path
                    thirdEdge.usage = USED;
                    globalEdgeList[indexThirdEdge].usage = thirdEdge.usage;
                }
                else{
                    thirdEdge.usage = FULL;
                    globalEdgeList[indexThirdEdge].usage = thirdEdge.usage;
                }
            }

            if(validTriangle){
                std::cout << "THE TRIANGLE IS VALID" << std::endl;
                triangleList[triangleIndex] = make_int3(currentEdge.x.z, currentEdge.y.z, bestThirdPoint.z); //TODO STORE ONLY DIRECT TIRANGLE
                triangleIndex++;
            }
        }
        std::cout << "======================" << std::endl;
        startEdgeIndex++;
    }
    return triangleIndex;
}