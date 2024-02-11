#include "mesh.cu"

int parDeTri(float3* points, edge* edgeList, int3* triangleList, int nbPoints, int nbEdges){ //TODO Verify if possible to use reference //TODO Avoid if/else as much as possible
    // for (int i = 0; i < 10; i++){
    //     std::cout << "X: " << points[i].x << " Y: " << points[i].y << " Z: " << points[i].z << std::endl;
    // }
    
    edge currentEdge;
    float3 edgeStartPoint = make_float3(0.0, 0.0, -1.0);
    float3 edgeEndPoint = make_float3(0.0, 0.0, -1.0);
    float3 bestThirdPoint;
    int bestThirdPointSide;
    int startEdgeIndex = 0;
    int endEdgeIndex = nbEdges;
    int triangleIndex = 0;

    //TODO Work on the subset of points, edges and triangles concerned by the processor

    while (startEdgeIndex < endEdgeIndex){
        std::cout << "Start edge index: " << startEdgeIndex << std::endl;
        currentEdge = edgeList[startEdgeIndex];
        std::cout << "Current edge: " << currentEdge.x.z << " -> " << currentEdge.y.z << std::endl;
        std::cout << "Current edge side: " << currentEdge.side << std::endl;

        // int pointIndex = 0;
        // while (edgeStartPoint.z != currentEdge.x || edgeEndPoint.z != currentEdge.y){ //We save the points of the current edge used
        //     if (currentEdge.x == points[pointIndex].z){ //TODO To optimize
        //         edgeStartPoint = points[pointIndex];
        //     }
        //     else if (currentEdge.y == points[pointIndex].z){
        //         edgeEndPoint = points[pointIndex];
        //     }
        //     pointIndex++;
        // }
        float bestRadius = INFINITY;
        bool triangleFound = false;
        for (int i = 0; i<nbPoints; i++){
            float3 firstVector = currentEdge.y-currentEdge.x;
            float3 secondVector = points[i]-currentEdge.x;
            float zVectorialProduct = firstVector.x*secondVector.y - firstVector.y*secondVector.x;
            int pointSide = zVectorialProduct/fabs(zVectorialProduct);
            // std::cout << "Edge side (1): " << currentEdge.z << std::endl;
            if(points[i].z != currentEdge.x.z && points[i].z != currentEdge.y.z && pointSide != currentEdge.side && currentEdge.side != 2){ //If pointSide==currentEdge.z==0 it is skipped but we don't care about this case
                float radius = delaunayDistance(currentEdge.x, currentEdge.y, points[i]);
                // std::cout << "Radius: " << radius << " for point : " << points[i].z << std::endl;
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
        // std::cout << "Best radius: " << bestRadius << std::endl;
        std::cout << "Best third point: " << bestThirdPoint.z << std::endl;
        std::cout << "------------------------" << std::endl;
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
            for (int k = 0; k<=endEdgeIndex; k++){
                if (secondEdge == edgeList[k]){
                    secondEdgeNew = false;
                    secondEdge = edgeList[k];
                    indexSecondEdge = k;
                    // std::cout << "EDGE 2 OLD" << edgeList[k].z << std::endl;
                }
                if (thirdEdge == edgeList[k]){
                    thirdEdgeNew = false;
                    thirdEdge = edgeList[k];
                    indexThirdEdge = k;
                    // std::cout << "EDGE 3 OLD" << edgeList[k].z << std::endl;
                }
            }



            //The case where the edge has two points on the same side is not possible here otherwise the bestThirdPoint wouldn't have been choosen

            //Add the two side edges only if they are new (the first one is always preexisting)
            // std::cout << "SECOND EDGE SIDE USED: " << secondEdge.z << std::endl;
            if (secondEdgeNew){
                secondEdge.side = 1; //New edges are created according to a direct repere 
                edgeList[endEdgeIndex] = secondEdge;
                endEdgeIndex++;
            }
            else{
                float3 firstVectorSecondEdge = secondEdge.y-secondEdge.x;
                float3 secondVectorSecondEdge = currentEdge.x-currentEdge.y;
                float zVectorialProductSecondEdge = firstVectorSecondEdge.x*secondVectorSecondEdge.y - firstVectorSecondEdge.y*secondVectorSecondEdge.x;
                if (zVectorialProductSecondEdge == 1 || secondEdge.side == 2){ //If used in the same way as created or already used twice
                    validTriangle = false;
                }
                else{
                    secondEdge.side = 2;
                    edgeList[indexSecondEdge].side = secondEdge.side;
                }
            }

            // std::cout << "THIRD EDGE SIDE USED: " << thirdEdge.z << std::endl;
            if (thirdEdgeNew){
                thirdEdge.side = 1; //New edges are created according to a direct repere 
                edgeList[endEdgeIndex] = thirdEdge;
                endEdgeIndex++;
            }
            else{
                float3 firstVectorThirdEdge = thirdEdge.y-thirdEdge.x;
                float3 secondVectorThirdEdge = currentEdge.y-bestThirdPoint;
                float zVectorialProductThirdEdge = firstVectorThirdEdge.x*secondVectorThirdEdge.y - firstVectorThirdEdge.y*secondVectorThirdEdge.x;
                if (zVectorialProductThirdEdge == 1 || thirdEdge.side == 2){ //If they have the same sign
                    validTriangle = false;
                }
                else{
                    thirdEdge.side = 2;
                    edgeList[indexThirdEdge].side = thirdEdge.side;
                }
            }

            // std::cout << "Edge side (2): " << secondEdge.z << std::endl;
            // std::cout << "Edge side (3): " << thirdEdge.z << std::endl;

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