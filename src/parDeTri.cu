#include "mesh.cu"

// int3 makeTriangle(int2 edge, float3[] points, int3[] triangleList, int triangleIndex){
//     float bestRadius = INFINITY;
//     int3 delaunayTriangle;
//     for (int i = 0; i<sizeof(points)/sizeof(float3); i++){
//         if(points[i].z != edge.x && points[i].z != edge.y){   
//             float radius = delaunayDistance(edge, point); //FIXME Use the good params
//             if (radius < bestRadius){
//                 int3 currentTriangle = {edge.a, edge.b, point.z};
//                 bool alreadyExisting = false;
//                 for (int j = 0; j<triangleIndex; j++){
//                     if (currentTriangle == triangleList[j]){
//                         alreadyExisting = true;
//                     }
//                 }
//                 if (!alreadyExisting){
//                     bestRadius = radius;
//                     delaunayTriangle = currentTriangle;
//                 }
//             }
//         }
//     }
//     return delaunayTriangle;
// }

// __global__ void parDeTri(float3[] points, int2[] edgeList, int3[] triangleList){ //TODO Verify if possible to use reference //TODO triangle or face ?
//     //FIXME Add case if the triangle has two edge from the path
//     int2 currentEdge;
//     int3 newTriangle;
//     int startEdgeIndex;
//     int endEdgeIndex;
//     int triangleIndex;

//     while (startIndex != endIndex){
//         currentEdge = edgeList[startEdgeIndex];
//         newTriangle = makeTriangle(currentEdge, points, triangleSet, triangleIndex);

//         if (newTriangle != (-1, -1, -1)){ //If the three vertices of the new triangle are valid
//             triangleList[triangleIndex] = newTriangle;
//             triangleIndex++;
//             int2 firstEdge(newTriangle.x, newTriangle.y);
//             int2 secondEdge(newTriangle.y, newTriangle.z);
//             int2 thirdEdge(newTriangle.z, newTriangle.x);
            
//             if (firstEdge != currentEdge){
//                 edgeList[endEdgeIndex] = firstEdge;
//                 endEdgeIndex++;
//             }
//             if (secondEdge != currentEdge){
//                 edgeList[endEdgeIndex] = secondEdge;
//                 endEdgeIndex++;
//             }
//             if (thirdEdge != currentEdge){
//                 edgeList[endEdgeIndex] = thirdEdge;
//                 endEdgeIndex++;
//             }
//             startEdgeIndex++;
//         }
//     }
// }

int parDeTri(float3* points, int3* edgeList, int3* triangleList, int nbPoints, int nbEdges){ //TODO Verify if possible to use reference //TODO Avoid if/else as much as possible
    // for (int i = 0; i < 10; i++){
    //     std::cout << "X: " << points[i].x << " Y: " << points[i].y << " Z: " << points[i].z << std::endl;
    // }
    
    int3 currentEdge;
    float3 edgeStartPoint = make_float3(0.0, 0.0, -1.0);
    float3 edgeEndPoint = make_float3(0.0, 0.0, -1.0);
    float3 bestThirdPoint;
    int startEdgeIndex = 0;
    int endEdgeIndex = nbEdges;
    int triangleIndex = 0;

    //TODO Work on the subset of points, edges and triangles concerned by the processor

    while (startEdgeIndex < endEdgeIndex){
        std::cout << "Start edge index: " << startEdgeIndex << std::endl;
        currentEdge = edgeList[startEdgeIndex];
        int pointIndex = 0;
        while (edgeStartPoint.z != currentEdge.x || edgeEndPoint.z != currentEdge.y){ //We save the points of the current edge used
            if (currentEdge.x == points[pointIndex].z){
                edgeStartPoint = points[pointIndex];
            }
            else if (currentEdge.y == points[pointIndex].z){
                edgeEndPoint = points[pointIndex];
            }
            pointIndex++;
        }
        float bestRadius = INFINITY;
        bool triangleFound = false;
        for (int i = 0; i<nbPoints; i++){
            float3 firstVector = edgeEndPoint-edgeStartPoint;
            float3 secondVector = points[i]-edgeStartPoint;
            float scalarProduct = firstVector.x*secondVector.x + firstVector.y*secondVector.y;
            int pointSide = 0;
            if (scalarProduct < 0){
                pointSide = -1;
            }
            else if (scalarProduct > 0){
                pointSide = 1;
            }
            if(points[i].z != currentEdge.x && points[i].z != currentEdge.y && pointSide != currentEdge.z){ //If pointSide==currentEdge.z==0 it is skipped but we don't care about this case
                float radius = delaunayDistance(edgeStartPoint, edgeEndPoint, points[i]);
                std::cout << "Radius: " << radius << " for point : " << points[i].z << std::endl;
                if (radius < bestRadius){
                    int3 currentTriangle = make_int3(currentEdge.x, currentEdge.y, points[i].z);
                    bool alreadyExisting = false;
                    for (int j = 0; j<triangleIndex; j++){
                        if (currentTriangle == triangleList[j]){
                            alreadyExisting = true;
                        }
                    }
                    if (!alreadyExisting){
                        bestRadius = radius;
                        bestThirdPoint = points[i];
                        triangleFound = true;
                    }
                }
            }
        }
        std::cout << "Best radius: " << bestRadius << std::endl;
        std::cout << "======================" << std::endl;
        if (triangleFound){
            bool validTriangle = true;
            int3 secondEdge = make_int3(currentEdge.y, bestThirdPoint.z, 0);
            int3 thirdEdge = make_int3(bestThirdPoint.z, currentEdge.x, 0);
            bool secondEdgeNew = true;
            bool thirdEdgeNew = true;
            for (int k = 0; k<=endEdgeIndex; k++){
                if (!(secondEdge != edgeList[k])){
                    secondEdgeNew = false;
                    secondEdge = edgeList[k];
                }
                if (!(thirdEdge != edgeList[k])){
                    thirdEdgeNew = false;
                    thirdEdge = edgeList[k];
                }
            }

            float3 firstVectorSecondEdge = bestThirdPoint-edgeEndPoint;
            float3 secondVectorSecondEdge = edgeStartPoint-edgeEndPoint;
            float scalarProductSecondEdge = firstVectorSecondEdge.x*secondVectorSecondEdge.x + firstVectorSecondEdge.y*secondVectorSecondEdge.y;

            float3 firstVectorThirdEdge = edgeStartPoint-bestThirdPoint;
            float3 secondVectorThirdEdge = edgeEndPoint-bestThirdPoint;
            float scalarProductThirdEdge = firstVectorThirdEdge.x*secondVectorThirdEdge.x + firstVectorThirdEdge.y*secondVectorThirdEdge.y;

            //The case where the edge has two points on the same side is not possible here otherwise the bestThirdPoint wouldn't have been choosen

            //Add the two side edges only if they are new (the first one is always preexisting)
            if (secondEdgeNew){
                if (scalarProductSecondEdge < 0){
                    secondEdge.z = -1;
                }
                else{
                    secondEdge.z = 1;
                }
                edgeList[endEdgeIndex] = secondEdge;
                endEdgeIndex++;
            }
            else if (scalarProductSecondEdge*secondEdge.z > 0){ //If they have the same sign
                validTriangle = false;
            }

            if (thirdEdgeNew){
                if (scalarProductThirdEdge < 0){
                    thirdEdge.z = -1;
                }
                else{
                    thirdEdge.z = 1;
                }
                edgeList[endEdgeIndex] = thirdEdge;
                endEdgeIndex++;
            }
            else if (scalarProductSecondEdge*thirdEdge.z > 0){ //If they have the same sign
                validTriangle = validTriangle && false;
            }

            triangleList[triangleIndex] = make_int3(currentEdge.x, currentEdge.y, bestThirdPoint.z);
            triangleIndex++;
        }
        startEdgeIndex++;
    }
    return triangleIndex;
}