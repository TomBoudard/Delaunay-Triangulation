#include <vector>
#include "mesh.cu"

triangle makeTriangle(edge edge, std::vector<vertex> points, std::vector<triangle> triangleList){
    vertex bestThirdPoint;
    float bestRadius = INFINITY;
    for (vertex point : points){
        if(point != edge.a && point != edge.b){   
            float radius = delaunayDistance(edge, point);
            if (radius < bestRadius){
                triangle currentTriangle = {edge.a, edge.b, point}
                bool alreadyExisting = false;
                for (triangle previousTriangle : triangleList){
                    if (currentTriangle == previousTriangle){
                        alreadyExisting = true;
                    }
                }
                if (!alreadyExisting){
                    bestRadius = radius;
                    bestThirdPoint = point;
                }
            }
        }
    }
    triangle delaunayTriangle;
    if (bestThirdPoint){
        delaunayTriangle = {edge.a, edge.b, bestThirdPoint};
    }
    else{
        delaunayTriangle = {edge.a, edge.a, edge.a}; //No valid triangle
    }
    return delaunayTriangle;
}

__shared__ void parDeTri(std::vector<vertex> points, std::vector<edge> edgeList, std::vector<triangle> triangleList){ //TODO Verify if possible to use reference //TODO triangle or face ?

    edge currentEdge;
    triangle newTriangle;
    std::vector<edge> usedEdgeList;

    while (!edgeList.empty()){
        currentEdge = edgeList.back();
        newTriangle = makeTriangle(currentEdge, points, triangleList);

        if (newTriangle.a != newTriangle.b){ //If the two first vertices of the triangle is not valid
            triangleList.push_back(newTriangle);
            edgeList.pop_back(); //Remove the last edge which is the currentEdge
            edge newEdgeA = {newTriangle.a, newTriangle.c};
            edge newEdgeB = {newTriangle.b, newTriangle.c};
            edgeList.push_back(newEdgeA);
            edgeList.push_back(newEdgeB);
        }
    }

}