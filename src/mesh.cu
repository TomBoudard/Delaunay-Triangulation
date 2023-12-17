// Defining mesh struct

#ifndef PT_STRUCT
#define PT_STRUCT

struct vertex { //TODO Improve with float3
    unsigned int index;
    float x;
    float y;

};

vertex operator+(vertex a, vertex b){
    vertex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
};

vertex operator-(vertex a, vertex b){
    vertex c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
};

vertex operator*(vertex a, vertex b){
    vertex c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    return c;
};

bool operator==(vertex a, vertex b){
    return (a.x == b.x && a.y == b.y);
};

bool operator!=(vertex a, vertex b){
    return !(a.x == b.x && a.y == b.y);
};

float length(vertex a){
    return a.x*a.x + a.y*a.y;
};

struct edge{
    vertex a;
    vertex b;
};

struct triangle{
    vertex a;
    vertex b;
    vertex c;
};

bool operator==(triangle A, triangle B){
    return (A.a == A.a && A.b == B.b && A.c == B.c);
}


float delaunayDistance(edge localEdge, vertex point){

    float area4 = 2.0*(localEdge.a.x*(localEdge.b.y-point.y) + localEdge.b.x*(point.y-localEdge.a.y) + point.x*(localEdge.a.y-localEdge.b.y));
    float oppEdgeA = length(point-localEdge.b); 
    float oppEdgeB = length(localEdge.a-point); 
    float oppEdgeC = length(localEdge.b-localEdge.a);
    oppEdgeA *= oppEdgeA;
    oppEdgeB *= oppEdgeB;
    oppEdgeC *= oppEdgeC;
    float barCircumCenterA = oppEdgeA*(oppEdgeB+oppEdgeC-oppEdgeA);
    float barCircumCenterB = oppEdgeB*(oppEdgeC+oppEdgeA-oppEdgeB);
    float barCircumCenterC = oppEdgeA*(oppEdgeA+oppEdgeB-oppEdgeC);

    vertex circumCenter;
    float sumBarycenters = barCircumCenterA + barCircumCenterB + barCircumCenterC;
    circumCenter.x = (barCircumCenterA*localEdge.a.x + barCircumCenterB*localEdge.b.x + barCircumCenterC*point.x)/sumBarycenters;
    circumCenter.y = (barCircumCenterA*localEdge.a.y + barCircumCenterB*localEdge.b.y + barCircumCenterC*point.y)/sumBarycenters;

    float radius = sqrt(length(circumCenter-point));

    if (barCircumCenterC < 0){ //Check whether the circumCenter is in the half space of the point or not
        return -radius;
    }
    else{
        return radius;
    }
}

#endif