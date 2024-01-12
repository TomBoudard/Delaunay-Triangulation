// Defining mesh struct

#ifndef PT_STRUCT
#define PT_STRUCT

// bool operator==(int2 const& a, int2 const& b){
//     if (a.x == b.x && a.y == b.y){
//         return true;
//     }
//     else if (a.x == b.y && a.y == b.x){
//         return true;
//     }
//     return false;
// }

// bool operator!=(int2 const& a, int2 const& b){
//     return !(a==b);
// }

bool operator==(int3 const& a, int3 const& b){
    if (a.x == b.x && a.y == b.y && a.z == b.z){
        return true;
    }
    else if (a.x == b.x && a.y == b.z && a.z == b.y){
        return true;
    }
    else if (a.x == b.y && a.y == b.x && a.z == b.z){
        return true;
    }
    else if (a.x == b.y && a.y == b.z && a.z == b.x){
        return true;
    }
    else if (a.x == b.z && a.y == b.y && a.z == b.x){
        return true;
    }
    else if (a.x == b.z && a.y == b.x && a.z == b.y){
        return true;
    }
    return false;
}

bool operator!=(int3 const& a, int3 const& b){ //Used only for edges on the two first int
    if (a.x == b.x && a.y == b.y){
        return false;
    }
    else if (a.x == b.y && a.y == b.x){
        return false;
    }
    return true;
}

float3 operator-(float3 const& a, float3 const& b){
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

float length(float3 a){
    return a.x*a.x + a.y*a.y;
};


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

    std::cout << "Circumcenter : " << circumCenter.x << " and " << circumCenter.y << " for edge : " << edgeStart.z << " and " << edgeEnd.z << std::endl;

    float radius = sqrt(length(circumCenter-point));

    if (barCircumCenterC < 0){ //Check whether the circumCenter is in the half space of the point or not
        return -radius;
    }
    else{
        return radius;
    }
}

#endif