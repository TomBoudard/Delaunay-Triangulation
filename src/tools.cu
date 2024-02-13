// Defining mesh struct

#ifndef PT_STRUCT
#define PT_STRUCT

#define INVALID -1
#define UNUSED 0
#define USED 1
#define FULL 2

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

struct edge{
    float3 x;
    float3 y;
    int8_t usage; // -1 : Invalid | 0: Unused | 1: Used once | 2: Used twice (can't be used anymore)
};

__device__ bool operator==(edge const& a, edge const& b){
    if (a.x.z == b.x.z && a.y.z == b.y.z){
        return true;
    }
    else if(a.x.z == b.y.z && a.y.z == b.x.z){
        return true;
    }
    return false;
}

__device__ bool operator==(int3 const& a, int3 const& b){
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

// bool operator!=(int3 const& a, int3 const& b){ //Used only for edges on the two first int
//     if (a.x == b.x && a.y == b.y){
//         return false;
//     }
//     else if (a.x == b.y && a.y == b.x){
//         return false;
//     }
//     return true;
// }

__device__ float3 operator-(float3 const& a, float3 const& b){
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ float length(float3 a){
    return a.x*a.x + a.y*a.y;
};


#endif