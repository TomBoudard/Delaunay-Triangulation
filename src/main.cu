#include <iostream>
#include <fstream>
#include <string>
#include <vector>

struct point2D {
    int index;
    float x;
    float y;
};

std::vector<point2D> readFile(std::string nameFile){
    std::vector<point2D> pointsVector;

    std::ifstream inputFile;
    inputFile.open(nameFile);
    
    int i=0;
    float x,y;

    while(inputFile >> x >> y) {
        pointsVector.push_back({i++, x, y});
    }

    return pointsVector;
    
}

int main(int argc, char *argv[]) {

    // TODO arguments reading and errors (filename, splitting method, ...)

    std::vector<point2D> pointsVector = readFile(argv[1]);

    return 0;
}