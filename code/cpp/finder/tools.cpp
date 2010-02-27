

#include "tools.h"
#include "cv.h"

using namespace cv;


float sum(const vector<float>& x) {
    float total = 0.0;
    for (unsigned int i=0; i<x.size(); i++)
        total = total + x[i]; 
    return total;
}

Rect sub_region(Rect region) {
    Rect r;
    r.x = region.x + region.width * 0.2;
    r.y = region.y + region.height * 0.15;
    r.width = region.width * 0.6;
    r.height = region.height * 0.7;
    return r;
}