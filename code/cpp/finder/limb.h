
#ifndef _LIMB_H
#define	_LIMB_H

#include "cv.h"

using namespace cv;

struct Limb {
public:
    vector<Point> contour;
    float radius;
    Point2f center;
    Limb();
    Limb(vector<Point> c);
};

bool compare_limbs(const Limb& a, const Limb& b);

#endif