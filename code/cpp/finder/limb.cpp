

#include "limb.h"


Limb::Limb() {
    center = Point();
    radius = 0;
};

Limb::Limb(vector<Point> c) {
    contour = c;
    minEnclosingCircle(contour, center, radius);
};

bool compare_limbs(const Limb& a, const Limb& b) {
    return a.radius > b.radius;
}
