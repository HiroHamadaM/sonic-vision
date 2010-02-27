
#include <iostream>

#include "cv.h"
#include "highgui.h"

#include "finder.h"
#include "hand.h"
#include "skin.h"
#include "settings.h"

using namespace cv;
using namespace std;

int main(int, char**) {
    setNumThreads(5);
    Skin skin(HEAD, FACEHAAR);
	vector<Hand> hands;
    hands.push_back(Hand(HANDA, skin.histogram));
	hands.push_back(Hand(HANDB, skin.histogram));
	hands.push_back(Hand(HANDC, skin.histogram));
	hands.push_back(Hand(HANDD, skin.histogram));
    VideoCapture cap(DEVICE);
    Finder finder(cap);
    finder.mainloop();
    return 0;
}