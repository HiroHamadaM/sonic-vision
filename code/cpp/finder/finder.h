
#ifndef _FINDER_H
#define	_FINDER_H

#include "cv.h"
#include "highgui.h"
#include "limb.h"

using namespace cv;


struct Finder {
public:
    Mat frame, small, hsv, bw, backproj, mask;
    Mat facepixels, visuals, combi, temp, blurred, morphed, th, limb_zoom;
    MatND histogram, new_hist, old_hist;
    VideoCapture cap;
    CascadeClassifier haar;
    Size frame_size;
    Size small_size;
    float scale;
    vector<Rect> faces;
    Rect face;
    vector<vector<Point> > contours;
	vector<Point> face_contour;
    Limb left_hand, right_hand, head;
    
    Finder(VideoCapture c);
    void grab_frame();
    void find_face();
    void make_histogram();
    void make_backproject();
    void make_mask();
    void visualize();
    void find_contours();
    void find_limbs();
    void match_hands();
    void mainloop();
};


#endif	/* _FINDER_H */

