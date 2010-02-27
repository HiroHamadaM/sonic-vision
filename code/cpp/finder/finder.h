
#ifndef _FINDER_H
#define	_FINDER_H

struct Finder {
public:
    Mat frame, small, hsv, bw, backproj, mask, facepixels, visuals, combi, temp;
    MatND histogram;
    VideoCapture cap;
    CascadeClassifier haar;
    Size frame_size;
    Size small_size;
    float scale;
    vector<Rect> faces;
    Rect face;
    Finder(VideoCapture c);
    void grab_frame();
    void find_face();
    void make_histogram();
    void make_backproject();
    void make_mask();
    void visualize();
    void mainloop();
};


#endif	/* _FINDER_H */

