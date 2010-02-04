#include "cv.h"
#include "cv.hpp"
#include "cvaux.hpp"
#include "highgui.h"
#include <iostream>

using namespace cv;

int main(int, char**)
{
    VideoCapture cap(0);
    if(!cap.isOpened()) {
        std::cout << "couldn't open video\n";
        return -1;
    }

    HOGDescriptor h = HOGDescriptor();

    Mat edges;
    namedWindow("edges",1);
    for(;;)
    {
        Mat frame;
        cap >> frame;
        cvtColor(frame, edges, CV_BGR2GRAY);
        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        Canny(edges, edges, 0, 30, 3);
        imshow("edges", edges);
        if(waitKey(30) >= 0) break;
    }
    return 0;
}

