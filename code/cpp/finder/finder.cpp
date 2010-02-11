#define CV_NO_BACKWARD_COMPATIBILITY

#define SKIN "/home/gijs/Work/sonic-vision/data/hand/skin.png"
#define HEAD "/home/gijs/Work/sonic-vision/data/hand/head.png"
#define FACEHAAR "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"

#include "cv.h"
#include "highgui.h"
#include "cvaux.h"
//#include "cv.hpp"
//#include "cvaux.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int, char**)
{

    setNumThreads(0);

    CascadeClassifier haarzoeker;
    Mat skin = imread(SKIN, 1);
    Mat head = imread(HEAD, 1);
    Mat hsv_skin, hsv_head, result, bw_head;
    cvtColor(skin, hsv_skin, CV_BGR2HSV);
    cvtColor(head, hsv_head, CV_BGR2HSV);
    cvtColor(head, bw_head, CV_BGR2GRAY);

    vector<Rect> faces;
    if ( !haarzoeker.load(FACEHAAR) ) {
        cerr << "haar werkt niet" << endl;
    };

    //VideoCapture cap(0);
    //if(!cap.isOpened()) {
        //cout << "couldn't open video\n";
        //return -1;
    //}

    HOGDescriptor h = HOGDescriptor();

    haarzoeker.detectMultiScale(bw_head, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    const float hranges[] = { 0, 180 };
    const float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    MatND hist;
    int channels[] = {0, 1};

    calcHist( &hsv_skin,  1, channels, Mat(), hist, 2,  histSize, ranges );
    calcBackProject( &hsv_head, 1, channels, hist, result, ranges );


    namedWindow( "Source", 1 );
    imshow( "Source", result );


    waitKey();

    /* Mat edges;
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
    return 0; */
}

