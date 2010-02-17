//#define CV_NO_BACKWARD_COMPATIBILITY

#define SKIN "../../../data/hand/skin.png"
#define HEAD "../../../data/hand/head.png"
#define FACEHAAR "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"
#define HANDA "../../../data/hand/a.png"
#define HANDB "../../../data/hand/b.png"
#define HANDC "../../../data/hand/c.png"
#define HANDD "../../../data/hand/d.png"

#include "cv.h"
#include "cvtypes.h"
#include "highgui.h"
#include "cvaux.h"

#include <iostream>


using namespace cv;
using namespace std;

Rect face_region(Rect face) {
    Rect r;
    r.x = face.x + face.width * 0.2;
    r.y = face.y + face.height * 0.15;
    r.width = face.width * 0.6;
    r.height = face.height * 0.7;
    return r;
}


    //VideoCapture cap(0);
    //if(!cap.isOpened()) {
        //cout << "couldn't open video\n";
        //return -1;
    //}


int main(int, char**) {

    //setNumThreads(5);

    CascadeClassifier haarzoeker;
	Mat hsv_skin, hsv_head, result, bw_head, hsv_handa, bp;
    Mat skin = imread(SKIN, 1);
	if (!skin.data) {
	 cerr << "can't load skin" << endl;
		return -1;
	}
    Mat head = imread(HEAD, 1);
	if (!head.data) {
		cerr << "can't load head" << endl;
		return -1;
	}
	Mat handa = imread(HANDA, 1);
	if (!handa.data) {
		cerr << "can't load handa" << endl;
		return -1;
	}

    if ( !haarzoeker.load(FACEHAAR) ) {
        cerr << "haar werkt niet" << endl;
    };
	
	
    cvtColor(skin, hsv_skin, CV_BGR2HSV);
    cvtColor(head, hsv_head, CV_BGR2HSV);
    cvtColor(handa, hsv_handa, CV_BGR2HSV);
    cvtColor(head, bw_head, CV_BGR2GRAY);

    vector<Rect> faces;
    haarzoeker.detectMultiScale(bw_head, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    Rect face = faces[0];
    Rect region = face_region(face);
    Mat sub_face = hsv_head(region);
    //rectangle(head, region, CV_RGB(0,0,255), 1, 1);
    MatND hist;

    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    const float hranges[] = { 0, 180 };
    const float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    int channels[] = {0, 1};
    calcHist( &sub_face,  1, channels, Mat(), hist, 2,  histSize, ranges );

    calcBackProject( &hsv_handa, 1, channels, hist, bp, ranges );
    GaussianBlur( bp, bp, Size(51, 51), 0);
    threshold(bp, result, 80, 1, CV_THRESH_BINARY);


    vector<vector<Point> > contours;
    findContours( result, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
    drawContours( handa, contours, -1, Scalar( 0, 0, 255 ));

    Rect box = boundingRect(contours.at(0));
    Mat inv;
    inv.ones(handa.size(), 0);
    inv = inv - handa;
    rectangle(handa, Point(box.x, box.y), Point(box.x+box.width, box.y+box.height), Scalar(0, 255, 0) );
    //HOGDescriptor h = HOGDescriptor();

    imshow( "sub_face", handa);
    //imshow( "head", head);
    imshow( "backproject", result );


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

