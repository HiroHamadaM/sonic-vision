//#define CV_NO_BACKWARD_COMPATIBILITY


#define SKIN "../../../data/hand/skin.png"
#define HEAD "../../../data/hand/head.png"
#define HANDA "../../../data/hand/a.png"
#define HANDB "../../../data/hand/b.png"
#define HANDC "../../../data/hand/c.png"
#define HANDD "../../../data/hand/d.png"
#define FACEHAAR "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"

#include "cv.h"
#include "cvtypes.h"
#include "highgui.h"
#include "cvaux.h"
//#include "hog/hog.h"
//#include "rewrite.h"

#include <iostream>
#include <vector>
#include <iterator>


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


float sum(const vector<float>& x) {
    float total = 0.0;  // the sum is accumulated here
    for (unsigned int i=0; i<x.size(); i++) {
        total = total + x[i]; 
		cout << "x: " << x[i] << endl;
		cout << "xsize: " << x.size() << endl;
		cout << "i: " << i << endl;
    }
    return total;
}

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
    rectangle(handa, Point(box.x, box.y), Point(box.x+box.width, box.y+box.height), Scalar(0, 255, 0) );

    Mat clean;
    handa.copyTo(clean, result);
    Mat sub = clean(box);

    Mat sized;
    CvSize window = Size(64,128);
    resize(sub, sized, window);
	
	Mat sized_bw;
	cvtColor(sized, sized_bw, CV_BGR2GRAY);
	equalizeHist(sized_bw, sized_bw);
	
    HOGDescriptor h = HOGDescriptor();
    vector<float> descriptors;
    vector<Point> locations;
	Size winStride = Size();
	Size padding = Size(10, 10);
    h.compute(sized_bw, descriptors, winStride, padding, locations);
	
	//cout << "sum: "	<< sum( descriptors );

	//imshow("clean", sized);
    //imshow( "sub_face", handa);
    //imshow( "head", head);
    //imshow( "backproject", result );
    imshow( "sized", sized_bw );

	waitKey();
	
	return 0;
	//cout << descriptors.

    //IplImage** integrals;
    //IplImage img = IplImage(sized);
	//IplImage img = sized;


    //integrals = calculateIntegralHOG(&img);

	//int normalization = 4;
    //CvMat* img_feature_vector;
    //img_feature_vector = calculateHOG_window(integrals, cvRect(0, 0, window.width, window.height), normalization);


    //cvShowImage("henk", &img);
    //waitKey();


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
	 */


}

