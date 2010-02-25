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
#include "finder.h"
#include <iostream>
#include <exception>

using namespace cv;
using namespace std;







float sum(const vector<float>& x) {
    float total = 0.0;
    for (unsigned int i=0; i<x.size(); i++)
        total = total + x[i]; 
    return total;
}


struct Hand {
public:
    Mat img, hsv, backproj, cutout;
	HOGDescriptor hog;
	vector<float> descriptors;
    Hand(const string& filename, MatND histogram);
    void load_image(const string& filename);
    void make_hsv();
    void make_backproject(MatND histogram);
    void make_cutout();
	void find_hog();
};

Hand::Hand(const string& filename, MatND histogram) {
    load_image(filename);
    make_backproject(histogram);
    make_cutout();
	find_hog();
}

void Hand::load_image(const string& filename) {
    img = imread(filename, 1);
    if (!img.data) {
        throw exception();
	}
    cvtColor(img, hsv, CV_BGR2HSV);
};

void Hand::make_backproject(MatND histogram) {
    const float hranges[] = { 0, 180 };
    const float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    int channels[] = {0, 1};
    calcBackProject( &hsv, 1, channels, histogram, backproj, ranges );
}

void Hand::make_cutout() {
    Mat mask, clean, sized, sub, bw;
    GaussianBlur( backproj, mask, Size(51, 51), 0);
    threshold(mask, mask, 80, 1, CV_THRESH_BINARY);

    vector<vector<Point> > contours;
    findContours( mask, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
    //drawContours( working, contours, -1, Scalar( 0, 0, 255 ));

    Rect box = boundingRect(contours.at(0));
    //rectangle(mask, box.tl(), box.br(), Scalar(0, 255, 0) );

    img.copyTo(clean, mask);
    sub = clean(box);
    resize(sub, sized, Size(64,128));
    cvtColor(sized, bw, CV_BGR2GRAY);
    equalizeHist(bw, bw);
    bw.copyTo(cutout);
}

void Hand::find_hog() {
	hog = HOGDescriptor();
	vector<Point> locations;
	Size winStride = Size(8, 8);
	Size padding = Size(0, 0);
	hog.compute(cutout, descriptors, winStride, padding, locations);
};

struct Skin {
public:
    Mat img, hsv, bw, facepixels;
    MatND histogram;
	vector<Rect> faces;
    CascadeClassifier haarzoeker;    
    Skin(const string& facefile, const string& haarfile);
    void load_image(const string& filename);
    void load_haar(const string& filename);
    void find_face();
    void make_histogram();
    Rect face_region(Rect face);
};

Skin::Skin(const string& facefile, const string& haarfile) {
    load_image(facefile);
	load_haar(haarfile);
    find_face();
    make_histogram();
}

void Skin::load_image(const string& filename) {
    img = imread(filename, 1);
    if (!img.data) {
		cout << "can't load" << filename << endl;
        throw exception();
	}
    cvtColor(img, hsv, CV_BGR2HSV);
    cvtColor(img, bw, CV_BGR2GRAY);
};

void Skin::load_haar(const string& filename) {
	if ( !haarzoeker.load(FACEHAAR) ) {
		cerr << "can't load" << filename << endl;
		throw exception();
    }
}

Rect Skin::face_region(Rect face) {
    Rect r;
    r.x = face.x + face.width * 0.2;
    r.y = face.y + face.height * 0.15;
    r.width = face.width * 0.6;
    r.height = face.height * 0.7;
    return r;
}

void Skin::find_face() {
    haarzoeker.detectMultiScale(img, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	if (faces.size() == 0) {
		cerr << "no faces found in image" << endl;
		throw exception();
	}
	Rect face = faces.at(0);
    Rect region = face_region(face);
    facepixels = hsv(region);
}

void Skin::make_histogram() {
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    const float hranges[] = { 0, 180 };
    const float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    int channels[] = {0, 1};
    calcHist( &facepixels,  1, channels, Mat(), histogram, 2,  histSize, ranges );
}

int main(int, char**) {

    //setNumThreads(5);

    Skin skin(HEAD, FACEHAAR);
	Hand handa(HANDA, skin.histogram);
	Hand handb(HANDB, skin.histogram);
	cout << handa.descriptors.size() << endl;
	cout << handb.descriptors.size() << endl;
	
	
	

//	VideoCapture cap(0);
//    if(!cap.isOpened()) {
//	  cout << "couldn't open video\n";
//	  return -1;
//    }
//	
//	Mat frame;
//	for(;;)
//    {
//		CascadeClassifier haarzoeker;
//        cap >> frame;
//		vector<Rect> faces;
//		Rect face;
//		haarzoeker.detectMultiScale(frame, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(10, 10) );
//		//cout << faces.size() << endl;
//		for (int i = 0; i <faces.size(); i++) {
//			face = faces[i];
//			cout << "ja!" << endl;
//			rectangle(frame, face.tl(), face.br(), Scalar(0, 255, 0));
//        } 
//        imshow("edges", frame);
//        if(waitKey(30) >= 0)
//			break;
//    }
//	
	


}

