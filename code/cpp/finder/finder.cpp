//#define CV_NO_BACKWARD_COMPATIBILITY


#define SKIN "../../../data/hand/skin.png"
#define HEAD "../../../data/hand/head.png"
#define HANDA "../../../data/hand/a.png"
#define HANDB "../../../data/hand/b.png"
#define HANDC "../../../data/hand/c.png"
#define HANDD "../../../data/hand/d.png"
#define FACEHAAR "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"
#define DEVICE "../../../data/movies/heiligenacht.mp4"
#define WORKSIZE 200
#define XWINDOWS 3

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


Rect sub_region(Rect region) {
    Rect r;
    r.x = region.x + region.width * 0.2;
    r.y = region.y + region.height * 0.15;
    r.width = region.width * 0.6;
    r.height = region.height * 0.7;
    return r;
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

void Skin::find_face() {
    haarzoeker.detectMultiScale(img, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	if (faces.size() == 0) {
		cerr << "no faces found in image" << endl;
		throw exception();
	}
	Rect face = faces.at(0);
    Rect region = sub_region(face);
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

Finder::Finder(VideoCapture c) {
    if(!c.isOpened()) {
        cout << "couldn't open video\n";
        throw exception();
    }
    cap = c;
    haar = CascadeClassifier(FACEHAAR);
    cap >> frame;
    frame_size = frame.size();
    scale = float(WORKSIZE)/frame.rows;
    resize(frame, small, Size(), scale, scale);
    small_size = small.size();

}

void Finder::grab_frame() {
    cap >> frame;
    if (!frame.data) {
        cout << "end of movie" << endl;
        throw exception();
    }
    resize(frame, small, Size(), scale, scale);
    cvtColor(small, hsv, CV_BGR2HSV);
    cvtColor(small, bw, CV_BGR2GRAY);
}

void Finder::make_histogram() {
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    const float hranges[] = { 0, 180 };
    const float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    int channels[] = {0, 1};
    facepixels = hsv(face);
    calcHist( &facepixels,  1, channels, Mat(), histogram, 2,  histSize, ranges );
}

void Finder::make_backproject() {
    const float hranges[] = { 0, 180 };
    const float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    int channels[] = {0, 1};
    calcBackProject( &hsv, 1, channels, histogram, backproj, ranges );
}

void Finder::make_mask() {
    double maxVal;
    minMaxLoc(backproj, NULL, &maxVal);
    if (maxVal > 0) {
       float scaler = 255.0/maxVal;
       convertScaleAbs(backproj, backproj, scaler);
    }
    GaussianBlur( backproj, mask, Size(31, 31), 0);
    morphologyEx(mask, mask, MORPH_CLOSE, Mat());
    threshold(mask, mask, 20, 255, THRESH_BINARY);
}

void Finder::find_face() {
    haar.detectMultiScale(small, faces, 1.2, 2, CV_HAAR_SCALE_IMAGE +
        CV_HAAR_DO_CANNY_PRUNING + CV_HAAR_FIND_BIGGEST_OBJECT, Size(10, 10) );
    if (faces.size() > 0) {
        face = faces.at(0);
        face = sub_region(face);
    }
}

void Finder::visualize() {
    small.copyTo(visuals);
    convertScaleAbs(visuals, visuals, 0.2);
    small.copyTo(visuals, mask);
    rectangle(visuals, face.tl(), face.br(), Scalar(0, 255, 0));
    
    vector<Mat> presentation;

    presentation.push_back(backproj);
    presentation.push_back(small);
    presentation.push_back(mask);
    presentation.push_back(visuals);

    int w = MIN(XWINDOWS, presentation.size())*small_size.width;
    int h = ceil(float(presentation.size())/XWINDOWS)*small_size.height;
    combi.create(Size(w, h), CV_8UC3);
    for(int i=0; i < presentation.size(); i++) {
        Mat current = presentation.at(i);
        int xoffset = (i % XWINDOWS) * small_size.width;
        int yoffset = (i / XWINDOWS) * small_size.height;
        cout << xoffset << " " << yoffset << endl;
        Mat roi(combi, Rect(xoffset, yoffset, small_size.width, small_size.height));
        if (current.channels() == 3) {
            current.copyTo(roi);
        } else {
            merge(vector<Mat>(current, current, current), temp);
            temp.copyTo(roi);
        }

    }
//            if image.nChannels == 1:
//                cv.Merge(image, image, image, None, self.temp3)
//            else:
//                cv.Copy(image, self.temp3)
//            xoffset = (i % XWINDOWS) * self.smallsize[0]
//            yoffset = (i / XWINDOWS) * self.smallsize[1]
//            cv.SetImageROI(self.combined, (xoffset, yoffset, self.smallsize[0],
//                self.smallsize[1]))
//            cv.Copy(self.temp3, self.combined)
//            cv.PutText(self.combined, name, (5, 10), font, (0, 0, 200))
//            cv.ResetImageROI(self.combined)
//        return self.combined

}

void Finder::mainloop() {
    for(;;) {
        grab_frame();
        find_face();
        make_histogram();
        make_backproject();

        make_mask();
        visualize();
        //imshow("small", small);
        //imshow("bp", backproj);
        imshow("Sonic Gesture", combi);
        if(waitKey(30) >= 0) break;
    }
}

int main(int, char**) {
    setNumThreads(5);
    Skin skin(HEAD, FACEHAAR);
    Hand handa(HANDA, skin.histogram);
    Hand handb(HANDB, skin.histogram);
    cout << handa.descriptors.size() << endl;
    cout << handb.descriptors.size() << endl;
    VideoCapture cap(DEVICE);
    Finder finder(cap);
    finder.mainloop();
    return 0;
}

