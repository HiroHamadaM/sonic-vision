

#include "hand.h"



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


