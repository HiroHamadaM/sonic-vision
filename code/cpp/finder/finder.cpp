

#include <iostream>
#include <exception>

#include "finder.h"
#include "tools.h"
#include "settings.h"
#include "limb.h"


using namespace std;


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
    flip(small, small, 1);
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
    if (!(face == Rect())) {
        facepixels = hsv(face);
        calcHist( &facepixels,  1, channels, Mat(), histogram, 2,  histSize, ranges );
    }
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
    GaussianBlur( backproj, blurred, Size(31, 31), 0);
    threshold(blurred, th, 20, 255, THRESH_BINARY);
    morphologyEx(th, mask, MORPH_CLOSE, Mat());
}

void Finder::find_contours() {
    findContours( mask, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
}
	
void Finder::find_face() {
    haar.detectMultiScale(small, faces, 1.2, 2, CV_HAAR_SCALE_IMAGE +
        CV_HAAR_DO_CANNY_PRUNING + CV_HAAR_FIND_BIGGEST_OBJECT, Size(10, 10) );
    if (faces.size() > 0) {
        face = faces.at(0);
        face = sub_region(face);
    } else {
        face = Rect();
    }
}



void Finder::find_limbs() {
    Point facepoint = Point(face.x+face.width/2, face.y+face.height/2);
    vector<Limb> limbs;

    for (unsigned int i = 0; i < contours.size(); i++) {
        vector<Point> contour = contours.at(i);
        if (pointPolygonTest(contour, facepoint, false) > 0) {
            head = Limb(contour);
            limbs.push_back(head);
        } else {
            limbs.push_back(Limb(contour));
        }
    }
    sort(limbs.begin(), limbs.end(), compare_limbs);
    for(unsigned int i = 0; i < limbs.size(); i++) {
        if (limbs.at(i).contour == head.contour)
            continue;
        if (limbs.at(i).center.x < facepoint.x) {
            left_hand = limbs.at(i);
        } else {
            right_hand = limbs.at(i);
        }
    }
}

void Finder::visualize() {
    small.copyTo(visuals);
    convertScaleAbs(visuals, visuals, 0.2);
    small.copyTo(visuals, mask);
    rectangle(visuals, face.tl(), face.br(), Scalar(0, 255, 0));
	
    if (head.contour.size() > 0) {
        vector<vector<Point> > cs;
        cs.push_back(head.contour);
        drawContours( visuals, cs, -1, Scalar( 0, 0, 255 ));
    }

    if (left_hand.contour.size() > 0) {
        vector<vector<Point> > cs;
        cs.push_back(left_hand.contour);
        drawContours( visuals, cs, -1, Scalar( 0, 255, 0 ));
    }    
    
    if (right_hand.contour.size() > 0) {
        vector<vector<Point> > cs;
        cs.push_back(right_hand.contour);
        drawContours( visuals, cs, -1, Scalar( 255, 0, 0 ));
    }
    
    vector<Mat> presentation;

	presentation.push_back(small);
    presentation.push_back(backproj);
	presentation.push_back(blurred);
	presentation.push_back(th);
    presentation.push_back(mask);
    presentation.push_back(visuals);

    int w = MIN(XWINDOWS, presentation.size())*small_size.width;
    int h = ceil(float(presentation.size())/XWINDOWS)*small_size.height;
    combi.create(Size(w, h), CV_8UC3);
    for(unsigned int i=0; i < presentation.size(); i++) {
        Mat current = presentation.at(i);
        int xoffset = (i % XWINDOWS) * small_size.width;
        int yoffset = (i / XWINDOWS) * small_size.height;
        Mat roi(combi, Rect(xoffset, yoffset, small_size.width, small_size.height));
        if (current.channels() == 3) {
            current.copyTo(roi);
        } else {
			cvtColor(current, roi, CV_GRAY2RGB);
        }
    }
}

void Finder::mainloop() {
    for(;;) {
        grab_frame();
        find_face();
        make_histogram();
        make_backproject();
        make_mask();
		find_contours();
        find_limbs();
        visualize();

        imshow("Sonic Gesture", combi);
		
        if(waitKey(4) >= 0)
			break;
    }
}



