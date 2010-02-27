

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
            //merge(vector<Mat>(current, current, current), temp);
            //temp.copyTo(roi);
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
        visualize();
        //imshow("small", small);
        //imshow("bp", backproj);
        imshow("Sonic Gesture", combi);
        if(waitKey(30) >= 0) break;
    }
}



