
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
