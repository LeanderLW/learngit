

#ifndef RPPG_hpp
#define RPPG_hpp

#include <fstream>
#include <string>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/dnn.hpp>

#include <stdio.h>

using namespace cv;
using namespace dnn;
using namespace std;
;

class RPPG {

public:

    // Constructor
    RPPG() {;}

    // Load Settings
    bool load(
              const int width, const int height, const double timeBase, const int downsample,
              const double samplingFrequency, const double rescanFrequency,
              const int minSignalSize, const int maxSignalSize,
              const string &logPath, const string &haarPath,
              const string &dnnProtoPath, const string &dnnModelPath,
              const bool log, const bool gui);

    void processFrame(Mat &frameRGB, Mat &frameGray, int time);

    void exit();

    typedef vector<Point2f> Contour2f;

private:

    void detectFace(Mat &frameRGB, Mat &frameGray);
    void setNearestBox(vector<Rect> boxes);
    void detectCorners(Mat &frameGray);
    void trackFace(Mat &frameGray, Mat &frameRGB);
    void updateMask(Mat &frameGray, Mat &frameRGB);
    void updateROI();
    void extractSignal_g();
    void extractSignal_pca();
    void extractSignal_xminay();
    void estimateHeartrate();
    void draw(Mat &frameRGB);
    void invalidateFace();
    void log();
	//void avgGC(Mat &src,double &avgageG);
	void HSL_detector(Mat& src, double& avgG);

    CascadeClassifier haarClassifier;
    Net dnnClassifier;

    // Settings
    Size minFaceSize;
    int maxSignalSize;
    int minSignalSize;
    double rescanFrequency;
    double samplingFrequency;
    double timeBase;
    bool logMode;
    bool guiMode;

    // State variables
    int64_t time;
    double fps;
    int high;
    int64_t lastSamplingTime;
    int64_t lastScanTime;
    int low;
    int64_t now;
    bool faceValid;
    bool rescanFlag;

    // Tracking
    Mat lastFrameGray;
    Contour2f corners;

    // Mask
    Rect box;
    Mat1b mask;
    Rect roi;//额头
	Rect roi2;//脸颊
	Rect roi3;//脸颊

	Mat forehead;
	Mat leftface;
	Mat rightface;


    // Raw signal
    Mat1d s;
    Mat1d t;
    Mat1b re;

    // Estimation
    Mat1d s_f;
    Mat1d bpms;
    Mat1d powerSpectrum;
    double bpm = 0.0;
    double meanBpm;
    double minBpm;
    double maxBpm;

    // Logfiles
    ofstream logfile;
    ofstream logfileDetailed;
    string logfilepath;
};


#endif /* RPPG_hpp */
