

#include "RPPG.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>

#include "opencv.hpp"

using namespace cv;
using namespace dnn;
using namespace std;

#define LOW_BPM 42
#define HIGH_BPM 240
#define REL_MIN_FACE_SIZE 0.4
#define SEC_PER_MIN 60
#define MAX_CORNERS 10
#define MIN_CORNERS 5
#define QUALITY_LEVEL 0.01
#define MIN_DISTANCE 25

bool RPPG::load(
                const int width, const int height, const double timeBase, const int downsample,
                const double samplingFrequency, const double rescanFrequency,
                const int minSignalSize, const int maxSignalSize,
                const string &logPath, const string &haarPath,
                const string &dnnProtoPath, const string &dnnModelPath,
                const bool log, const bool gui) {

   
    this->guiMode = gui;
    this->lastSamplingTime = 0;
    this->logMode = log;
    this->minFaceSize = Size(min(width, height) * REL_MIN_FACE_SIZE, min(width, height) * REL_MIN_FACE_SIZE);
    this->maxSignalSize = maxSignalSize;
    this->minSignalSize = minSignalSize;
    this->rescanFlag = false;
    this->rescanFrequency = rescanFrequency;
    this->samplingFrequency = samplingFrequency;
    this->timeBase = timeBase;

    // Load classifier
    haarClassifier.load(haarPath);
       

    // Setting up logfilepath
    ostringstream path_1;
    path_1 << logPath <<  "_min=" << minSignalSize << "_max=" << maxSignalSize << "_ds=" << downsample;
    this->logfilepath = path_1.str();

    // Logging bpm according to sampling frequency
    std::ostringstream path_2;
    path_2 << logfilepath << "_bpm.csv";
    logfile.open(path_2.str());
    logfile << "time;face_valid;mean;min;max\n";
    logfile.flush();

    // Logging bpm detailed
    std::ostringstream path_3;
    path_3 << logfilepath << "_bpmAll.csv";
    logfileDetailed.open(path_3.str());
    logfileDetailed << "time;face_valid;bpm\n";
    logfileDetailed.flush();

    return true;
}

void RPPG::exit() {
    logfile.close();
    logfileDetailed.close();
}

void RPPG::processFrame(Mat &frameRGB, Mat &frameGray, int time) {

    // Set time
    this->time = time;

    if (!faceValid) {

        cout << "Not valid, finding a new face" << endl;

        lastScanTime = time;
        detectFace(frameRGB, frameGray);

    } else if ((time - lastScanTime) * timeBase >= 1/rescanFrequency) {

        cout << "Valid, but rescanning face" << endl;

        lastScanTime = time;
        detectFace(frameRGB, frameGray);
        rescanFlag = true;

    } else {

        cout << "Tracking face" << endl;

        trackFace(frameGray,frameRGB);
    }

    if (faceValid) {

        // Update fps
        fps = getFps(t, timeBase);

        // Remove old values from raw signal buffer
        while (s.rows > fps * maxSignalSize) {
            push(s);
            push(t);
            push(re);
        }

        assert(s.rows == t.rows && s.rows == re.rows);

        // New values
        Scalar means = mean(frameRGB, mask);
		

		double avgG = 0,avgf=0,avgl=0,avgr=0;
		//avgGC(mask, avgG);
		//cout << "G通道平均值:" << avgG << endl;
		HSL_detector(this->forehead, avgf);
		HSL_detector(this->leftface, avgl);
		HSL_detector(this->rightface, avgr);
		avgG = (avgf + avgl + avgr) / 3;
		cout << "GChanelValue:" << avgG<<endl;
        // Add new values to raw signal buffer
        //double values[] = {means(0), avgG, means(2)};
		double values[] = { means(0), means(1), means(2) };
        s.push_back(Mat(1, 3, CV_64F, values));
        t.push_back(time);

        // Save rescan flag
        re.push_back(rescanFlag);

        // Update fps
        fps = getFps(t, timeBase);

        // Update band spectrum limits
        low = (int)(s.rows * LOW_BPM / SEC_PER_MIN / fps);
        high = (int)(s.rows * HIGH_BPM / SEC_PER_MIN / fps) + 1;

        // If valid signal is large enough: estimate
        if (s.rows >= fps * minSignalSize) {

            // Filtering
            extractSignal_g();
                   

            // HR estimation
            estimateHeartrate();

            // Log
            log();
        }

        if (guiMode) {
            draw(frameRGB);
        }
    }

    rescanFlag = false;

    frameGray.copyTo(lastFrameGray);
}

void RPPG::detectFace(Mat &frameRGB, Mat &frameGray) {

    //cout << "Scanning for faces…" << endl;
    vector<Rect> boxes = {};

    
    // Detect faces with Haar classifier
    haarClassifier.detectMultiScale(frameGray, boxes, 1.1, 2, CV_HAAR_SCALE_IMAGE, minFaceSize);
        

    if (boxes.size() > 0) {

        cout << "Found a face" << endl;

        setNearestBox(boxes);
        detectCorners(frameGray);
        updateROI();
        updateMask(frameGray,frameRGB);
        faceValid = true;

    } else {

        cout << "Found no face" << endl;
        invalidateFace();
    }
}

void RPPG::setNearestBox(vector<Rect> boxes) {
    int index = 0;
    Point p = box.tl() - boxes.at(0).tl();
    int min = p.x * p.x + p.y * p.y;
    for (int i = 1; i < boxes.size(); i++) {
        p = box.tl() - boxes.at(i).tl();
        int d = p.x * p.x + p.y * p.y;
        if (d < min) {
            min = d;
            index = i;
        }
    }
    box = boxes.at(index);
}

void RPPG::detectCorners(Mat &frameGray) {

    // Define tracking region
    Mat trackingRegion = Mat::zeros(frameGray.rows, frameGray.cols, CV_8UC1);
    Point points[1][4];
    points[0][0] = Point(box.tl().x + 0.22 * box.width,
                         box.tl().y + 0.21 * box.height);
    points[0][1] = Point(box.tl().x + 0.78 * box.width,
                         box.tl().y + 0.21 * box.height);
    points[0][2] = Point(box.tl().x + 0.70 * box.width,
                         box.tl().y + 0.65 * box.height);
    points[0][3] = Point(box.tl().x + 0.30 * box.width,
                         box.tl().y + 0.65 * box.height);
    const Point *pts[1] = {points[0]};
    int npts[] = {4};
    fillPoly(trackingRegion, pts, npts, 1, WHITE);

    // Apply corner detection
    goodFeaturesToTrack(frameGray,
                        corners,
                        MAX_CORNERS,
                        QUALITY_LEVEL,
                        MIN_DISTANCE,
                        trackingRegion,
                        3,
                        false,
                        0.04);
}

void RPPG::trackFace(Mat &frameGray, Mat &frameRGB) {

    // Make sure enough corners are available
    if (corners.size() < MIN_CORNERS) {
        detectCorners(frameGray);
    }

    Contour2f corners_1;
    Contour2f corners_0;
    vector<uchar> cornersFound_1;
    vector<uchar> cornersFound_0;
    Mat err;

    // Track face features with Kanade-Lucas-Tomasi (KLT) algorithm
    calcOpticalFlowPyrLK(lastFrameGray, frameGray, corners, corners_1, cornersFound_1, err);

    // Backtrack once to make it more robust
    calcOpticalFlowPyrLK(frameGray, lastFrameGray, corners_1, corners_0, cornersFound_0, err);

    // Exclude no-good corners
    Contour2f corners_1v;
    Contour2f corners_0v;
    for (size_t j = 0; j < corners.size(); j++) {
        if (cornersFound_1[j] && cornersFound_0[j]
            && norm(corners[j]-corners_0[j]) < 2) {
            corners_0v.push_back(corners_0[j]);
            corners_1v.push_back(corners_1[j]);
        } else {
            cout << "Mis!" << std::endl;
        }
    }

    if (corners_1v.size() >= MIN_CORNERS) {

        // Save updated features
        corners = corners_1v;

        // Estimate affine transform
        Mat transform = estimateRigidTransform(corners_0v, corners_1v, false);

        if (transform.total() > 0) {

            // Update box
            Contour2f boxCoords;
            boxCoords.push_back(box.tl());
            boxCoords.push_back(box.br());
            Contour2f transformedBoxCoords;

            cv::transform(boxCoords, transformedBoxCoords, transform);
            box = Rect(transformedBoxCoords[0], transformedBoxCoords[1]);

            // Update roi
            Contour2f roiCoords;
            roiCoords.push_back(roi.tl());
            roiCoords.push_back(roi.br());
            Contour2f transformedRoiCoords;
            cv::transform(roiCoords, transformedRoiCoords, transform);
            roi = Rect(transformedRoiCoords[0], transformedRoiCoords[1]);

            updateMask(frameGray,frameRGB);
        }

    } else {
        cout << "Tracking failed! Not enough corners left." << endl;
        invalidateFace();
    }
}

void RPPG::updateROI() {
    //this->roi = Rect(Point(box.tl().x + 0.3 * box.width, box.tl().y + 0.1 * box.height),
    //                Point(box.tl().x + 0.7 * box.width, box.tl().y + 0.25 * box.height));
	this->roi = Rect(Point(box.tl().x + 0.33 * box.width, box.tl().y + 0.08* box.height),
		Point(box.tl().x + 0.67 * box.width, box.tl().y + 0.25 * box.height));
	
	this->roi2 = Rect(Point(box.tl().x + 0.17 * box.width, box.tl().y + 0.5* box.height),
		Point(box.tl().x + 0.33 * box.width, box.tl().y + 0.67 * box.height));
	this->roi3 = Rect(Point(box.tl().x + 0.67 * box.width, box.tl().y + 0.5* box.height),
		Point(box.tl().x + 0.83 * box.width, box.tl().y + 0.67 * box.height));



}

void RPPG::updateMask(Mat &frameGray,Mat &frameRGB) {

    cout << "Update mask" << endl;
    mask = Mat::zeros(frameGray.size(), frameGray.type());
    rectangle(mask, this->roi, WHITE, FILLED);
	rectangle(mask, this->roi2, WHITE, FILLED);
	rectangle(mask, this->roi3, WHITE, FILLED);
	this->forehead = frameRGB(this->roi);
	this->leftface = frameRGB(this->roi2);
	this->rightface = frameRGB(this->roi3);



}

void RPPG::invalidateFace() {

    s = Mat1d();
    s_f = Mat1d();
    t = Mat1d();
    re = Mat1b();
    powerSpectrum = Mat1d();
    faceValid = false;
}

void RPPG::extractSignal_g() {

    // Denoise
    Mat s_den = Mat(s.rows, 1, CV_64F);
    denoise(s.col(1), re, s_den);

    // Normalise
    normalization(s_den, s_den);

    // Detrend
    Mat s_det = Mat(s_den.rows, s_den.cols, CV_64F);
    detrend(s_den, s_det, fps);

    // Moving average
    Mat s_mav = Mat(s_det.rows, s_det.cols, CV_64F);
    movingAverage(s_det, s_mav, 3, fmax(floor(fps/6), 2));

    s_mav.copyTo(s_f);

    // Logging
    if (logMode) {
        std::ofstream log;
        std::ostringstream filepath;
        filepath << logfilepath << "_signal_" << time << ".csv";
        log.open(filepath.str());
        log << "re;g;g_den;g_det;g_mav\n";
        for (int i = 0; i < s.rows; i++) {
            log << re.at<bool>(i, 0) << ";";
            log << s.at<double>(i, 1) << ";";
            log << s_den.at<double>(i, 0) << ";";
            log << s_det.at<double>(i, 0) << ";";
            log << s_mav.at<double>(i, 0) << "\n";
        }
        log.close();
    }
}

void RPPG::extractSignal_pca() {

    // Denoise signals
    Mat s_den = Mat(s.rows, s.cols, CV_64F);
    denoise(s, re, s_den);

    // Normalize signals
    normalization(s_den, s_den);

    // Detrend
    Mat s_det = Mat(s.rows, s.cols, CV_64F);
    detrend(s_den, s_det, fps);

    // PCA to reduce dimensionality
    Mat s_pca = Mat(s.rows, 1, CV_32F);
    Mat pc = Mat(s.rows, s.cols, CV_32F);
    pcaComponent(s_det, s_pca, pc, low, high);

    // Moving average
    Mat s_mav = Mat(s.rows, 1, CV_32F);
    movingAverage(s_pca, s_mav, 3, fmax(floor(fps/6), 2));

    s_mav.copyTo(s_f);

    // Logging
    if (logMode) {
        std::ofstream log;
        std::ostringstream filepath;
        filepath << logfilepath << "_signal_" << time << ".csv";
        log.open(filepath.str());
        log << "re;r;g;b;r_den;g_den;b_den;r_det;g_det;b_det;pc1;pc2;pc3;s_pca;s_mav\n";
        for (int i = 0; i < s.rows; i++) {
            log << re.at<bool>(i, 0) << ";";
            log << s.at<double>(i, 0) << ";";
            log << s.at<double>(i, 1) << ";";
            log << s.at<double>(i, 2) << ";";
            log << s_den.at<double>(i, 0) << ";";
            log << s_den.at<double>(i, 1) << ";";
            log << s_den.at<double>(i, 2) << ";";
            log << s_det.at<double>(i, 0) << ";";
            log << s_det.at<double>(i, 1) << ";";
            log << s_det.at<double>(i, 2) << ";";
            log << pc.at<double>(i, 0) << ";";
            log << pc.at<double>(i, 1) << ";";
            log << pc.at<double>(i, 2) << ";";
            log << s_pca.at<double>(i, 0) << ";";
            log << s_mav.at<double>(i, 0) << "\n";
        }
        log.close();
    }
}

void RPPG::extractSignal_xminay() {

    // Denoise signals
    Mat s_den = Mat(s.rows, s.cols, CV_64F);
    denoise(s, re, s_den);

    // Normalize raw signals
    Mat s_n = Mat(s_den.rows, s_den.cols, CV_64F);
    normalization(s_den, s_n);

    // Calculate X_s signal
    Mat x_s = Mat(s.rows, s.cols, CV_64F);
    addWeighted(s_n.col(0), 3, s_n.col(1), -2, 0, x_s);

    // Calculate Y_s signal
    Mat y_s = Mat(s.rows, s.cols, CV_64F);
    addWeighted(s_n.col(0), 1.5, s_n.col(1), 1, 0, y_s);
    addWeighted(y_s, 1, s_n.col(2), -1.5, 0, y_s);

    // Bandpass
    Mat x_f = Mat(s.rows, s.cols, CV_32F);
    bandpass(x_s, x_f, low, high);
    x_f.convertTo(x_f, CV_64F);
    Mat y_f = Mat(s.rows, s.cols, CV_32F);
    bandpass(y_s, y_f, low, high);
    y_f.convertTo(y_f, CV_64F);

    // Calculate alpha
    Scalar mean_x_f;
    Scalar stddev_x_f;
    meanStdDev(x_f, mean_x_f, stddev_x_f);
    Scalar mean_y_f;
    Scalar stddev_y_f;
    meanStdDev(y_f, mean_y_f, stddev_y_f);
    double alpha = stddev_x_f.val[0]/stddev_y_f.val[0];

    // Calculate signal
    Mat xminay = Mat(s.rows, 1, CV_64F);
    addWeighted(x_f, 1, y_f, -alpha, 0, xminay);

    // Moving average
    movingAverage(xminay, s_f, 3, fmax(floor(fps/6), 2));

    // Logging
    if (logMode) {
        std::ofstream log;
        std::ostringstream filepath;
        filepath << logfilepath << "_signal_" << time << ".csv";
        log.open(filepath.str());
        log << "r;g;b;r_den;g_den;b_den;x_s;y_s;x_f;y_f;s;s_f\n";
        for (int i = 0; i < s.rows; i++) {
            log << s.at<double>(i, 0) << ";";
            log << s.at<double>(i, 1) << ";";
            log << s.at<double>(i, 2) << ";";
            log << s_den.at<double>(i, 0) << ";";
            log << s_den.at<double>(i, 1) << ";";
            log << s_den.at<double>(i, 2) << ";";
            log << x_s.at<double>(i, 0) << ";";
            log << y_s.at<double>(i, 0) << ";";
            log << x_f.at<double>(i, 0) << ";";
            log << y_f.at<double>(i, 0) << ";";
            log << xminay.at<double>(i, 0) << ";";
            log << s_f.at<double>(i, 0) << "\n";
        }
        log.close();
    }
}

void RPPG::estimateHeartrate() {

    powerSpectrum = cv::Mat(s_f.size(), CV_32F);
    timeToFrequency(s_f, powerSpectrum, true);

    // band mask
    const int total = s_f.rows;
    Mat bandMask = Mat::zeros(s_f.size(), CV_8U);
    bandMask.rowRange(min(low, total), min(high, total) + 1).setTo(ONE);

    if (!powerSpectrum.empty()) {

        // grab index of max power spectrum
        double min, max;
        Point pmin, pmax;
        minMaxLoc(powerSpectrum, &min, &max, &pmin, &pmax, bandMask);

        // calculate BPM
        bpm = pmax.y * fps / total * SEC_PER_MIN;
        bpms.push_back(bpm);

        cout << "FPS=" << fps << " Vals=" << powerSpectrum.rows << " Peak=" << pmax.y << " BPM=" << bpm << endl;

        // Logging
        if (logMode) {
            std::ofstream log;
            std::ostringstream filepath;
            filepath << logfilepath << "_estimation_" << time << ".csv";
            log.open(filepath.str());
            log << "i;powerSpectrum\n";
            for (int i = 0; i < powerSpectrum.rows; i++) {
                if (low <= i && i <= high) {
                    log << i << ";";
                    log << powerSpectrum.at<double>(i, 0) << "\n";
                }
            }
            log.close();
        }
    }

    if ((time - lastSamplingTime) * timeBase >= 1/samplingFrequency) {
        lastSamplingTime = time;

        cv::sort(bpms, bpms, SORT_EVERY_COLUMN);

        // average calculated BPMs since last sampling time
        meanBpm = mean(bpms)(0);
        minBpm = bpms.at<double>(0, 0);
        maxBpm = bpms.at<double>(bpms.rows-1, 0);

        std::cout << "meanBPM=" << meanBpm << " minBpm=" << minBpm << " maxBpm=" << maxBpm << std::endl;

        bpms.pop_back(bpms.rows);
    }
}

void RPPG::log() {

    if (lastSamplingTime == time || lastSamplingTime == 0) {
        logfile << time << ";";
        logfile << faceValid << ";";
        logfile << meanBpm << ";";
        logfile << minBpm << ";";
        logfile << maxBpm << "\n";
        logfile.flush();
    }

    logfileDetailed << time << ";";
    logfileDetailed << faceValid << ";";
    logfileDetailed << bpm << "\n";
    logfileDetailed.flush();
}

void RPPG::draw(cv::Mat &frameRGB) {

    // Draw roi
    rectangle(frameRGB, roi, GREEN);

    // Draw bounding box
    rectangle(frameRGB, box, RED);

    // Draw signal
    if (!s_f.empty() && !powerSpectrum.empty()) {

        // Display of signals with fixed dimensions
        double displayHeight = box.height/2.0;
        double displayWidth = box.width*0.8;

        // Draw signal
        double vmin, vmax;
        Point pmin, pmax;
        minMaxLoc(s_f, &vmin, &vmax, &pmin, &pmax);
        double heightMult = displayHeight/(vmax - vmin);
        double widthMult = displayWidth/(s_f.rows - 1);
        double drawAreaTlX = box.tl().x + box.width + 20;
        double drawAreaTlY = box.tl().y;
        Point p1(drawAreaTlX, drawAreaTlY + (vmax - s_f.at<double>(0, 0))*heightMult);
        Point p2;
        for (int i = 1; i < s_f.rows; i++) {
            p2 = Point(drawAreaTlX + i * widthMult, drawAreaTlY + (vmax - s_f.at<double>(i, 0))*heightMult);
            line(frameRGB, p1, p2, RED, 2);
            p1 = p2;
        }

        // Draw powerSpectrum
        const int total = s_f.rows;
        Mat bandMask = Mat::zeros(s_f.size(), CV_8U);
        bandMask.rowRange(min(low, total), min(high, total) + 1).setTo(ONE);
        minMaxLoc(powerSpectrum, &vmin, &vmax, &pmin, &pmax, bandMask);
        heightMult = displayHeight/(vmax - vmin);
        widthMult = displayWidth/(high - low);
        drawAreaTlX = box.tl().x + box.width + 20;
        drawAreaTlY = box.tl().y + box.height/2.0;
        p1 = Point(drawAreaTlX, drawAreaTlY + (vmax - powerSpectrum.at<double>(low, 0))*heightMult);
        for (int i = low + 1; i <= high; i++) {
            p2 = Point(drawAreaTlX + (i - low) * widthMult, drawAreaTlY + (vmax - powerSpectrum.at<double>(i, 0)) * heightMult);
            line(frameRGB, p1, p2, RED, 2);
            p1 = p2;
        }
    }

    std::stringstream ss;

    // Draw BPM text
    if (faceValid) {
        ss.precision(3);
        ss << meanBpm << " bpm";
        putText(frameRGB, ss.str(), Point(box.tl().x, box.tl().y - 10), FONT_HERSHEY_PLAIN, 2, RED, 2);
    }

    // Draw FPS text
    ss.str("");
    ss << fps << " fps";
    putText(frameRGB, ss.str(), Point(box.tl().x, box.br().y + 40), FONT_HERSHEY_PLAIN, 2, GREEN, 2);

    // Draw corners
    //for (int i = 0; i < corners.size(); i++) {
    //    //circle(frameRGB, corners[i], r, WHITE, -1, 8, 0);
    //    line(frameRGB, Point(corners[i].x-5,corners[i].y), Point(corners[i].x+5,corners[i].y), GREEN, 1);
    //    line(frameRGB, Point(corners[i].x,corners[i].y-5), Point(corners[i].x,corners[i].y+5), GREEN, 1);
    //}
}

//void RPPG::avgGC(Mat &src,double &avgG) {
//	
//		//cvtColor(src, hsl_image, COLOR_BGR2HLS); //首先转换成到HSL空间,输入的roi必须检查是否已经取值，如果将未取值的roi送入cvtColor()将报错。
//		double sum = 0;
//		int count = 0;
//		for (int i = 0; i < src.rows; i++)
//		{
//			for (int j = 0; j < src.cols; j++)
//			{
//				int B = src.at<Vec3b>(i, j)[0];
//				int G = src.at<Vec3b>(i, j)[1];
//				int R = src.at<Vec3b>(i, j)[2];
//				if (!((R == 0) && (B == 0) && (G == 0))) {
//					int Vmax;
//					int Vmin;
//					if ((R > G) && (R > B)) Vmax = R;
//					else if ((G > R) && (G > B)) Vmax = G;
//					else Vmax = B;
//					if ((R < G) && (R < B)) Vmin = R;
//					else if ((G < R) && (G < B)) Vmin = G;
//					else Vmin = B;
//					if ((Vmax > 0)&&(Vmax!=Vmin)&& ((2 - (Vmax + Vmin))!=0)) {
//						int L = (Vmax + Vmin) / 2;
//						int S = (L < 0.5) ? (((Vmax - Vmin) / (Vmax + Vmin))) : ((Vmax - Vmin) / (2 - (Vmax + Vmin)));
//						int H;
//						if (S != 0) {
//
//							if (Vmax == R)
//							{
//								H = 60 * (G - B) / S;
//							}
//							else if (Vmax == G)
//							{
//								H = 120 + 60 * (B - R) / S;
//							}
//							else
//							{
//								H = 240 + 60 * (R - G) / S;
//							}
//							double LS_ratio = ((double)L) / ((double)S);
//							if (((H <= 14) || (H >= 165)) && (S >= 50) && (LS_ratio > 0.5) && (LS_ratio < 3.0))
//							{
//								sum += src.at<Vec3b>(i, j)[1];
//								count++;
//							}
//						}
//					}
//					
//				
//				}
//				//bool skin_pixel = (S >= 50) && (LS_ratio > 0.5) &&(LS_ratio < 3.0) && ((H <= 14)||(H >= 165));
//			}
//		}
//		avgG = sum / ((double)count);
//	
//}

void RPPG::HSL_detector(Mat& src, double& avgG)
{
	Mat hsl_image;
	cvtColor(src, hsl_image, COLOR_BGR2HLS); //首先转换成到HSL空间,输入的roi必须检查是否已经取值，如果将未取值的roi送入cvtColor()将报错。
	Mat dst = Mat::zeros(src.size(), CV_8UC3);
	double sum = 0;
	int count = 0;
	for (int i = 0; i < hsl_image.rows; i++)
	{
		for (int j = 0; j < hsl_image.cols; j++)
		{
			int H = hsl_image.at<Vec3b>(i, j)[0];
			int L = hsl_image.at<Vec3b>(i, j)[1];
			int S = hsl_image.at<Vec3b>(i, j)[2];
			double LS_ratio = ((double)L) / ((double)S);
			if (((H <= 14) || (H >= 165)) && (S >= 50) && (LS_ratio >0.5) && (LS_ratio< 3.0))
			{
				dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
				dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
				sum += src.at<Vec3b>(i, j)[1];
				dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
				count++;
			}
			//bool skin_pixel = (S >= 50) && (LS_ratio > 0.5) &&(LS_ratio < 3.0) && ((H <= 14)||(H >= 165));
		}
	}
	avgG = sum / ((double)count);
}
