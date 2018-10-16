#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include<opencv2/objdetect/objdetect.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include<opencv2/imgproc/imgproc.hpp> 
#include "RPPG.hpp"
#include "opencv.hpp"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

#define DEFAULT_RPPG_ALGORITHM "g"//RPPG�㷨
#define DEFAULT_FACEDET_ALGORITHM "haar"//��������㷨
#define DEFAULT_RESCAN_FREQUENCY 1//��ɨ��Ƶ��
#define DEFAULT_SAMPLING_FREQUENCY 1//����Ƶ��
#define DEFAULT_MIN_SIGNAL_SIZE 5//��С�źŷ�Χ
#define DEFAULT_MAX_SIGNAL_SIZE 5//����źŷ�Χ
#define DEFAULT_DOWNSAMPLE 1 //�²�������Сͼ�񣨻��Ϊ�²�����subsampled���򽵲�����downsampled��������ҪĿ����������1��ʹ��ͼ�������ʾ����Ĵ�С��2�����ɶ�Ӧͼ�������ͼ��
// x means only every xth frame is used

#define HAAR_CLASSIFIER_PATH "haarcascade_frontalface_alt.xml"
#define DNN_PROTO_PATH "opencv/deploy.prototxt"
#define DNN_MODEL_PATH "opencv/res10_300x300_ssd_iter_140000.caffemodel"

using namespace cv;
using namespace std;




int main()
{

	//faceCascade.load("D://opencv//sources//data//haarcascades//haarcascade_frontalface_alt2.xml");
	bool offlineMode=true;
	VideoCapture capture;
	//capture.open(0);// ������ͷ
	capture.open("D:/��Ƭ/video2.mp4");// ����Ƶ
	if (!capture.isOpened())
	{
		cout << "open camera failed. " << endl;
		return -1;
	}

	//Load video information
	const int WIDTH = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	const int HEIGHT = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	const double FPS = capture.get(cv::CAP_PROP_FPS);
	const double TIME_BASE = 0.001;
	cout << "SIZE: " << WIDTH << "x" << HEIGHT << endl;
	cout << "FPS: " << FPS << endl;
	cout << "TIME BASE: " << TIME_BASE << endl;

	std::ostringstream window_title;
	window_title << WIDTH << "x" << HEIGHT;

	
	string LOG_PATH;
	std::ostringstream filepath;
	filepath << "Live_ffmpeg";
	LOG_PATH = filepath.str();
	bool log = false;
	bool gui = true;
	string baseline_input = "helloworld";
	//Set up rPPG
	RPPG rppg = RPPG();
	rppg.load(
		WIDTH, HEIGHT, TIME_BASE, DEFAULT_DOWNSAMPLE,
		DEFAULT_SAMPLING_FREQUENCY, DEFAULT_RESCAN_FREQUENCY,
		DEFAULT_MIN_SIGNAL_SIZE, DEFAULT_MAX_SIGNAL_SIZE,
		LOG_PATH, HAAR_CLASSIFIER_PATH,
		DNN_PROTO_PATH, DNN_MODEL_PATH,
		log, gui);



	Mat img, imgGray;
	int i = 0;
	//Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);//������ģ��������ǿͼ��ĶԱȶ�

	vector<Rect> faces;
	while (1)
	{
		capture >> img;// ��ȡͼ����img
		if (img.empty())
		{
			continue;
		}

		if (img.channels() == 3)
		{
			//blur(img, dst, Size(11, 11), Point(-1, -1));//��ֵģ��
			//GaussianBlur(img, dst, Size(5, 5), 11, 11);//��˹ģ��
			//filter2D(dst, dst, img.depth(), kernel);//��ǿ�Աȶ�
			cvtColor(img, imgGray, CV_RGB2GRAY);
		}
		else
		{
			imgGray = img;
		}
		equalizeHist(imgGray, imgGray);
		int time;
		if (offlineMode) time = (int)capture.get(CV_CAP_PROP_POS_MSEC);//��Ƶ�ļ��ĵ�ǰλ�ã��Ժ���Ϊ��λ������Ƶ����ʱ���
		else time = (cv::getTickCount()*1000.0) / cv::getTickFrequency();
		
		if (i % DEFAULT_DOWNSAMPLE == 0) {
			rppg.processFrame(img, imgGray, time);//����֡
		}
		else {
			cout << "SKIPPING FRAME TO DOWNSAMPLE!" << endl;
		}

		if (gui) {
			imshow(window_title.str(), img);
			if (waitKey(30) >= 0) break;
		}
		i++;
	}

	return 0;
}

