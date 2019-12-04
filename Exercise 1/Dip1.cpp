#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "Dip1.h"
using namespace cv;

Mat Dip1::doSomethingThatMyTutorIsGonnaLike(Mat & img)
{
	Mat newImg(img.rows, img.cols, CV_8UC3);

	for (int r = 0; r < img.rows; ++r) {
		for (int c = 0; c < img.cols; ++c) {
			newImg.at<cv::Vec3b>(r, c)[0] = img.at<Vec3b>(r, c)[2]; // blue pixel intensity
			newImg.at<cv::Vec3b>(r, c)[1] = img.at<Vec3b>(r, c)[1]; // green pixel intensity
			newImg.at<cv::Vec3b>(r, c)[2] = img.at<Vec3b>(r, c)[0]; // red pixel intensity
		}
	}
	return newImg;
}

void Dip1::run(String fname){
	Mat img = imread(fname);

	namedWindow("example");
	imshow("example", img);

	img = doSomethingThatMyTutorIsGonnaLike(img);

	imshow("example", img);
	imwrite("coolResult.png", img);
}
