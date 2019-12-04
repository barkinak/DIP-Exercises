#include "stdafx.h"
#include "Dip5.h"

// Print matrices to console for testing purposes:
void Dip5::printMat(Mat& mat) {
	cout << endl;
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			cout << mat.at<float>(i, j) << " ";
			
		}
		cout << endl;
	}

}

// uses structure tensor to define interest points (foerstner)
void Dip5::getInterestPoints(Mat& img, double sigma, vector<KeyPoint>& points) {
	// 1 - Get gradient matrices in x and y direction:
	Mat Ky = createFstDevKernel(sigma);
	Mat Kx = Ky.t();
	
	Mat Gx, Gy;
	filter2D(img, Gx, CV_32FC1, Kx);
	filter2D(img, Gy, CV_32FC1, Ky);
	//showImage(Gx, "Gx", 0, true, true);

	// 2 - Multiplication of gradients to obtain second order terms (gx^2, gy^2 and gx.gy):
	Mat gxy, gxx, gyy;
	multiply(Gx, Gy, gxy);
	multiply(Gx, Gx, gxx);
	multiply(Gy, Gy, gyy);
	//showImage(gxy, "gxy", 0, true, true);

	// 3 - Average (Gaussian Window):
	GaussianBlur(gxy, gxy, Size(3, 3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(gxx, gxx, Size(3, 3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(gyy, gyy, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//showImage(gxy, "gxy after blur", 0, true, true);

	Mat A, B, C;
	A = gxx;
	B = gyy;
	C = gxy;

	// 4 - Trace of structure tensor:
	Mat Trace = Mat::zeros(img.rows, img.cols, CV_32FC1);
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			// Trace = A+B:
			Trace.at<float>(i, j) = A.at<float>(i, j) + B.at<float>(i, j);
		}
	}
	//showImage(Trace, "trace", 0, true, true);

	// 5 - Determinant of structure tensor:
	Mat Determinant = Mat::zeros(img.rows, img.cols, CV_32FC1);
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			// Determinant = A.B - C^2:
			Determinant.at<float>(i, j) = (A.at<float>(i, j) * B.at<float>(i, j)) - (C.at<float>(i, j) * C.at<float>(i, j));
		}
	}
	//showImage(Determinant, "Determinant", 0, true, true);

	// 6 - Weight calculation:
	Mat w = Mat::zeros(img.rows, img.cols, CV_32FC1);
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			// w = Det / Trace:
			if (Trace.at<float>(i, j) != 0) {
				w.at<float>(i, j) = Determinant.at<float>(i, j) / Trace.at<float>(i, j);
			}
		}
	}
	//showImage(w, "w", 0, true, true);
	
	// 7 - Weight non-max suppression:
	nonMaxSuppression(w);
	//showImage(w, "step 7", 0, true, true);

	// 8 - Weight thresholding:
	Mat w_thresh = Mat::zeros(w.rows, w.cols, CV_32FC1);
	float w_mean = 0;
	for (int i = 3; i < w.rows - 3; i++) {
		for (int j = 3; j < w.cols - 3; j++) {
			w_mean += w.at<float>(i, j);
		}
	}
	w_mean = w_mean / ((w.rows-3) * (w.cols-3));

	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			if (w.at<float>(i, j) > (0.5) && w.at<float>(i, j) < (1.5 * w_mean)) {
				w_thresh.at<float>(i, j) = w.at<float>(i, j);
			}
		}
	}
	//showImage(w_thresh, "step 8", 0, true, true);

	// 9 - Isotrophy calculation:
	Mat q = Mat::zeros(w.rows, w.cols, CV_32FC1);
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			// q = 4.Det / Trace^2:
			if (Trace.at<float>(i, j) != 0) {
				q.at<float>(i, j) = (4 * Determinant.at<float>(i, j)) / (Trace.at<float>(i, j) * Trace.at<float>(i, j));
			}
		}
	}
	showImage(q, "step 9", 0, true, true);
	
	// 10 - Isotrophy non-max suppression:
	nonMaxSuppression(q);
	//showImage(q, "step 10", 0, true, true);

	// 11/12 - Isotropy thresholding and finding keypoints:
	Mat q_thresh = Mat::zeros(w.rows, w.cols, CV_32FC1);
	for (int i = 0; i < q.rows; i++) {
		for (int j = 0; j < q.cols; j++) {
			if (w.at<float>(i, j) > 0.5 && w.at<float>(i, j) < w_mean * 1.5) {
				if (q.at<float>(i, j) > 0.5 && q.at<float>(i, j) < 0.75) {
					points.push_back(KeyPoint(j, i, 5));
				}
			}
		}
	}
	//showImage(q, "step 11", 0, true, true);
}

Mat Dip5::createGaussianKernel(int kSize) {
	// TO DO !!!
	Mat kernel = Mat::zeros(kSize, kSize, CV_32FC1);
	double sigma = 0.3 * ((kSize - 1) * 0.5 - 1) + 0.8;
	double mean = kSize / 2;
	double total = 0;

	for (int i = 0; i < kSize; i++) {
		for (int j = 0; j < kSize; j++) {
			double result = (1 / ((pow(sigma, 2) * 2 * CV_PI))) * exp(-0.5 * (pow(i - mean, 2) / (2 * pow(sigma, 2)) + pow(j - mean, 2) / (2 * pow(sigma, 2))));
			total += result;
			kernel.at<float>(i, j) = result;
		}
	}

	// Normalizing kernel values:
	for (int x = 0; x < kSize; x++) {
		for (int y = 0; y < kSize; y++) {
			kernel.at<float>(x, y) = (kernel.at<float>(x, y) / total);
		}
	}
	return kernel;
}

// creates kernel representing fst derivative of a Gaussian kernel in x-direction
/*
sigma	standard deviation of the Gaussian kernel
return	the calculated kernel
*/
Mat Dip5::createFstDevKernel(double sigma) {
	// TO DO !!!
	Mat result;
	Mat gaussianKernel = createGaussianKernel(3);

	Mat Gx = Mat::ones(3, 3, CV_32FC1);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			Gx.at<float>(i, j) = (-(i-1) * gaussianKernel.at<float>(i, j) / (pow(sigma, 2)));
		}
	}
	return Gx;
}

/* *****************************
GIVEN FUNCTIONS
***************************** */

// function calls processing function
/*
in		:  input image
points	:	detected keypoints
*/
void Dip5::run(Mat& in, vector<KeyPoint>& points) {
	this->getInterestPoints(in, this->sigma, points);
}

// non-maxima suppression
// if any of the pixel at the 4-neighborhood is greater than current pixel, set it to zero
Mat Dip5::nonMaxSuppression(Mat& img) {

	Mat out = img.clone();

	for (int x = 1; x<out.cols - 1; x++) {
		for (int y = 1; y<out.rows - 1; y++) {
			if (img.at<float>(y - 1, x) >= img.at<float>(y, x)) {
				out.at<float>(y, x) = 0;
				continue;
			}
			if (img.at<float>(y, x - 1) >= img.at<float>(y, x)) {
				out.at<float>(y, x) = 0;
				continue;
			}
			if (img.at<float>(y, x + 1) >= img.at<float>(y, x)) {
				out.at<float>(y, x) = 0;
				continue;
			}
			if (img.at<float>(y + 1, x) >= img.at<float>(y, x)) {
				out.at<float>(y, x) = 0;
				continue;
			}
		}
	}
	return out;
}

// Function displays image (after proper normalization)
/*
win   :  Window name
img   :  Image that shall be displayed
cut   :  whether to cut or scale values outside of [0,255] range
*/
void Dip5::showImage(Mat& img, const char* win, int wait, bool show, bool save) {

	Mat aux = img.clone();

	// scale and convert
	if (img.channels() == 1)
		normalize(aux, aux, 0, 255, CV_MINMAX);
	aux.convertTo(aux, CV_8UC1);
	// show
	if (show) {
		imshow(win, aux);
		waitKey(wait);
	}
	// save
	if (save)
		imwrite((string(win) + string(".png")).c_str(), aux);
}
