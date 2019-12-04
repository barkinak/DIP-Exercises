#include "stdafx.h"
#include "Dip3.h"

// Generates a gaussian filter kernel of given size
/*
kSize:     kernel size (used to calculate standard deviation)
return:    the generated filter kernel
*/
Mat Dip3::createGaussianKernel(int kSize) {
	// TO DO !!!
	// Assumed that sigma_x = sigma_y, mean_x = mean_y ---> Can we assume this? 
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


// Performes a circular shift in (dx,dy) direction
/*
in       input matrix
dx       shift in x-direction
dy       shift in y-direction
return   circular shifted matrix
*/
Mat Dip3::circShift(Mat& in, int dx, int dy) {
	// TO DO !!!
	int x, y;

	Mat out = Mat::zeros(in.rows, in.cols, CV_32FC1);

	for (int i = 0; i < out.rows; i++) {
		for (int j = 0; j < out.cols; j++) {
			x = (i + dx) % out.cols;
			y = (j + dy) % out.rows;

			if (x < 0) {
				x = out.cols + x;
			}
			if (y < 0) {
				y = out.rows + y;
			}
			
			out.at<float>(x, y) = in.at<float>(i, j);
		}
	}

	return out;
}

//Performes a convolution by multiplication in frequency domain
/*
in       input image
kernel   filter kernel
return   output image
*/
Mat Dip3::frequencyConvolution(Mat& in, Mat& kernel) {
	// TO DO !!!
	Mat resized_kernel = Mat::zeros(in.rows, in.cols, CV_32FC1);

	for (int i = 0; i < kernel.rows; i++) {
		for (int j = 0; j < kernel.cols; j++) {
			resized_kernel.at<float>(i, j) = kernel.at<float>(i, j);
		}
	}

	Mat result = Mat::zeros(in.rows, in.cols, CV_32FC1);
	Mat input_dft, kernel_dft;

	resized_kernel = circShift(resized_kernel, -1, -1);
	dft(in, input_dft, 0);
	dft(resized_kernel, kernel_dft, 0);
	mulSpectrums(input_dft, kernel_dft, result, 0);
	dft(result, result, DFT_INVERSE + DFT_SCALE);
	return result;
}

// Performs UnSharp Masking to enhance fine image structures
/*
in       the input image
type     integer defining how convolution for smoothing operation is done
0 <==> spatial domain; 1 <==> frequency domain; 2 <==> seperable filter; 3 <==> integral image
size     size of used smoothing kernel
thresh   minimal intensity difference to perform operation
scale    scaling of edge enhancement
return   enhanced image
*/
Mat Dip3::usm(Mat& in, int type, int size, double thresh, double scale) {
	// some temporary images 
	Mat tmp(in.rows, in.cols, CV_32FC1);

	// calculate edge enhancement

	// 1: smooth original image
	//    save result in tmp for subsequent usage
	switch (type) {
	case 0:
		tmp = mySmooth(in, size, 0);
		break;
	case 1:
		tmp = mySmooth(in, size, 1);
		break;
	case 2:
		tmp = mySmooth(in, size, 2);
		break;
	case 3:
		tmp = mySmooth(in, size, 3);
		break;
	default:
		GaussianBlur(in, tmp, Size(floor(size / 2) * 2 + 1, floor(size / 2) * 2 + 1), size / 5., size / 5.);
	}

	// TO DO !!!
	for (int i = 0; i < in.rows; i++) {
		for (int j = 0; j < in.cols; j++) {
			tmp.at<float>(i, j) = in.at<float>(i, j) - tmp.at<float>(i, j);
		}
	}	

	for (int i = 0; i < in.rows; i++) {
		for (int j = 0; j < in.cols; j++) {
			if (tmp.at<float>(i, j) > thresh) {
				in.at<float>(i, j) = in.at<float>(i, j) + (scale * tmp.at<float>(i, j));
			}
		}
	}

	return in;
}

// convolution in spatial domain
/*
src:    input image
kernel:  filter kernel
return:  convolution result
*/
Mat Dip3::spatialConvolution(Mat& src, Mat& kernel) {
	Mat newImg = Mat::zeros(src.rows, src.cols, CV_32FC1);
	int kSize = kernel.rows; // Size of the kernel

	for (int j = (kSize / 2); j < src.rows - (kSize / 2); j++) {
		for (int k = (kSize / 2); k < src.cols - (kSize / 2); ++k) {
			float total = 0;
			float normalization_factor = kSize * kSize;

			for (int p = 0; p < kSize; p++) {
				for (int q = 0; q < kSize; q++) {
					if (j - p + 1 >= 0 && j - p + 1 < src.rows && k - q + 1 >= 0 && k - q + 1 < src.cols) {
						float a = kernel.at<float>(p, q);
						float b = src.at<float>((j - p + 1), (k - q + 1));
						total += a*b;
					}
					else {
						normalization_factor--;
					}
				}
			}

			if (normalization_factor == kSize*kSize) {
				newImg.at<float>(j, k) = total;
			}
			else {
				newImg.at<float>(j, k) = 0;
			}
		}
	}
	return newImg.clone();
}

// convolution in spatial domain by seperable filters
/*
src:    input image
size     size of filter kernel
return:  convolution result
*/
Mat Dip3::seperableFilter(Mat& src, int size) {

	// optional

	return src;

}

// convolution in spatial domain by integral images
/*
src:    input image
size     size of filter kernel
return:  convolution result
*/
Mat Dip3::satFilter(Mat& src, int size) {

	// optional

	return src;

}

/* *****************************
GIVEN FUNCTIONS
***************************** */

// function calls processing function
/*
in       input image
type     integer defining how convolution for smoothing operation is done
0 <==> spatial domain; 1 <==> frequency domain
size     size of used smoothing kernel
thresh   minimal intensity difference to perform operation
scale    scaling of edge enhancement
return   enhanced image
*/
Mat Dip3::run(Mat& in, int smoothType, int size, double thresh, double scale) {

	return usm(in, smoothType, size, thresh, scale);

}


// Performes smoothing operation by convolution
/*
in       input image
size     size of filter kernel
type     how is smoothing performed?
return   smoothed image
*/
Mat Dip3::mySmooth(Mat& in, int size, int type) {

	// create filter kernel
	Mat kernel = createGaussianKernel(size);

	// perform convolution
	switch (type) {
		case 0: return spatialConvolution(in, kernel);	// 2D spatial convolution
		case 1: return frequencyConvolution(in, kernel);	// 2D convolution via multiplication in frequency domain
		case 2: return seperableFilter(in, size);	// seperable filter
		case 3: return satFilter(in, size);		// integral image
		default: return frequencyConvolution(in, kernel);
	}
}

// function calls basic testing routines to test individual functions for correctness
void Dip3::test(void) {

	test_createGaussianKernel();
	test_circShift();
	test_frequencyConvolution();
	cout << "Press enter to continue" << endl;
	cin.get();

}

void Dip3::test_createGaussianKernel(void) {
	cout << "inside test_createGaussianKernel" << endl;
	Mat k = createGaussianKernel(11);

	if (abs(sum(k).val[0] - 1) > 0.0001) {
		cout << "ERROR: Dip3::createGaussianKernel(): Sum of all kernel elements is not one!" << endl;
		return;
	}
	if (sum(k >= k.at<float>(5, 5)).val[0] / 255 != 1) {
		cout << "ERROR: Dip3::createGaussianKernel(): Seems like kernel is not centered!" << endl;
		return;
	}
	cout << "Message: Dip3::createGaussianKernel() seems to be correct" << endl;
}

void Dip3::test_circShift(void) {

	Mat in = Mat::zeros(3, 3, CV_32FC1);
	in.at<float>(0, 0) = 1;
	in.at<float>(0, 1) = 2;
	in.at<float>(1, 0) = 3;
	in.at<float>(1, 1) = 4;
	Mat ref = Mat::zeros(3, 3, CV_32FC1);
	ref.at<float>(0, 0) = 4;
	ref.at<float>(0, 2) = 3;
	ref.at<float>(2, 0) = 2;
	ref.at<float>(2, 2) = 1;

	if (sum((circShift(in, -1, -1) == ref)).val[0] / 255 != 9) {
		cout << "ERROR: Dip3::circShift(): Result of circshift seems to be wrong!" << endl;
		return;
	}
	cout << "Message: Dip3::circShift() seems to be correct" << endl;
}

void Dip3::test_frequencyConvolution(void) {

	Mat input = Mat::ones(9, 9, CV_32FC1);
	input.at<float>(4, 4) = 255;
	Mat kernel = Mat(3, 3, CV_32FC1, 1. / 9.);

	Mat output = frequencyConvolution(input, kernel);

	if ((sum(output < 0).val[0] > 0) || (sum(output > 255).val[0] > 0)) {
		cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains too large/small values!" << endl;
		return;
	}
	float ref[9][9] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 1, 1, 1, 1, 1, 1, 1, 0 },
	{ 0, 1, 1, 1, 1, 1, 1, 1, 0 },
	{ 0, 1, 1, (8 + 255) / 9., (8 + 255) / 9., (8 + 255) / 9., 1, 1, 0 },
	{ 0, 1, 1, (8 + 255) / 9., (8 + 255) / 9., (8 + 255) / 9., 1, 1, 0 },
	{ 0, 1, 1, (8 + 255) / 9., (8 + 255) / 9., (8 + 255) / 9., 1, 1, 0 },
	{ 0, 1, 1, 1, 1, 1, 1, 1, 0 },
	{ 0, 1, 1, 1, 1, 1, 1, 1, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
	for (int y = 1; y<8; y++) {
		for (int x = 1; x<8; x++) {
			if (abs(output.at<float>(y, x) - ref[y][x]) > 0.0001) {
				cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains wrong values!" << endl;
				return;
			}
		}
	}
	cout << "Message: Dip3::frequencyConvolution() seems to be correct" << endl;
}
