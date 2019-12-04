#include "Dip4.h"

// Performes a circular shift in (dx,dy) direction
/*
in       :  input matrix
dx       :  shift in x-direction
dy       :  shift in y-direction
return   :  circular shifted matrix
*/
Mat Dip4::circShift(Mat& in, int dx, int dy){
	// TO DO !!!

    Mat out = Mat::zeros(in.rows, in.cols, CV_32FC1);

    for (int x = 0; x < out.rows; x++) {
        for (int y = 0; y < out.cols; y++) {
            int newX = (x + dx);
            int newY = (y + dy);

            if (newX < 0) {
                newX = out.cols + newX;
            }
            if (newY < 0) {
                newY = out.rows + newY;
            }

            out.at<float>(newX, newY) = in.at<float>(x, y);
        }
    }

    return out;
}

// Function applies inverse filter to restorate a degraded image
/*
degraded :  degraded input image
filter   :  filter which caused degradation
return   :  restorated output image
*/
Mat Dip4::inverseFilter(Mat& degraded, Mat& filter){
	// TO DO !!!
    float e = 0.05;
    Mat inputdft;
    Mat outputdft;
    Mat filterdft = Mat::zeros(degraded.size(), CV_32FC1);
    vector<Mat> inputcomp;
    vector<Mat> output;
    vector<Mat> filtercomp;
    inputcomp.push_back(degraded.clone());
    inputcomp.push_back(Mat::zeros(degraded.size(), CV_32FC1));
    output.push_back(Mat::zeros(degraded.size(), CV_32FC1));
    output.push_back(Mat::zeros(degraded.size(), CV_32FC1));
    merge(inputcomp, inputdft);
    dft(inputdft, outputdft, DFT_COMPLEX_OUTPUT);
    split(outputdft, inputcomp);

    for (int x = 0; x < filter.rows; x++) {
        for (int y = 0; y < filter.cols; y++) {
            filterdft.at<float>(x, y) = filter.at<float>(x, y);
        }
    }
    filterdft = circShift(filterdft, -filter.rows / 2, -filter.cols / 2);
    filtercomp.push_back(filterdft.clone());
    filtercomp.push_back(Mat::zeros(degraded.size(), CV_32FC1));
    Mat outputf;
    merge(filtercomp, outputf);
    dft(outputf, filterdft, DFT_COMPLEX_OUTPUT);
    split(filterdft, filtercomp);
    Mat absolute = Mat(degraded.rows, degraded.cols, CV_32FC1);
    vector<Mat> magn;
    float max = 0;
    for (int n = 0; n < degraded.rows; n++) {
        for (int m = 0; m < degraded.cols; m++) {
            float real = filtercomp[0].at<float>(n, m);
            float imag = filtercomp[1].at<float>(n, m);
            float a = sqrt((real * real) + (imag * imag));
            absolute.at<float>(n, m) = a;
            if (a > max) {
                max = a;
            }
        }
    }
    float T = e * max;
    Mat Q = Mat(degraded.rows, degraded.cols, CV_32FC1);
    for (int n = 0; n < absolute.rows; n++) {
        for (int m = 0; m < absolute.cols; m++) {
            if (absolute.at<float>(n, m) >= T) {
                Q.at<float>(n, m) = 1 / absolute.at<float>(n, m);
            }
            else {
                Q.at<float>(n, m) = 1 / T;
            }
        }
    }
    magn.push_back(Q.clone());
    magn.push_back(Mat::zeros(Q.size(), CV_32FC1));
    merge(magn, Q);
    mulSpectrums(outputdft, Q, outputdft, 0);
    dft(outputdft, outputdft, DFT_INVERSE + DFT_REAL_OUTPUT + DFT_SCALE);
    return outputdft;
}

// Function applies wiener filter to restorate a degraded image
/*
degraded :  degraded input image
filter   :  filter which caused degradation
snr      :  signal to noise ratio of the input image
return   :   restorated output image
*/
Mat Dip4::wienerFilter(Mat& degraded, Mat& filter, double snr){
	// TO DO !!!
    float e = 0.05;
    Mat inputdft;
    Mat outputdft;
    Mat filterdft = Mat::zeros(degraded.size(), CV_32FC1);
    vector<Mat> inputcomp;
    vector<Mat> output;
    vector<Mat> filtercomp;
    inputcomp.push_back(degraded.clone());
    inputcomp.push_back(Mat::zeros(degraded.size(), CV_32FC1));
    output.push_back(Mat::zeros(degraded.size(), CV_32FC1));
    output.push_back(Mat::zeros(degraded.size(), CV_32FC1));
    merge(inputcomp, inputdft);
    dft(inputdft, outputdft, DFT_COMPLEX_OUTPUT);
    split(outputdft, inputcomp);

    for (int x = 0; x < filter.rows; x++) {
        for (int y = 0; y < filter.cols; y++) {
            filterdft.at<float>(x, y) = filter.at<float>(x, y);
        }
    }
    filterdft = circShift(filterdft, -filter.rows / 2, -filter.cols / 2);
    filtercomp.push_back(filterdft.clone());
    filtercomp.push_back(Mat::zeros(degraded.size(), CV_32FC1));
    Mat outputf;
    merge(filtercomp, outputf);
    dft(outputf, filterdft, DFT_COMPLEX_OUTPUT);
    split(filterdft, filtercomp);
    Mat absolute = Mat(degraded.rows, degraded.cols, CV_32FC1);
    vector<Mat> magn;
    float max = 0;
    for (int n = 0; n < degraded.rows; n++) {
        for (int m = 0; m < degraded.cols; m++) {
            float real = filtercomp[0].at<float>(n, m);
            float imag = filtercomp[1].at<float>(n, m);
            float a = (real * real) + (imag * imag);
            absolute.at<float>(n, m) = a;
            if (a > max) {
                max = a;
            }
        }
    }
    //float T = e * max;
    Mat Q = Mat(degraded.rows, degraded.cols, CV_32FC1);
    vector<Mat> Qk;
    Qk.push_back(Mat::zeros(Q.size(), CV_32FC1));
    Qk.push_back(Mat::zeros(Q.size(), CV_32FC1));
    for (int n = 0; n < absolute.rows; n++) {
        for (int m = 0; m < absolute.cols; m++) {
            float a = filtercomp[0].at<float>(n, m);
            float b = filtercomp[1].at<float>(n, m);
            float c = absolute.at<float>(n, m);
            float d = c + (1 / (snr*snr));
            Qk[0].at<float>(n,m) = a / d;
            Qk[1].at<float>(n,m) = -b / d;
        }
    }
    merge(Qk, Q);
    mulSpectrums(outputdft, Q, outputdft, 0);
    dft(outputdft, outputdft, DFT_INVERSE + DFT_REAL_OUTPUT + DFT_SCALE);
    return outputdft;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function calls processing function
/*
in                   :  input image
restorationType     :  integer defining which restoration function is used
kernel               :  kernel used during restoration
snr                  :  signal-to-noise ratio (only used by wieder filter)
return               :  restorated image
*/
Mat Dip4::run(Mat& in, string restorationType, Mat& kernel, double snr){

   if (restorationType.compare("wiener")==0){
      return wienerFilter(in, kernel, snr);
   }else{
      return inverseFilter(in, kernel);
   }

}

// function degrades the given image with gaussian blur and additive gaussian noise
/*
img         :  input image
degradedImg :  degraded output image
filterDev   :  standard deviation of kernel for gaussian blur
snr         :  signal to noise ratio for additive gaussian noise
return      :  the used gaussian kernel
*/
Mat Dip4::degradeImage(Mat& img, Mat& degradedImg, double filterDev, double snr){

    int kSize = round(filterDev*3)*2 - 1;
   
    Mat gaussKernel = getGaussianKernel(kSize, filterDev, CV_32FC1);
    gaussKernel = gaussKernel * gaussKernel.t();

    Mat imgs = img.clone();
    dft( imgs, imgs, CV_DXT_FORWARD, img.rows);
    Mat kernels = Mat::zeros( img.rows, img.cols, CV_32FC1);
    int dx, dy; dx = dy = (kSize-1)/2.;
    for(int i=0; i<kSize; i++) for(int j=0; j<kSize; j++) kernels.at<float>((i - dy + img.rows) % img.rows,(j - dx + img.cols) % img.cols) = gaussKernel.at<float>(i,j);
	dft( kernels, kernels, CV_DXT_FORWARD );
	mulSpectrums( imgs, kernels, imgs, 0 );
	dft( imgs, degradedImg, CV_DXT_INV_SCALE, img.rows );
	
    Mat mean, stddev;
    meanStdDev(img, mean, stddev);

    Mat noise = Mat::zeros(img.rows, img.cols, CV_32FC1);
    randn(noise, 0, stddev.at<double>(0)/snr);
    degradedImg = degradedImg + noise;
    threshold(degradedImg, degradedImg, 255, 255, CV_THRESH_TRUNC);
    threshold(degradedImg, degradedImg, 0, 0, CV_THRESH_TOZERO);

    return gaussKernel;
}

// Function displays image (after proper normalization)
/*
win   :  Window name
img   :  Image that shall be displayed
cut   :  whether to cut or scale values outside of [0,255] range
*/
void Dip4::showImage(const char* win, Mat img, bool cut){

   Mat tmp = img.clone();

   if (tmp.channels() == 1){
      if (cut){
         threshold(tmp, tmp, 255, 255, CV_THRESH_TRUNC);
         threshold(tmp, tmp, 0, 0, CV_THRESH_TOZERO);
      }else
         normalize(tmp, tmp, 0, 255, CV_MINMAX);
         
      tmp.convertTo(tmp, CV_8UC1);
   }else{
      tmp.convertTo(tmp, CV_8UC3);
   }
   imshow(win, tmp);
}

// function calls basic testing routines to test individual functions for correctness
void Dip4::test(void){

   test_circShift();
   cout << "Press enter to continue"  << endl;
   cin.get();

}

void Dip4::test_circShift(void){
   
   Mat in = Mat::zeros(3,3,CV_32FC1);
   in.at<float>(0,0) = 1;
   in.at<float>(0,1) = 2;
   in.at<float>(1,0) = 3;
   in.at<float>(1,1) = 4;
   Mat ref = Mat::zeros(3,3,CV_32FC1);
   ref.at<float>(0,0) = 4;
   ref.at<float>(0,2) = 3;
   ref.at<float>(2,0) = 2;
   ref.at<float>(2,2) = 1;
   
   if (sum((circShift(in, -1, -1) == ref)).val[0]/255 != 9){
      cout << "ERROR: Dip4::circShift(): Result of circshift seems to be wrong!" << endl;
      return;
   }
   cout << "Message: Dip4::circShift() seems to be correct" << endl;
}
