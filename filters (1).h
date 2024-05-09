#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/core.hpp>
#include <vector>



// Declaring functions
int greyscale(cv::Mat &src, cv::Mat &dst);

int Sepia(cv::Mat &src, cv::Mat &dst);

int SepiaWithVignette(cv::Mat &src, cv::Mat &dst);

int blur5x5_1(cv::Mat &src, cv::Mat &dst);

int blur5x5_2(cv::Mat &src, cv::Mat &dst);

int sobelX3x3(cv::Mat &src, cv::Mat &dst);

int sobelY3x3(cv::Mat &src, cv::Mat &dst);

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

int animate(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold);

int Canny(cv::Mat &src, cv::Mat &dst, int a, int b);

void isolateStrongColor(const cv::Mat &inputImage, cv::Mat &outputImage, const cv::Scalar &targetColor, int threshold);

void invertColors(const cv::Mat &src, cv::Mat &dst);

void highPassFilter(const cv::Mat &src, cv::Mat &dst);

void processImage(const std::string &imagePath);

void dilateCustom(const cv::Mat &input, cv::Mat &output);
void erodeCustom(const cv::Mat &input, cv::Mat &output);
void classify(double percentageFilled, double aspectRatio);

void computeFeatures(const cv::Mat &regionMap, int regionID);

void computeFeaturesAndDrawBox(const cv::Mat& frame, const cv::Mat& regionMap, int regionID);

std::vector<cv::Mat> loadFeaturesFromCSV(const std::string& filename);

void extractFeaturesFromImage(const cv::Mat& image, cv::Mat& features);



#endif 

