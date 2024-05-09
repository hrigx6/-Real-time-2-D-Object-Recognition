// Author: Hrigved Suryawanshi & Haard shah (1/19/24)
// CODE: Implemention of filters 

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include "filters.h"
#include <fstream>
#include <sstream>
#include <iomanip>

// Function to apply a 5x5 Gaussian blur filter using 1D kernels
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    // Ensure the input image is not empty
    if (src.empty()) {
        return -1;
    }

    // Ensure the input image has 3 channels (BGR)
    if (src.channels() != 3) {
        return -1;
    }

    // Clone the source image for the destination
    dst = src.clone();

    // 1x5 blur kernel weights
    int kernel1x5[5] = {1, 2, 4, 2, 1};

    // Apply horizontal blur using a 1x5 filter
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 2; x < src.cols - 2; ++x) {
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0;
                uchar *ptr = src.ptr<uchar>(y) + (x - 2) * 3;  // Pointer to the pixel of interest
                for (int i = 0; i < 5; ++i) {
                    sum += ptr[i * 3 + c] * kernel1x5[i];
                }
                dst.ptr<uchar>(y)[x * 3 + c] = static_cast<uchar>(sum / 10);  // 10 is the sum of the kernel values
            }
        }
    }

    // Apply vertical blur using a 1x5 filter
    for (int y = 2; y < src.rows - 2; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0;
                for (int i = -2; i <= 2; ++i) {
                    sum += dst.ptr<uchar>(y + i)[x * 3 + c] * kernel1x5[i + 2];
                }
                dst.ptr<uchar>(y)[x * 3 + c] = static_cast<uchar>(sum / 10);  // 10 is the sum of the kernel values
            }
        }
    }

    return 0;
}

void dilateCustom(const cv::Mat& input, cv::Mat& output) {
    output = input.clone();

    for (int i = 1; i < input.rows - 1; i++) {
        for (int j = 1; j < input.cols - 1; j++) {
            if (input.at<uchar>(i, j) > 0 ||
                input.at<uchar>(i - 1, j) > 0 ||
                input.at<uchar>(i + 1, j) > 0 ||
                input.at<uchar>(i, j - 1) > 0 ||
                input.at<uchar>(i, j + 1) > 0) {
                output.at<uchar>(i, j) = 255;
            }
        }
    }
}

void erodeCustom(const cv::Mat& input, cv::Mat& output) {
    output = input.clone();

    for (int i = 1; i < input.rows - 1; i++) {
        for (int j = 1; j < input.cols - 1; j++) {
            if (input.at<uchar>(i, j) == 0 ||
                input.at<uchar>(i - 1, j) == 0 ||
                input.at<uchar>(i + 1, j) == 0 ||
                input.at<uchar>(i, j - 1) == 0 ||
                input.at<uchar>(i, j + 1) == 0) {
                output.at<uchar>(i, j) = 0;
            }
        }
    }
}


void classify(double percentageFilled, double aspectRatio){

    const std::string& csvFile = "features.csv";
    std::vector<double> testObjFeatures;
    testObjFeatures.push_back(percentageFilled);
    testObjFeatures.push_back(aspectRatio);
    
    std::ifstream dataFile(csvFile);
    if (!dataFile.is_open()) {
        std::cout << "Error while opening the file: " << csvFile << std::endl;
        return;
    }

    std::string row;
    std::vector<double> rowData;
    std::vector<double> l2DistanceVector;
    std::vector<std::string> firstElements;

    while (std::getline(dataFile, row)) {
        std::istringstream ss(row);
        std::string token;

        bool isFirstElement = true;

        while (std::getline(ss, token, ',')) {
            if (isFirstElement) {
                firstElements.push_back(token);
                isFirstElement = false;
                continue;
            } else {
                try {
                    double numToken = std::stod(token);
                    rowData.push_back(numToken);
                    
                } catch (const std::invalid_argument& e) {
                    // Handle the error, or you can choose to ignore this token and continue
                }
            }
        }

        if (rowData.size() > 0) {
            double l2Distance = 0;
            for (int i = 0; i < 3; ++i) {
                l2Distance += abs(testObjFeatures[i] - rowData[i]);
            }
            l2DistanceVector.push_back(l2Distance);
            l2Distance = 0;
        }

        rowData.clear();
    }

    dataFile.close();  // Close the file after reading

    

    auto minDistanceIterator = std::min_element(l2DistanceVector.begin(), l2DistanceVector.end());
    int minDistanceIndex = std::distance(l2DistanceVector.begin(), minDistanceIterator);


    std::cout << "Similar Object in csv: " << minDistanceIndex + 1 << std::endl;
    std::cout << "Object Name is " << firstElements[minDistanceIndex] << std::endl;  // Adjusted index
}

