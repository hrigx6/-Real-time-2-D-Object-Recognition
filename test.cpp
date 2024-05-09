//Author: Hrigved Suryawanshi && Haard Shah
//Code:  Real-time 2-D Object Recognition


#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "filters.h"  
#include "kmeans.h"
#include <iomanip>
#include <fstream>
#include <sstream>
using namespace cv;

cv::Mat blurred_frame;

//writing CSV file
void writeFeaturesToCSV(const std::string& filename, const std::string& objectName, int regionID, double percentFilled, double aspectRatio, const cv::Mat& huMoments) {
    std::ofstream csvFile(filename, std::ios::app);  // Open the file in append mode

    if (!csvFile.is_open()) {
        std::cerr << "Error opening CSV file!" << std::endl;
        return;
    }

    // Write features to CSV
    csvFile << objectName << ",";
    csvFile << regionID << ",";
    csvFile << percentFilled << ",";
    csvFile << aspectRatio << ",";

    for (int i = 0; i < huMoments.rows; ++i) {
        csvFile << huMoments.at<double>(i, 0) << ",";
    }
    csvFile << std::endl;

    csvFile.close();
}

// function to compute features and display on the screen
void computeFeaturesAndDrawBox(cv::Mat& frame, const cv::Mat& regionMap, int regionID) {
    // Connected Components Analysis to get region pixels
    cv::Mat regionMask = (regionMap == regionID);
    
    // Convert binary mask to 3-channel image
    cv::Mat regionMaskColor;
    cv::cvtColor(regionMask, regionMaskColor, cv::COLOR_GRAY2BGR);

    int regionPixels = cv::countNonZero(regionMask);

    // Check if region is empty
    if (regionPixels == 0) {
        std::cout << "Invalid bounding box for Region ID: " << regionID << std::endl;
        return;
    }

    // Calculate the bounding box
    cv::Rect boundingBox = cv::boundingRect(regionMask);

    // Calculate features
    double percentFilled = static_cast<double>(regionPixels) / (regionMask.rows * regionMask.cols) * 100.0;
    double aspectRatio = static_cast<double>(boundingBox.width) / boundingBox.height;

    // Calculate Hu moments
    cv::Moments moments = cv::moments(regionMask);
    cv::Mat huMoments;
    cv::HuMoments(moments, huMoments);

    
    // Display features
    std::cout << "Region ID: " << regionID << std::endl;
    std::cout << "Percent Filled: " << percentFilled << "%" << std::endl;
    std::cout << "Bounding Box Aspect Ratio: " << aspectRatio << std::endl;
    std::cout << "Hu Moments: ";
    for (int i = 0; i < huMoments.rows; ++i) {
        std::cout << huMoments.at<double>(i, 0) << " ";
    }
    std::cout << std::endl;
    std::cout << "-----------------------------" << std::endl;  // Separate features for different regions

    // Display features on the frame
    std::stringstream ss;
    ss << "ID: " << regionID << " %Filled: " << percentFilled << " Aspect Ratio: " << aspectRatio;
    cv::putText(frame, ss.str(), cv::Point(boundingBox.x, boundingBox.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    

    // Draw the bounding box
    rectangle(frame, boundingBox, Scalar(0, 255, 0), 2);

    classify(percentFilled, aspectRatio);

    // Check for the 'o' key press
    char key = cv::waitKey(0);

    if (key == 'o') {
        // Prompt user for object name
        std::string objectName;
        std::cout << "Enter the object name: ";
        std::cin >> objectName;

        // Write features to CSV
        writeFeaturesToCSV("features.csv", objectName, regionID, percentFilled, aspectRatio, huMoments);
    }
    
}

int main(int argc, char* argv[]) {
    cv::Mat frame;

    // Load the input image
    frame = cv::imread("PRCV3/ball1.jpg");
    
    if (frame.empty()) {
        printf("Unable to load the image\n");
        return -1;
    }


    // Apply Gaussian blur to the frame
    blur5x5_2(frame, blurred_frame);

    int K = 2;  // Number of clusters
    TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 0.2);


    // Flatten the frame for k-means
    Mat flattened = blurred_frame.reshape(1, frame.total());
    flattened.convertTo(flattened, CV_32F);

    // Apply k-means
    Mat labels, centers;
    kmeans(flattened, K, labels, criteria, 3, cv::KMEANS_RANDOM_CENTERS, centers);

    // Normalize labels to range [0, 255]
    normalize(labels, labels, 0, 255, NORM_MINMAX, CV_8U);

    // Reshape the labels to the shape of the original frame
    Mat segmented = labels.reshape(0, frame.rows);

    imshow("Video", segmented);


    // Dilation and Erosion
    cv::Mat dilatedImage, erodedImage, dilated2, dilated3, erodedImage2, erodedImage3;

    dilateCustom(segmented, dilatedImage);
    dilateCustom(dilatedImage, dilated2);
    dilateCustom(dilated2, dilated3);
    erodeCustom(dilated2, erodedImage);
    erodeCustom(erodedImage, erodedImage2);
    erodeCustom(erodedImage2, erodedImage3);

    // Connected Components Analysis
    cv::Mat labeledImage;
    cv::Mat stats, centroids;
    int connectivity = 8;  // You can change this based on your needs

    // Apply connected components analysis
    int numLabels = connectedComponentsWithStats(erodedImage3, labeledImage, stats, centroids, connectivity, CV_32S);

    // Filter out small regions
    int minRegionSize = 10000;  // You can adjust this threshold
    for (int i = 1; i < numLabels; ++i) {
        if (stats.at<int>(i, CC_STAT_AREA) >= minRegionSize) {
            
            //computeFeaturesAndDrawBox(frame, labeledImage, i);
        }
    }

    // Draw bounding boxes and filter out small regions 
    cv::Mat regionMap = cv::Mat::zeros(erodedImage.size(), CV_8UC3);
    cv::RNG rng(12345);  

    for (int i = 1; i < numLabels; ++i) {
        if (stats.at<int>(i, CC_STAT_AREA) >= minRegionSize) {
            cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            regionMap.setTo(color, labeledImage == i);

            computeFeaturesAndDrawBox(regionMap, labeledImage, i);

            
        }
    }

    

    // Display
    imshow("Video", regionMap);
    imshow("threshold", segmented);
    imshow("dilated", dilated3);
    imshow("eroded", erodedImage3);
   
    cv::waitKey(0);

    return 0;
}
