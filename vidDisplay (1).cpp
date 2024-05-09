// Author: Hrigved Suryawanshi & Hard shah (1/19/24)
// CODE: Read and display a video with implemented filters 

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "filters.h"   // Assuming this file contains your filter function declarations
#include "kmeans.h"
#include "filters.cpp"
using namespace cv;

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;
    cv::Mat grey;  // Declare the grey variable

    // Open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    // Get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // Identifies a window
    cv::Mat frame;
    cv::Mat filter;

    int colorState = 0; // Variable to track color state (if needed)
    cv::Mat blurred_frame; // New variable to store the blurred frame

    for (;;) {
        *capdev >> frame; // Get a new frame from the camera, treat as a stream

        if (frame.empty()) {
            printf("Frame is empty\n");
            break;
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

        Mat dilatedImage, erodedImage, dilated2, dilated3, erodedImage2;

        dilateCustom(segmented, dilatedImage);
        dilateCustom(dilatedImage, dilated2);
        dilateCustom(dilated2, dilated3);
        erodeCustom(dilated2, erodedImage);
        erodeCustom(erodedImage,erodedImage2);
        
        imshow("Dilated", dilatedImage);
        imshow("Eroded", erodedImage);

        // Connected Components Analysis
        Mat labeledImage;
        Mat stats, centroids;
        int connectivity = 8;  // You can change this based on your needs

        // Apply connected components analysis
        int numLabels = connectedComponentsWithStats(erodedImage, labeledImage, stats, centroids, connectivity, CV_32S);

        // Filter out small regions
        int minRegionSize = 1000;  // You can adjust this threshold
        for (int i = 1; i < numLabels; ++i) {
            if (stats.at<int>(i, CC_STAT_AREA) < minRegionSize) {
                erodedImage.setTo(0, labeledImage == i);
            }
        }

        // Display the regions
        Mat regionMap = Mat::zeros(erodedImage.size(), CV_8UC3);
        RNG rng(12345);  // Random number generator for picking colors

        for (int i = 1; i < numLabels; ++i) {
            Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            regionMap.setTo(color, labeledImage == i);

            computeFeatures(regionMap, i);
        }

        imshow("Regions", regionMap);


        //waitKey(1000);
        // See if there is a waiting keystroke
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;  // Break the loop if 'q' key is pressed
        }
    }
    

    // Clean up resources
    delete capdev;

    return 0;
}

#include <opencv2/opencv.hpp>

// Function to compute features for a specified region
void computeFeatures(const cv::Mat& regionMap, int regionID) {
    // Connected Components Analysis to get region pixels
    cv::Mat regionMask = (regionMap == regionID);
    cv::cvtColor(regionMask, regionMask, cv::COLOR_BGR2GRAY); // Convert to single-channel

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

    // Display features
    std::cout << "Region ID: " << regionID << std::endl;
    std::cout << "Percent Filled: " << percentFilled << "%" << std::endl;
    std::cout << "Bounding Box Aspect Ratio: " << aspectRatio << std::endl;
    std::cout << "-----------------------------" << std::endl;  // Separate features for different regions
}
