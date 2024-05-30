/*
Project-3 | 21 Feb 2024
Kirti Kshirsagar | Saikiran Juttu
The is a header file for tasks.cpp. It contains the function declarations for the tasks that are to be performed on the input image for the project.
*/

#ifndef TASKS_H
#define TASKS_H

#include <opencv2/opencv.hpp>

struct Features {
    cv::Point centroid; // Centroid of the region
    double theta; // Angle of the axis of least central moment
    double percentFilled; // Percentage of the region filled
    double boundingBoxRatio; // Height/Width ratio of the bounding box
    std::vector<cv::Point> rotatedRectPoints; // Points defining the rotated rectangle enclosing the region
};
struct FeatureStdDeviations {
    double centroid_stddev;
    double theta_stddev;
    double percentFilled_stddev;
    double boundingBoxRatio_stddev;
};
// Data structure to store labeled feature vectors
struct LabeledFeature {
    std::string label;
    Features feature;
};
// Data structure to store user feedback
struct UserFeedback {
    std::string trueLabel;
    std::string classifiedLabel;
};
// Define a custom struct to store distances and labels
struct Neighbor {
    double distance;
    std::string label;
    bool operator<(const Neighbor& other) const {
        // Use greater than to have the smallest distance on top of the priority queue
        return distance > other.distance;
    }
};
// Structure to represent a point so it simplifies the handling of coordinates while drawing bounding box and axis of least central moment
struct Point {
    int x;
    int y;
    Point(int _x, int _y) : x(_x), y(_y) {}
};

int labelToIndex(const std::string& label);

FeatureStdDeviations computeFeatureStdDeviations(const std::vector<LabeledFeature>& knownObjects);

double scaledEuclideanDistance(const Features& f1, const Features& f2, const FeatureStdDeviations& stddev);

std::string classifyObject(const Features& unknownFeatures, const std::vector<LabeledFeature>& knownObjects, const FeatureStdDeviations& stddev);

std::vector<std::string> classifyObjectKNN(const Features& unknownFeatures, const std::vector<LabeledFeature>& knownObjects, const FeatureStdDeviations& stddev, int k);

void updateConfusionMatrix(cv::Mat& confusionMatrix, const std::string& trueLabel, const std::string& classifiedLabel, bool correctClassification);

void applyDynamicThreshold(const cv::Mat &input, cv::Mat &output, int numSamples = 1000);

void customDilate(const cv::Mat &input, cv::Mat &output, int kernelSize);

void customErode(const cv::Mat &input, cv::Mat &output, int kernelSize);

void segmentAndDisplayRegions(const cv::Mat &input, cv::Mat &output, int minRegionSize, std::vector<int>& regionIDs, std::vector<cv::Vec3b>& colors);

std::pair<cv::Mat, std::vector<std::vector<cv::Point2f>>> computeFeatures(const cv::Mat& output, const std::vector<int>& regionIDs, const std::vector<cv::Vec3b>& colors, std::vector<Features>& features);

#endif 
