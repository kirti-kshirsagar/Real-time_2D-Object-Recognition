/*
Project-3 | 21 Feb 2024 | Kirti Kshirsagar | Saikiran Juttu
This is a source file for tasks.h. It contains the function definitions for the tasks that are to be performed on the input image.
*/
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <ctime> 
#include <random>
#include <queue>
#include "tasks.h"
#include <unordered_map>

/*
Converts a string label into a unique index for use in a confusion matrix.

Parameters:
label: A string representing the label to be converted into an index.

Returns:
An integer index corresponding to the given label. If the label is new, it assigns and returns a new index. If the label has been encountered before, it returns the existing index.
*/
int labelToIndex(const std::string& label) {
    //function maps each unique label to an index in the confusion matrix
    static std::unordered_map<std::string, int> labelIndexMap;
    static int currentIndex = 0;
    
    auto it = labelIndexMap.find(label);
    if (it != labelIndexMap.end()) {
        return it->second; // Return index if label already exists
    } else {
        labelIndexMap[label] = currentIndex; // Assign a new index for unseen label
        return currentIndex++; // Increment index and return
    }
}

/*
Calculates the standard deviations of features from a set of known objects. This is used to normalize distances in feature space, making the comparison scale-invariant.

Parameters:
knownObjects: A vector of LabeledFeature, each containing feature vectors of known objects.

Returns:
A FeatureStdDeviations struct containing the standard deviations of each feature.
*/
FeatureStdDeviations computeFeatureStdDeviations(const std::vector<LabeledFeature>& knownObjects) {
    FeatureStdDeviations stddev;
    size_t numObjects = knownObjects.size();

    if (numObjects == 0) {
        // No known objects, hence return default stddev
        return stddev;
    }

    // Initialize sums for each feature
    double sumX = 0.0, sumY = 0.0, sumTheta = 0.0, sumPercentFilled = 0.0, sumBoundingBoxRatio = 0.0;

    // Compute sums
    for (const auto& object : knownObjects) {
        sumX += object.feature.centroid.x;
        sumY += object.feature.centroid.y;
        sumTheta += object.feature.theta;
        sumPercentFilled += object.feature.percentFilled;
        sumBoundingBoxRatio += object.feature.boundingBoxRatio;
    }

    // Compute means
    double meanX = sumX / numObjects;
    double meanY = sumY / numObjects;
    double meanTheta = sumTheta / numObjects;
    double meanPercentFilled = sumPercentFilled / numObjects;
    double meanBoundingBoxRatio = sumBoundingBoxRatio / numObjects;

    // Initialize squared differences for each feature
    double squaredDiffX = 0.0, squaredDiffY = 0.0, squaredDiffTheta = 0.0, squaredDiffPercentFilled = 0.0, squaredDiffBoundingBoxRatio = 0.0;

    // Compute squared differences
    for (const auto& object : knownObjects) {
        squaredDiffX += (object.feature.centroid.x - meanX) * (object.feature.centroid.x - meanX);
        squaredDiffY += (object.feature.centroid.y - meanY) * (object.feature.centroid.y - meanY);
        squaredDiffTheta += (object.feature.theta - meanTheta) * (object.feature.theta - meanTheta);
        squaredDiffPercentFilled += (object.feature.percentFilled - meanPercentFilled) * (object.feature.percentFilled - meanPercentFilled);
        squaredDiffBoundingBoxRatio += (object.feature.boundingBoxRatio - meanBoundingBoxRatio) * (object.feature.boundingBoxRatio - meanBoundingBoxRatio);
    }

    // Compute standard deviations and store them in the new object
    stddev.centroid_stddev = std::sqrt(squaredDiffX / numObjects);
    stddev.theta_stddev = std::sqrt(squaredDiffTheta / numObjects);
    stddev.percentFilled_stddev = std::sqrt(squaredDiffPercentFilled / numObjects);
    stddev.boundingBoxRatio_stddev = std::sqrt(squaredDiffBoundingBoxRatio / numObjects);
    return stddev;
}

/*
Calculates the scaled Euclidean distance between two feature vectors, considering the standard deviations
of the features to normalize the distance calculation.

Parameters:
f1, f2: Feature vectors of the objects to be compared.
stddev: The standard deviations of the features, used for scaling.

Returns:
The scaled Euclidean distance between the two feature vectors.
*/
double scaledEuclideanDistance(const Features& f1, const Features& f2, const FeatureStdDeviations& stddev) {
    double distance = 0.0;
    distance += ((f1.percentFilled - f2.percentFilled) / stddev.percentFilled_stddev) * ((f1.percentFilled - f2.percentFilled) / stddev.percentFilled_stddev);
    distance += ((f1.boundingBoxRatio - f2.boundingBoxRatio) / stddev.boundingBoxRatio_stddev) * ((f1.boundingBoxRatio - f2.boundingBoxRatio) / stddev.boundingBoxRatio_stddev);
    return distance;
}

/*
Classifies an unknown object by comparing its features to those of known objects. The closest match is found using the scaled Euclidean distance metric.

Parameters:
unknownFeatures: The feature vector of the unknown object.
knownObjects: A vector of LabeledFeature, each containing feature vectors of known objects.
stddev: The standard deviations of the features, used for scaling.

Returns:
The label of the closest matching known object.
*/
std::string classifyObject(const Features& unknownFeatures, const std::vector<LabeledFeature>& knownObjects, const FeatureStdDeviations& stddev) {
    double minDistance = std::numeric_limits<double>::max();
    std::string closestLabel;

    // Iterate over known objects to find the closest one
    for (const auto& knownObject : knownObjects) {
        double distance = scaledEuclideanDistance(unknownFeatures, knownObject.feature, stddev);
        if (distance < minDistance) {
            minDistance = distance;
            closestLabel = knownObject.label;
        }
    }
    std::cout << "closest Classified label " << closestLabel <<std::endl;

    return closestLabel;
}

/*
Updates a confusion matrix based on the comparison between the true label of an object and the label
assigned by the classifier.

Parameters:
confusionMatrix: The confusion matrix to be updated.
trueLabel: The actual label of the object.
classifiedLabel: The label assigned by the classifier.
correctClassification: A boolean indicating whether the classification was correct.
*/
void updateConfusionMatrix(cv::Mat& confusionMatrix, const std::string& trueLabel, const std::string& classifiedLabel, bool correctClassification) {
    int trueIndex = labelToIndex(trueLabel); // Convert true label to index in confusion matrix
    int classifiedIndex = labelToIndex(classifiedLabel); // Convert classified label to index in confusion matrix

    if (correctClassification) {
        // If classification is correct, increment the corresponding cell in the confusion matrix
        confusionMatrix.at<int>(trueIndex, trueIndex)++;
    } else {
        // If classification is incorrect, increment the cell corresponding to true label but classified incorrectly
        confusionMatrix.at<int>(trueIndex, classifiedIndex)++;
    }
}

/*
Applies a dynamic thresholding algorithm to segment an image into foreground (objects) and background.
The threshold value is determined using a simplified K-means clustering algorithm with two clusters.

input: The input grayscale image on which to apply thresholding.
output: The output binary image after thresholding.
numSamples: The number of pixel samples to use for the K-means algorithm.
*/
void applyDynamicThreshold(const cv::Mat &input, cv::Mat &output, int numSamples) {
    // Ensuring output is the same size as input, but in grayscale
    output = cv::Mat::zeros(input.size(), CV_8UC1);
    
    // Sample a set of pixels from the image
    std::vector<int> samples;
    samples.reserve(numSamples);
    for(int i = 0; i < numSamples; i++) {
        int x = std::rand() % input.cols;
        int y = std::rand() % input.rows;
        samples.push_back(static_cast<int>(input.at<uchar>(y, x)));
    }
    
    // Simple K-means with K=2. Initialize centroids to min and max of sampled values
    int centroid1 = *std::min_element(samples.begin(), samples.end()); // Object
    int centroid2 = *std::max_element(samples.begin(), samples.end()); // Background
    
    for(int iteration = 0; iteration < 10; iteration++) { // Limiting the number of iterations for real-time performance
        // Assign samples to nearest centroid
        std::vector<int> assignments(samples.size());
        for(size_t i = 0; i < samples.size(); i++) {
            assignments[i] = (std::abs(samples[i] - centroid1) < std::abs(samples[i] - centroid2)) ? 1 : 2;
        }
        
        // Updating centroids
        int sum1 = 0, count1 = 0;
        int sum2 = 0, count2 = 0;
        for(size_t i = 0; i < samples.size(); i++) {
            if(assignments[i] == 1) { 
                sum1 += samples[i]; 
                count1++; 
            }
            else { 
                sum2 += samples[i]; 
                count2++; 
            }
        }
        if(count1 > 0) centroid1 = sum1 / count1;
        if(count2 > 0) centroid2 = sum2 / count2;
    }
    
    // Calculating dynamic threshold as the average of the two centroids
    int thresholdValue = (centroid1 + centroid2) / 2;
    
    // Apply the threshold
    for(int i = 0; i < input.rows; i++) {
        for(int j = 0; j < input.cols; j++) {
            uchar pixelValue = input.at<uchar>(i, j);
            output.at<uchar>(i, j) = (pixelValue < thresholdValue) ? 255 : 0;
        }
    }
}

/*
Performs a dilation morphological operation on a binary image to enlarge white (foreground) regions.

input: The input binary image to dilate.
output: The output binary image after dilation.
kernelSize: The size of the structuring element used for dilation.
*/
void customDilate(const cv::Mat &input, cv::Mat &output, int kernelSize) {
    output = input.clone();
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            uchar maxPixel = 0;
            for (int ki = -kernelSize; ki <= kernelSize; ki++) {
                for (int kj = -kernelSize; kj <= kernelSize; kj++) {
                    int ni = i + ki; // New row index
                    int nj = j + kj; // New column index
                    if (ni >= 0 && ni < input.rows && nj >= 0 && nj < input.cols) {
                        maxPixel = std::max(maxPixel, input.at<uchar>(ni, nj));
                    }
                }
            }
            output.at<uchar>(i, j) = maxPixel;
        }
    }
}

/*
Performs an erosion morphological operation on a binary image to shrink white (foreground) regions.

input: The input binary image to erode.
output: The output binary image after erosion.
kernelSize: The size of the structuring element used for erosion.
*/
void customErode(const cv::Mat &input, cv::Mat &output, int kernelSize) {
    output = input.clone();
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            uchar minPixel = 255;
            for (int ki = -kernelSize; ki <= kernelSize; ki++) {
                for (int kj = -kernelSize; kj <= kernelSize; kj++) {
                    int ni = i + ki; // New row index
                    int nj = j + kj; // New column index
                    if (ni >= 0 && ni < input.rows && nj >= 0 && nj < input.cols) {
                        minPixel = std::min(minPixel, input.at<uchar>(ni, nj));
                    }
                }
            }
            // Only set to white if all pixels in the structuring element are white
            output.at<uchar>(i, j) = (minPixel == 255) ? 255 : 0;
        }
    }
}

/*
Segments the input binary image into connected regions and displays each region in a unique color.
Small regions below a certain size threshold are ignored.

input: The input binary image to segment.
output: The output image with each region colored differently.
minRegionSize: The minimum size a region must have to be kept and displayed.
*/
void segmentAndDisplayRegions(const cv::Mat &input, cv::Mat &output, int minRegionSize, std::vector<int>& regionIDs, std::vector<cv::Vec3b>& colors) {
    // Connected components with stats.
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(input, labels, stats, centroids);

    if (colors.empty()) {
        // If colors vector is empty, initialize with random colors
        std::random_device rd; 
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(0, 255);
        colors.push_back(cv::Vec3b(0,0,0)); // Background color
        for (int i = 1; i < nLabels; i++) {
            colors.push_back(cv::Vec3b(dist(mt), dist(mt), dist(mt)));
        }
    } else if (nLabels > static_cast<int>(colors.size())) {
        // If a new label is found, add new colors
        std::random_device rd; 
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(0, 255);
        for (int i = colors.size(); i < nLabels; i++) {
            colors.push_back(cv::Vec3b(dist(mt), dist(mt), dist(mt)));
        }
    }

    // Create the output image
    output = cv::Mat::zeros(input.size(), CV_8UC3);
    for (int r = 0; r < input.rows; r++) {
        for (int c = 0; c < input.cols; c++) {
            int label = labels.at<int>(r, c);
            if (label == 0) continue; // Skip background
            cv::Vec3b &pixel = output.at<cv::Vec3b>(r, c);
            int *stat = stats.ptr<int>(label);
            // Checks if the region size is above the threshold
            if (stat[cv::CC_STAT_AREA] >= minRegionSize) {
                pixel = colors[label];
                // Add the region ID to the vector if not already present
                if (std::find(regionIDs.begin(), regionIDs.end(), label) == regionIDs.end()) {
                    regionIDs.push_back(label);
                }
            }
        }
    }
}

/*
Applies a feature extraction algorithm using non zero pixels

input: The region map and regionIDs .
output: Features, Image with bounding box and axis of least central moment. 
*/
std::pair<cv::Mat, std::vector<std::vector<cv::Point2f>>> computeFeatures(const cv::Mat& output, const std::vector<int>& regionIDs, const std::vector<cv::Vec3b>& colors, std::vector<Features>& features) {
    cv::Mat image = output.clone();
    std::vector<std::vector<cv::Point2f>> rotatedRectPointsList;

    for (int regionID : regionIDs) {
        std::vector<std::vector<bool>> regionMask(image.rows, std::vector<bool>(image.cols, false));

        // Create a mask for the current region
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                if (output.at<cv::Vec3b>(y, x) == colors[regionID]) {
                    regionMask[y][x] = true;
                }
            }
        }

        // Find non-zero pixels (object pixels)
        std::vector<cv::Point> nonzeroPixels;
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                if (regionMask[y][x]) {
                    nonzeroPixels.push_back(cv::Point(x, y));
                }
            }
        }

        // Compute moments and features based on the mask
        double m00 = nonzeroPixels.size();
        double m10 = 0.0, m01 = 0.0, mu20 = 0.0, mu02 = 0.0, mu11 = 0.0;
        for (const auto& point : nonzeroPixels) {
            m10 += point.x;
            m01 += point.y;
        }

        // Calculate centroid to find central moment
        cv::Point centroid(static_cast<int>(m10 / m00), static_cast<int>(m01 / m00));

        // Recompute moments based on the centroid
        mu11 = mu20 = mu02 = 0.0;
        for (const auto& point : nonzeroPixels) {
            mu20 += (point.x - centroid.x) * (point.x - centroid.x);
            mu02 += (point.y - centroid.y) * (point.y - centroid.y);
            mu11 += (point.x - centroid.x) * (point.y - centroid.y);
        }

        // Calculate orientation
        double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);

        // Get the rotated rectangle of the object
        cv::Mat nonzeroPoints(nonzeroPixels);
        cv::RotatedRect rotatedRect = cv::minAreaRect(nonzeroPoints);
        double height = rotatedRect.size.height;
        double width = rotatedRect.size.width;
        if (height < width) {
            std::swap(height, width);
        }
        double percentFilled = m00 / (height * width);
        double boundingBoxRatio = height / width;

        // Store features
        Features currentFeatures;
        currentFeatures.centroid = centroid;
        currentFeatures.theta = theta;
        currentFeatures.percentFilled = percentFilled;
        currentFeatures.boundingBoxRatio = boundingBoxRatio;
        features.push_back(currentFeatures);

        // Draw rotated bounding box and axis of least central moment
        cv::Point2f rectPoints[4];
        rotatedRect.points(rectPoints);
        std::vector<cv::Point2f> rotatedRectPoints(std::begin(rectPoints), std::end(rectPoints));
        rotatedRectPointsList.push_back(rotatedRectPoints);
        for (int j = 0; j < 4; j++) {
            cv::line(image, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
        }
        cv::line(image, centroid, cv::Point(centroid.x + 50 * cos(theta), centroid.y + 50 * sin(theta)), cv::Scalar(0, 0, 255), 2);
    }

    return std::make_pair(image, rotatedRectPointsList);
}

/*
Calculates a scaled Euclidean distance between two feature vectors, adjusting for feature variance.

Parameters:
f1: The feature vector of the first object.
f2: The feature vector of the second object.
stddev: The standard deviations of the features across the dataset, used for scaling.

Returns:
- The scaled Euclidean distance between the two feature vectors as a double.
*/
double scalEuclideanDistance(const Features& f1, const Features& f2, const FeatureStdDeviations& stddev) {
    double distance = 0.0;
    distance += ((f1.percentFilled - f2.percentFilled) / stddev.percentFilled_stddev) * ((f1.percentFilled - f2.percentFilled) / stddev.percentFilled_stddev);
    distance += ((f1.boundingBoxRatio - f2.boundingBoxRatio) / stddev.boundingBoxRatio_stddev) * ((f1.boundingBoxRatio - f2.boundingBoxRatio) / stddev.boundingBoxRatio_stddev);
    return std::sqrt(distance);
}

/*
Classifies an unknown object using the K-Nearest Neighbors (KNN) approach based on scaled Euclidean distance.

Parameters:
unknownFeatures: The features of the unknown object to classify.
knownObjects: A vector containing the features and labels of known objects.
stddev: Standard deviations of the features for scaling distances.
k: The number of nearest neighbors to consider.

Returns:
A vector of strings containing the labels of the k nearest neighbors.
*/
std::vector<std::string> classifyObjectKNN(const Features& unknownFeatures, const std::vector<LabeledFeature>& knownObjects, const FeatureStdDeviations& stddev, int k) {
    std::vector<std::string> labels;

    // Calculate distances between query and training features
    std::vector<std::pair<double, std::string>> distances; // pair of distance and label
    for (const auto& knownObject : knownObjects) {
        double distance = scalEuclideanDistance(unknownFeatures, knownObject.feature, stddev);
        distances.push_back(std::make_pair(distance, knownObject.label));
    }

    // Sort distances in ascending order
    std::sort(distances.begin(), distances.end());

    // Get the labels of the k nearest neighbors
    for (int i = 0; i < k; i++) {
        labels.push_back(distances[i].second);
    }

    return labels;
}