/*
Project-3 | 21 Feb 2024 | Kirti Kshirsagar | Saikiran Juttu
This is a source file for tasks.h. It contains the function definitions for the tasks that are to be performed on the input image for the project Real-time 2-D Object Recognition.
*/
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdlib> 
#include <ctime> 
#include <cmath> 
#include <queue>
#include <unordered_map>
#include "tasks.h"

/*
Writes labeled features to a specified file, appending new entries without overwriting existing data.

Parameters:
filename: The name (and path) of the file where the labeled features should be stored.
labeledFeatures: A vector of LabeledFeature structs, each containing the label and features (centroid, theta, percent filled, and bounding box ratio) of an object.

If the file cannot be opened, an error message is printed to the standard error stream.
*/
void storeLabeledFeature(const std::string& filename, const std::vector<LabeledFeature>& labeledFeatures) {
     std::ofstream file(filename, std::ios::out | std::ios::app);
    if (file.is_open()) {
        // Check if the file is empty
        file.seekp(0, std::ios::end);
        bool isEmpty = file.tellp() == 0;

        // Write header line only in the start of file.
        if (isEmpty) {
            file << "Label,Centroid_X,Centroid_Y,Theta,Percent_Filled,BoundingBox_Ratio" << std::endl;
        }
        for (const auto& labeledFeature : labeledFeatures) {
            file << labeledFeature.label << " ";
            file << labeledFeature.feature.centroid.x << " " << labeledFeature.feature.centroid.y << " ";
            file << labeledFeature.feature.theta << " ";
            file << labeledFeature.feature.percentFilled << " ";
            file << labeledFeature.feature.boundingBoxRatio << " ";
            file << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

/*
Loads known objects' features and labels from the csv file into a vector.

Parameters:
filename: The path to the file containing the labeled feature vectors.

Returns:
A vector of LabeledFeature structs, where each struct contains the label and feature vector of a known object.

If the file cannot be opened, an error message is printed to the standard error stream, and an empty vector is returned.
*/
std::vector<LabeledFeature> loadKnownObjects(const std::string& filename) {
    std::vector<LabeledFeature> knownObjects;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return knownObjects;
    }
    // It takes the line, parses it and load features.
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string label;
        Features features;
        iss >> label >> features.centroid.x >> features.centroid.y >> features.theta >> features.percentFilled >> features.boundingBoxRatio;
        knownObjects.push_back({label, features});
    }

    file.close();
    return knownObjects;
}

/*
This is the main function of the project Real-time 2-D Object Recognition.
*/
int main(int argc, char *argv[]) {
    // Seed random number generator
    std::srand(static_cast<unsigned int>(std::time(0))); 
    // Initiate the video
    cv::VideoCapture capdev(2);
    if (!capdev.isOpened()) {
        std::cerr << "Unable to open video device" << std::endl;
        return -1;
    }

    // Creating VideoWriter object
    cv::VideoWriter videoWriter;
    int frame_width = static_cast<int>(capdev.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(capdev.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::Size refS(static_cast<int>(capdev.get(cv::CAP_PROP_FRAME_WIDTH)),
              static_cast<int>(capdev.get(cv::CAP_PROP_FRAME_HEIGHT)));
    std::cout << "Expected size: " << refS.width << " " << refS.height << std::endl;
    // Declaration of frames, features and label.
    cv::Mat frame, grayFrame, blurFrame,thresholdedFrame, dilatedFrame, cleanedFrame, segmentedFrame;
    std::vector<LabeledFeature> labeledFeatures; // Store labeled feature vectors
    std::string currentLabel;
    FeatureStdDeviations stddev;
    bool trainingMode = false;
    bool confusionMode = false;
    std::vector<UserFeedback> userFeedbacks;
    
    // Load known objects database from CSV file
    std::vector<LabeledFeature> knownObjectsDatabase;
    knownObjectsDatabase = loadKnownObjects("/media/sakiran/Internal/2nd Semester/PRCV/Project3/PRCV/Project_3/trial.csv");
    
    // Check if known objects are loaded correctly
    if (knownObjectsDatabase.empty()) {
        std::cerr << "No known objects loaded from the database" << std::endl;
        return -1;
    }

    // Compute standard deviations to use it in scaled Euclidean distances
    stddev = computeFeatureStdDeviations(knownObjectsDatabase);
    // Define the confusion matrix size
    const int objects = 5;
    // Define the number of images per class for each object
    const int images_per_object = 3;

    // Initialize confusion matrix
    cv::Mat confusionMatrix = cv::Mat::zeros(objects, objects, CV_32S);

    while (true) {
        capdev >> frame;
        if (frame.empty()) {
            std::cerr << "Frame is empty" << std::endl;
            break;
        }
        // Converting into Grayscale
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // Applying GaussianBlur to reduce noise in the frame
        cv::GaussianBlur(grayFrame, blurFrame, cv::Size(5,5), 0);

        // Applying dynamic thresholding
        applyDynamicThreshold(blurFrame, thresholdedFrame);

        // Applying custom morphological cleanup
        customDilate(thresholdedFrame, dilatedFrame, 1); 
        customErode(dilatedFrame, cleanedFrame, 1); 

        // minimum size for regions to keep
        int minRegionSize = 2000; 
        // Declaration of regionIDs, colors
        std::vector<int> regionIDs;
        std::vector<cv::Vec3b> colors;
        cv::Mat FeaturesOnSegmentedImage;
        std::vector<std::vector<cv::Point2f>> rotatedRectPointsList;
        // Vector to store computed features
        std::vector<Features> features;

        // This function semgents the image and add colours to each region
        segmentAndDisplayRegions(cleanedFrame, segmentedFrame, minRegionSize, regionIDs, colors);

        // Call the function to compute features and get modified image along with rotated rectangle points and axis of least central moment
        std::tie(FeaturesOnSegmentedImage, rotatedRectPointsList) = computeFeatures(segmentedFrame, regionIDs, colors, features);

        // Access and draw rotated rectangle points
        for (size_t i = 0; i < features.size(); i++) {
            for (size_t j = 0; j < rotatedRectPointsList[i].size(); j++) {
                cv::line(frame, rotatedRectPointsList[i][j], rotatedRectPointsList[i][(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
            }

            // Draw axis of least central moment on the main frame
            cv::Point pt1(features[i].centroid.x + 150 * cos(features[i].theta), features[i].centroid.y + 150 * sin(features[i].theta));
            //cv::Point pt2(features[i].centroid.x - 50 * cos(features[i].theta), features[i].centroid.y - 50 * sin(features[i].theta));
            cv::Point pt2(features[i].centroid.x, features[i].centroid.y);
            cv::arrowedLine(frame, pt2,pt1, cv::Scalar(0, 0, 255), 2);

            // Construct the text to display on main frame
            std::string featureText = "Centroid: (" + std::to_string(features[i].centroid.x) + ", " + std::to_string(features[i].centroid.y) + "), Percent Filled: " + std::to_string(features[i].percentFilled);

            // Display the text on the main frame
            cv::putText(frame, featureText, cv::Point(features[i].centroid.x + 10, features[i].centroid.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(50, 50, 100), 1);
        }

        //Displaying all the frames
        cv::imshow("live Video with bounding box and axis of least central moment", frame);
        cv::imshow("Thresholded video", thresholdedFrame);
        cv::imshow("Cleaned video", cleanedFrame);
        cv::imshow("Segmented video", segmentedFrame);
        cv::imshow("Segmented Video with bounding box and axis of least central moment", FeaturesOnSegmentedImage);

        // Assigning different keys to perform different tasks
        char key = cv::waitKey(10);

        // Exit from the program 
        if (key == 'q') {
            break;
        } else if (key == 's') {
            // Save screenshots if 's' is pressed
            cv::imwrite("/media/sakiran/Internal/2nd Semester/PRCV/Project3/PRCV/Project_3/watch.jpg", frame);
            cv::imwrite("/media/sakiran/Internal/2nd Semester/PRCV/Project3/PRCV/Project_3/thresholded2try.jpg", thresholdedFrame);
            cv::imwrite("/media/sakiran/Internal/2nd Semester/PRCV/Project3/PRCV/Project_3/2try.jpg", cleanedFrame);
            cv::imwrite("/media/sakiran/Internal/2nd Semester/PRCV/Project3/PRCV/Project_3/segmented-2try.jpg", segmentedFrame);
            cv::imwrite("/media/sakiran/Internal/2nd Semester/PRCV/Project3/PRCV/Project_3/modified-2try.jpg", FeaturesOnSegmentedImage);
            std::cout << "Screenshots saved!" << std::endl;
        } // Lets the user save a video recording of the effects applied to the live video
        else if (key == 'r' && !videoWriter.isOpened()) {
            // Start recording
            videoWriter.open("/media/sakiran/Internal/2nd Semester/PRCV/Project3/PRCV/Project_3/Recording1st.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), 25, cv::Size(frame_width, frame_height));
            std::cout << "Recording started!" << std::endl; 
            if (!videoWriter.isOpened()) {
                std::cerr << "Failed to open video writer" << std::endl;
            }
        } else if (key == 'p' && videoWriter.isOpened()) {
            // Stop recording
            videoWriter.release();
            std::cout << "Recording stopped!" << std::endl; 
        } else if (key == 'd') {
            // Enter classifyObject mode
            std::cout << "Classifying unknown object..." << std::endl;
            std::string nearestLabel = classifyObject(features.back(), knownObjectsDatabase, stddev);
            std::cout << "Nearest object label: " << nearestLabel<< std::endl;
             // Display the nearest label near its centroid on the main webcam frame
            std::cout << "Looping through known objects" << std::endl;
            for (const auto& knownObject : knownObjectsDatabase) {
                if (knownObject.label == nearestLabel) {
                    std::cout << "Found nearest label: " << knownObject.label << std::endl;
                    cv::Point textPosition(knownObject.feature.centroid.x, knownObject.feature.centroid.y - 20); 
                    cv::Point textPosition1(knownObject.feature.centroid.x, knownObject.feature.centroid.y + 20);
                    cv::putText(frame, nearestLabel, textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(55, 55, 150), 2);
                    cv::putText(frame, "Classified obj using NN", textPosition1, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(55, 55, 150), 1);
                    break;
                }
            }
            // Update the frame after adding the text
            cv::imshow("live Video with bounding box and axis of least central moment", frame);
            cv::waitKey(0);
        } else if (key == 'k') {
            // Enter classifyObject mode using KNN matching
            // Set the value of K
            int k = 4; 
            std::vector<std::string> nearestLabels = classifyObjectKNN(features.back(), knownObjectsDatabase, stddev, k);
            // Print the nearest labels
            std::cout << "Nearest Labels: ";
            for (const auto& label : nearestLabels) {
                std::cout << label << " ";
            }
            std::cout << std::endl;
            std::cout << std::endl;
            // Find the most frequent label among the K nearest object labels
            std::unordered_map<std::string, int> labelCounts;
            std::string mostFrequentLabel;
            int maxCount = 0;
            for (const auto& label : nearestLabels) {
                labelCounts[label]++;
                if (labelCounts[label] > maxCount) {
                    maxCount = labelCounts[label];
                    mostFrequentLabel = label;
                }
            }
            std::cout << "Object class found using KNN matching: " << mostFrequentLabel << std::endl;
    
            for (const auto& knownObject : knownObjectsDatabase) {
                if (knownObject.label == mostFrequentLabel) {
                    std::cout << "Found nearest label: " << knownObject.label << std::endl;
                    cv::Point textPosition(knownObject.feature.centroid.x + 10 , knownObject.feature.centroid.y - 20); 
                    cv::putText(frame, mostFrequentLabel, textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
                    cv::Point textPosition1(knownObject.feature.centroid.x + 10, knownObject.feature.centroid.y + 10); 
                    cv::putText(frame, "Classified obj using KNN", textPosition1, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
                    break;
                }
            }
            // Update the frame after adding the text
            cv::imshow("live Video with bounding box and axis of least central moment", frame);
            cv::waitKey(0);
        } else if (key == 'n') {
            // Pressing "n" key it enters training mode and takes labels to store data into a CSV file
            trainingMode = !trainingMode;
            if (trainingMode) {
                std::cout << "Training mode activated. Press 'n' to label objects." << std::endl;
                // Prompt user for label and store labeled feature vector
                std::cout << "Enter label for the current object: ";
                std::cin >> currentLabel;
                labeledFeatures.push_back({currentLabel, features.back()});
                storeLabeledFeature("/media/sakiran/Internal/2nd Semester/PRCV/Project3/PRCV/Project_3/trial.csv", labeledFeatures);
                std::cout << "Feature vector labeled and stored." << std::endl;
                labeledFeatures.clear();
            } else {
                std::cout << "Training mode deactivated." << std::endl;
            }

        } else if (key == 'c') {
            // Pressing "n" key it enters confusion mode where it compares true label given by user with the classified label
            std::cout << "Entering confusion mode..." << std::endl;
            std::string trueLabel;
            std::cout << "Enter the true label for the object: ";
            std::cin >> trueLabel;
            // Classify the object
            std::string classifiedLabel = classifyObject(features.back(), knownObjectsDatabase, stddev);

            // It matches the user given label with classifiedlabel
            bool correctClassification = (trueLabel == classifiedLabel);
            std::cout << "done bool correction..." << correctClassification << std::endl;

            // Update user feedbacks
            userFeedbacks.push_back({trueLabel, classifiedLabel});
            // Print userFeedbacks
            std::cout << "userFeedbacks:" << std::endl;
            for (const auto& feedback : userFeedbacks) {
                std::cout << "True Label: " << feedback.trueLabel << ", Classified Label: " << feedback.classifiedLabel << std::endl;
            }
            
            // Update confusion matrix based on user feedback
            updateConfusionMatrix(confusionMatrix, trueLabel, classifiedLabel, correctClassification);
            std::cout << "updateConfusionMatrix "<< updateConfusionMatrix << std::endl;

            // Exit the confusion mode once we take required number of samples
            if (userFeedbacks.size() >= objects * images_per_object) {
                confusionMode = false;
                std::cout << "Exiting confusion mode..." << std::endl;
            }
    
        }

        // Record processed frame
        if (videoWriter.isOpened()) {
            videoWriter.write(frame);
        }
    }
    // Print confusion matrix
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << confusionMatrix << std::endl;

    // release the video capture resources
    capdev.release();
    if (videoWriter.isOpened()) {
        videoWriter.release();
    }
    return 0;
}