# README
Name - **KIRTI KSHIRSAGAR** & **SAIKIRAN JUTTU**
- Using Windows 10, VScode, CMake,OpenCV 4.9(recent one) and C++.
- Not using any Time Travel Days
- Task-8 is video record capability so: [Demo of Recognizing one object](https://drive.google.com/file/d/1N5F-4br-ycPhdNtKliZ8IOUNJSyc8RH7/view?usp=drive_link) 
- Task-9 is the about implementing a secong method for classification, we have recorded a video of using both the classifiers [Demo of both classifiers](https://drive.google.com/file/d/1AFoqLXxugcIa1uGtxgGTe1So_fgyTtBC/view?usp=sharing)
## This is a Readme file for main.cpp and tasks.cpp file and this includes all the details that are necessary to run the program.

* We have integrated this project on live camera feed. Keypress **s** takes the screenshot of the live feed and saves it in the mentioned path. Keypress **q** is used to quit the live camera feed.

### Task-1 [Coded from scratch]
This is the primary task that is needed for our project to run. In this task, we implemented Dynamic Thresholding. This task involves separating dark objects from the white background in a video feed. Dynamic threshold is calculated using a simplified K-means clustering algorithm. We use the function applyDynamicThreshold().

### Task-2 [Coded from scratch]
This task focuses on cleaning up of the thresholded image, because in the thresholded image, we see some holes and noise. To address this, dilation is first applied to fill in the gaps, followed by erosion to remove the noise and refine the object shapes.
* For dilation, we use the function customDilate().
* For erosion, we use the function customErode().

### Task-3
In this task, we segment the image into regions. In this task, the cleaned binary image is processed to identify individual connected components, each representing a potential object. Components that are too small are discarded, in order to remove potential noise. The remaining components are then colored and displayed, providing a visual segmentation of the objects in the live feed. We use the function segmentAndDisplayRegions().

### Task-4 [Coded from scratch]
In this task, we compute features for each major region. We analyze each region by calculating its moments to determine centroids and central moments. This enables us to find the least central moment's axis angle. We also measure each region's bounding box dimensions and calculate the fill percentage by comparing the region's area to its bounding box area.

### Task-5
This task is for collecting training data. In this task, the activation of the **n** key in the main.cpp initiates the training mode, wherein the system captures the latest features of the object and prompts the user to assign a label to said object. This object label along with its features get appended to the CSV file for further recognition processes we implemented.


### Task-6
This task is based on classifying new objects in order to increase our DB. Keypress **d** in main.cpp activates classification mode. The system then compares the latest features of an unknown object with those of known objects from a CSV database. Using a scaled Euclidean distance, it calculates the closest match, identifying the unknown object's class based on the smallest distance.

### Task-7
This task focuses on evaluating the performance of our system. Keypress **c** in main.cpp starts the confusion matrix calculation mode. The user is asked to provide the true object label, which is then compared to the label determined by the distance metric from task-6. This comparison yields a boolean value, contributing to the update of a 5 x 5 confusion matrix initially filled with zeros. The matrix is then updated based on the correct and incorrect matches, with counts summarized and categorized by class.

### Task-8
- In this task, we capture the demo of our system working, so in order to start the recording of the frame that recognizes the object and stop the recording, we have integrated keypresses. The demo gets recordrd in the mp4 format.
    * Keypress **r** starts the recording of the demo.
    * Keypress **p** stops the recoding and saves it in the path given.

- [Demo](https://drive.google.com/file/d/1N5F-4br-ycPhdNtKliZ8IOUNJSyc8RH7/view?usp=drive_link) 

### Task-9
This task is used to implement a second classification method for our system. The task introduces a K-Nearest Neighbor (KNN) classification with \(K=4\) as a second method, activated by the **k** key in main.cpp. It compares an unknown object's features against a database, using scaled Euclidean distance to find the \(K=4\) closest matches. The object's class is determined by the most frequent label among these matches. This KNN approach outperforms the baseline method by considering multiple close matches, reducing misclassification risks associated with relying on a single closest match.

- We have recorded a video that demonstrates the usage of both the classification models i.e. the baseline model as well as the KNN classification model - [Demo](https://drive.google.com/file/d/1AFoqLXxugcIa1uGtxgGTe1So_fgyTtBC/view?usp=sharing)

# Extension:
For the extensions, we have implemented two tasks:
- Extension 1: We implemented three algorithms out of four that were asked to code from scratch, considering 1 to be the extension.
- Extension 2: In the task 3, our system is is to enabled recognise multiple objects simultaneously
- Extension 3: Apart from the 5 objects asked in the task to be recognized, we have additional 6 objects that our system can recognize. We took around 2-3 samples for each different object in different positions and orientations. Hence a total of 31 samples are present in our DB, thereby increasing the number of objects recognized by the system and testing the capability of identification our system can perform.
