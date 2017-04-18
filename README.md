# Vehicle Detection

In this project, the goal is to write a software pipeline to detect vehicles in a video.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Note: for those first two steps don't forget to normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. And [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment the training data.  

Pipeline detail
---
```sh
jupyter notebook Vehicle_Detection.ipynb
```
See the notebook for the pipeline satge

Usage
---
```sh
python Vehicle_Detection.py
```
This script will detect vehicle from provied video.

```sh
python Lane_Vehicle_Detection.py
```
This script will both detect vehicle and draw lane line from provied video.

