### Vehicle Detection project for self-driving car
### Author by Eddy Chu at Mar 2017
#################################################################################################################
# The goals / steps of this project are the following:
#
# Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images 
#     and train a classifier Linear SVM classifier
# Apply a color transform and append binned color features, as well as histograms of color,
#     to the HOG feature vector.
# Note: for those first two steps don't forget to normalize the features and randomize a selection for training and testing.
# Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
# Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) 
#     and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# Estimate a bounding box for vehicles detected.
# Add the lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection
###################################################################################################################
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label
import pickle
from lane_finder import *
from Vehicle_Detection import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from moviepy.editor import VideoFileClip
from IPython.display import HTML
##################################### Camera calibration #########################################################
# prepare object points
nx = 9 #the number of inside corners in x
ny = 6 #the number of inside corners in y

# Read in an image
images = glob.glob('camera_cal/calibration*.jpg')

mtx, dist = cal_camera(images, nx, ny)

##################################### Lane finder ###########################################################	
def pipeline(img):
    global left, right
    img = np.copy(img)
    # Transform undistort pic to "birds-eye view" with thresholded binary image
    warped, Minv = perspective_color_thresh(img)
    
    # Find the fit for the current image either sliding windows or use previous data as base
    if not left.detected and not right.detected:
        left_current_fit, right_current_fit = sliding_windows(warped)
        left.detected = right.detected = True
    else: 
        left_current_fit, right_current_fit = fit_next_frame(warped, left.get_best_fit(), right.get_best_fit())
        
    # Sanity Check for current fit, if current fit failed on Sanity Check, use the previous fit.     
    left.set_current_fit(left_current_fit)
    right.set_current_fit(right_current_fit)
    if not line_checker(left.get_current_fit(), right.get_current_fit()):
        left_fit, right_fit = left.get_best_fit(), right.get_best_fit()
        left.detected = right.detected = False
    else:   
        left_fit, right_fit = left.get_fit(right.get_best_fit()), right.get_fit(left.get_best_fit())

    # Calculate curvature for the line and offset for the camera    
    curves = curvature_real_world(left_fit, right_fit)
    curvature = (curves[0] + curves[1])/2
    offest, direction = get_direction(left_fit, right_fit, img) 
    result = draw_lines(img, warped, left_fit, right_fit, Minv)
    
    # Add curvature and offset info to the image
    cv2.putText(result, 'Vehicle is %.2fm %s of center' % (offest, direction), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (255, 255, 255), 2)    
    cv2.putText(result, 'Radius of Curvature = %d(m)' % curvature, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    #left.best_fit = left_fit
    #right.best_fit = right_fit
    #print (left_fit, right_fit)
    return result
    
# Read in cars and notcars
cars = glob.glob('vehicles\*\*.png')
notcars = glob.glob('non-vehicles\*\*.png')
print('Number of Vehicle images:', len(cars))
print('Number of Non-Vehicle images:', len(notcars))
# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 500
cars = cars
notcars = notcars

### Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV++, HLS, YUV+, YCrCb+++
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

ystart = 400
ystop = 656
scale = 1.5
    
		
def process(image):
	global left, right
	line_img = pipeline(image)
	image = np.copy(image)
	out_img, heatmap = find_cars(image, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins)
	heat = apply_threshold(heatmap,1)
	heat_map = np.clip(heat, 0, 255)
	labels = label(heat_map)
	draw_img = draw_labeled_bboxes(line_img, labels)
	return draw_img
	
if __name__ == "__main__":
	global left, right	
	left = Line()
	right = Line()
	project_output = 'test_output.mp4'
	clip2 = VideoFileClip('test_video.mp4')
	project_clip = clip2.fl_image(process)	
	project_clip.write_videofile(project_output, audio=False)	