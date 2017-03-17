### Vehicle Detection project for self-driving car
### Author by Eddy Chu at Mar 2017
####################################################################################################################
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
####################################################################################################################
from Lane_Finder import *
from Vehicle_Detection import *

##################################### Lane finder ##################################################################	
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
	
##################################### Vehicle Detection ##################################################################   		
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
	project_output = 'project_output_final.mp4'
	clip2 = VideoFileClip('project_video.mp4')
	project_clip = clip2.fl_image(process)	
	project_clip.write_videofile(project_output, audio=False)	
