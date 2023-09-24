# Highway-Lane-Line-Detection-
# 1. Introduction
This repository contains a lane detection and tracking program that uses a traditional (i.e. non-machine-learning) computer vision approach to detect lane lines under the following circumstances:

Can detect curved lane lines with shapes that can be described by a second-degree polynomial.
Detects exactly two lane lines, which are the left and right lane lines delimiting the lane that a vehicle is currently driving in. The lane lines of adjacent lanes cannot be detected. Detection will not work if the vehicle is not currently inside a lane and oriented along the direction of the lane.
If one of the two lane lines is successfully detected, but the other one is not, then the detection will be discarded as invalid. The reason for this is that the algorithm can only evaluate the validity of a detection if two lines were detected.
The purpose of this implementation is to demonstrate the capabilities of this approach, not to provide a real-time detector, which would obviously need to be implemented using parallel processing and in a compiled language.

# 2. Basic Use Instructions
The main purpose of this repository is to demonstrate the approach explained further below. The program uses camera calibration and perspective transform matrices that depend on the resolution of the video/image, the camera used to record it, and its mount position (i.e. the camera's exact perspective on the road). It is therefore not straight forward to use on your own videos or images. Here are some basic use instructions nonetheless.

Clone or fork this repository.

In order to process a video:

Open the file process_video.py with an editor.
Set the file path of the video to be processed and the name of the desired output video.
Set the file paths to load camera calibration and warp matrices. The camera calibration and warp matrices in this repository are for an input resolution of 1280x720 and a dashboard camera mounted at the top center of the windshield of a sedan. These matrices will only serve as very rough approximations for your own camera and setup. In order to generate a camera calibration matrix for your camera, please consult this OpenCV tutorial. In order to generate warp matrices and meters-per-pixel conversion parameters, please consult this Jupyter notebook.
Set all the options in the LaneTracker constructor, particularly the input image size and the size of the warped images that will result from the perspective transformation.
Once all these parameters are loaded/set, execute the file from the console: python process_video.py
Note: The processing of the video frames is performed by the process() method of the LaneTracker object. Since this method is passed

# Environement:
Python 3.x
Numpy
OpenCV
MoviePy (to process video files)

# Camera Calibration
1. Computation of the camera matrix and distortion coefficients with an example of distortion corrected calibration image.
The code for this step is called Camera_calibration.py.

![chessboard_undistorted](https://github.com/Vaiachal/Highway-Lane-Line-Detection-/assets/118053698/2a9188cf-aef5-4fed-a40e-98a310f6892f)

![test_image](https://github.com/Vaiachal/Highway-Lane-Line-Detection-/assets/118053698/7b9f1848-4701-4f4d-a010-8ff951e4df22)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I also save these two matrixes using np.savez such that I can use them later. Then, I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result:

# Pipeline architecture:
Load test images.
Apply Color Selection
Apply Canny edge detection.
Apply gray scaling to the images.
Apply Gaussian smoothing.
Perform Canny edge detection.
Determine the region of interest.
Apply Hough transform.
Average and extrapolating the lane lines.
Apply on video streams.
![road_undistorted](https://github.com/Vaiachal/Highway-Lane-Line-Detection-/assets/118053698/73dac749-c471-44c8-84b5-30ed98dfde00)

3. Canny Edge Detection
We need to detect edges in the images to be able to correctly detect lane lines. The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. The Canny algorithm involves the following steps:

Gray scaling the images: The Canny edge detection algorithm measures the intensity gradients of each pixel. So, we need to convert the images into gray scale in order to detect edges.
Gaussian smoothing: Since all edge detection results are easily affected by image noise, it is essential to filter out the noise to prevent false detection caused by noise. To smooth the image, a Gaussian filter is applied to convolve with the image. This step will slightly smooth the image to reduce the effects of obvious noise on the edge detector.
Find the intensity gradients of the image.
Apply non-maximum suppression to get rid of spurious response to edge detection.
Apply double threshold to determine potential edges.
Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges. If an edge pixel’s gradient value is higher than the high threshold value, it is marked as a strong edge pixel. If an edge pixel’s gradient value is smaller than the high threshold value and larger than the low threshold value, it is marked as a weak edge pixel. If an edge pixel's value is smaller than the low threshold value, it will be suppressed. The two threshold values are empirically determined and their definition will depend on the content of a given input image.

6. Averaging and extrapolating the lane lines
We have multiple lines detected for each lane line. We need to average all these lines and draw a single line for each lane line. We also need to extrapolate the lane lines to cover the full lane line length.
def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: The output lines from Hough Transform.
    """
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line
    
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 12): Line thickness. 
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
![sx_binary](https://github.com/Vaiachal/Highway-Lane-Line-Detection-/assets/118053698/860c76c3-6006-4347-b8ca-e5fe137b7d01)
![s_binary](https://github.com/Vaiachal/Highway-Lane-Line-Detection-/assets/118053698/dc3bacfe-de49-46eb-b245-c9602451ab1c)

    
7. Apply on video streams
Now, we'll use the above functions to detect lane lines from a video stream. The video inputs are in test_videos folder. The video outputs are generated in output_videos folder.
def frame_processor(image):
    """
    Process the input frame to detect lane lines.
        Parameters:
            image: Single video frame.
    """
    color_select = HSL_color_selection(image)
    gray         = gray_scale(color_select)
    smooth       = gaussian_smoothing(gray)
    edges        = canny_detector(smooth)
    region       = region_selection(edges)
    hough        = hough_transform(region)
    result       = draw_lane_lines(image, lane_lines(image, hough))
    return result

def process_video(test_video, output_video):
    """
    Read input video stream and produce a video file with detected lane lines.
        Parameters:
            test_video: Input video.
            output_video: A video file with detected lane lines.
    """
    input_video = VideoFileClip(os.path.join('test_videos', test_video), audio=False)
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(os.path.join('output_videos', output_video), audio=False)
