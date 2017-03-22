---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car Image](/output_images/car.png?raw=true)
![Non Car Image](/output_images/notcar.png?raw=true)


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

For car image:

![Car to be used for HOG](/output_images/car_before_hog.png?raw=true)
![Car HOG](/output_images/car_hog.png?raw=true)

For non-car image:

![Non Car to be used for HOG](/output_images/notcar_before_hog.png?raw=true)
![Non Car HOG](/output_images/notcar_hog.png?raw=true)

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally settled on the following parameters for HOG:

colorspace = 'YCrCb' 
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'

The colorspace and hog_channel was the only ones I had to change to get a good result. This was done mostly through experimentation and advice on the slack channel

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the function extract_features() function in cell 2. 

This was modified from the classroom to extract color histograms and spatial features in addition to HOG features.  This was done after the initial HOG features alone wasn't sufficient on the video pipeline

I further trained the SVM in code cell 6. Some of the approaches used were the usual Deep learning / ML techniques of shuffling the data as well as splitting the data into training and test sets of 80 / 20

The accuracy achieved in training was quite high at about 98.xx % and test set resulted in an accuracy of >94% giving me a good confidence that this is the right approach

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Code cell 8 contains the sliding window technique via the function find_cars(). I used a few different techniques here:

1) Used co-ordinates of 400, 656 to limit the search in the image (where cars are likely to be found)
2) I had to use a multi-scale approach, i.e. use scales of 1.3, 1.5 and 1.8 within the function in a loop to better improve detection as some frames were not being detected with a single scale due to car size differences

Sample images below:

![Sliding Window detection](/output_images/sliding_window1.png?raw=true)
![Sliding Window detection](/output_images/sliding_window2.png?raw=true)

You can see that there is a false positive in the 2nd image (though this could be argued as not false positive as it's detecting the car just at the far end of the road). We'll tackle this using heatmaps in the next section

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I used YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Furthermore, I added a heatmap to eliminate false positives. 

This was done using the functions add_heat(), appyl_threshold() and draw_labeled_bboxes()
Example of one such detection (same image as previous one given below):

![Sliding Window detection](/output_images/heatmap.png?raw=true)

Note that we've eliminated the false positive! Nice, isn't it?

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](/project_video_output.mp4?raw=true)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To avoid false positives I averaged the detections over 10 frames. The likelihood that a false positive over 10 frames is quite negligible and if it happens, then there is a good chance that the SVM itself is not properly trained. This is done via the process_image() function. 

When we find multiple detections on an area, we use the heatmap technique (previously discussed) to combine the bounding boxes into 1 central box. This is also integrated into the final pipeline.

A final technique was to add "heat" whenever a positive detection occurs. This ensures that only positive detections have the maximum possible values that will get picked up.

The threshold that has been used in the heatmap is 1 though experimentation with number of frames and threshold could lead to even better results


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially the project was taking too much time for processing when I was using only HOG features. (roughly 12 seconds per frame). This was cut drastically when I combined spatial and color features and then it started taking only about 4 seconds per frame. So, this was a good win

I also noticed that problems were cropping up in some frames and I was wasting a lot of time for the whole video to process. So, I used subclip() function to process specific parts and I also used this to process the final video in chunks and then combined it. This helped to process the video as my laptop was running into memory issues

Moving onto issues:

1) There are issues when 2 cars are close to each other. There is only 1 bounding box in such cases rather than 2

2) The pipeline could likely fail when there are too many cars in the image (for example, heavy traffic). It may not be robust enough to handle the averaging in such a scenario

3) It's very slow. So, this cannot be used in realtime.  

A few approaches that could be used to mitigate the above problems are:

a) Maybe using a different technique than frame averaging can help both #1 and #2 - i.e. draw bounding boxes better

b) Using a deep learning approach could make it faster and realtime. YOLO is another technique that I came across that seems to offer fast performance with great detection of various classes that could include pedestrians, different types of vehicles etc.

Overall, this was a challenging project, but enjoyable at the same time!
