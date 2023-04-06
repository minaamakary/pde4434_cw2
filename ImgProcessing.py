import os
import cv2
from cv2 import cuda_BufferPool # import cv2
#from cv2 import cuda_BufferPool # import cv2
import numpy as np
import matplotlib.pyplot as plt



#FOR LOOP FOR LOADING THE IMAGES IN FOR PRE PROCESSING
#loading the folder of the iamges
imageFolder = '/Users/minamakary/Documents/pde4434_cw2/myDataset/b0'


for filename in os.listdir(imageFolder):
    # Make sure the file is an image file
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the image using OpenCV
        image_path = os.path.join(imageFolder, filename)
        image = cv2.imread(image_path)
        cv2.imshow('image', image)
        cv2.waitKey(0)



imageFolder = '/Users/minamakary/Documents/pde4434_cw2/myDataset/b0'

images = {}  # creating a storage dictionary to save the images

for filename in os.listdir(imageFolder): #for images in the folder, check if its an image file with image extension and load it using open cv
    # Make sure the file is an image file
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the image using OpenCV
        image_path = os.path.join(imageFolder, filename)
        image = cv2.imread(image_path)
        
        
        #DETERMINE THE COLOR
        hsvGreen = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #print(hsvGreen)
        
        # Threshold the image using a binary threshold
        #blurring edges detection using canny and thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.blur(gray,(5,5))
        edgesDetect = cv2.Canny(blur_img, 100, 200) #Canny edge detection technique
        _, thresh = cv2.threshold(blur_img, 130, 130, cv2.THRESH_BINARY)
        #contours, hierarchy = cv2.findContours(edgesDetect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(edgesDetect, contours, -1, (0, 255, 0), 3)
        images[filename] = thresh  # store the thresholded image in the dictionary with the filename as the key


        
        # Display the processed image
        cv2.imshow('thresholded image', thresh)
        cv2.waitKey(0)
        
# Thresholded images
#image1_thresh = images['image1.jpg']
#image2_thresh = images['image2.jpg']


#rows, cols, channels = image1_thresh.shape

#plotting the image graph
#plt.imshow(image1_thresh)
#plt.show()

#finding mean and median
#mean_color = cv2.mean(image1_thresh)

#median_color = cv2.mean(image2_thresh)
