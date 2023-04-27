import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from tensorflow import *

#FOR LOOP FOR LOADING THE IMAGES IN FOR PRE PROCESSING
#loading the folder of the iamges
imageFolder = '/Users/minamakary/Documents/pde4434_cw2/myDataset/blue_0'


for filename in os.listdir(imageFolder):
    # Make sure the file is an image file
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the image using OpenCV
        image_path = os.path.join(imageFolder, filename)
        image = cv2.imread(image_path)
        cv2.imshow('image', image)
        cv2.waitKey(0)



imageFolder = '/Users/minamakary/Documents/pde4434_cw2/myDataset/red_0'

images = {}  # creating a storage dictionary to save the images

for filename in os.listdir(imageFolder): #for images in the folder, check if its an image file with image extension and load it using open cv
    # Make sure the file is an image file
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the image using OpenCV
        image_path = os.path.join(imageFolder, filename)
        image = cv2.imread(image_path)
        
        
        #DETERMINE THE COLOR
        hsvBlue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #print(hsvBlue)
        
        # Threshold the image using a binary threshold
        #blurring edges detection using canny and thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.blur(gray,(5,5))
        edgesDetect = cv2.Canny(blur_img, 100, 200) #Canny edge detection technique
        _, thresh = cv2.threshold(blur_img, 130, 130, cv2.THRESH_BINARY)
        
        #contours not working
        #contours, hierarchy = cv2.findContours(edgesDetect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(edgesDetect, contours, -1, (0, 255, 0), 3)
        images[filename] = thresh  # store the thresholded image in the dictionary with the filename as the key


        
        # Display the processed image
        cv2.imshow('thresholded image', thresh)
        cv2.waitKey(0)

#plotting the image graph
plt.imshow(thresh)
plt.show()
#finding mean and median
mean_color = cv2.mean(thresh)
mean_stdDev_color = cv2.meanStdDev(thresh)
#mean and standard deviation of the blue 0
print (mean_color)
print (mean_stdDev_color)



#load images function





# Test the model on live camera feed
vc = cv2.VideoCapture(1) 
while vc.isOpened():
    rval, frame = vc.read()    # read video frames again at each loop, as long as the stream is open
    
    # Predict the color and display it on the live feed
    img = cv2.resize(frame, (100, 100))
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([img_lab], [1, 2], None, [8, 8], [0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    prediction = knn_model.predict([hist])[0]
    cv2.putText(frame, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("stream", frame)
    
    key = cv2.waitKey(1)       # allows user intervention without stopping the stream (pause in ms)
    if key == 27:              # exit on ESC
        break
cv2.destroyWindow("stream")    # close image window upon exit
vc.release()


