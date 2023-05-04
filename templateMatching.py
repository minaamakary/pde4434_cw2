# Python program to illustrate
# template matching
import cv2
import numpy as np

# Read the main image
img_rgb = cv2.imread('./images/newred_1.JPG')

# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

#eroded = cv2.erode(thresh1, (3,3), iterations=1)

# Dilate the image
#dilated = cv2.dilate(eroded, (3,3), iterations=1)

#cv2.imshow("thresh",dilated)

contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

rect = cv2.minAreaRect(largest_contour)
box = cv2.boxPoints(rect)

box = np.int0(box)  #converting calcluated coordinates with decimal places in to integers

cv2.drawContours(img_rgb, [box], -1, (0, 255, 0), 5) #drawing box

x,y,w,h = cv2.boundingRect(largest_contour)

cropped_img = img_rgb[y:y+h,x:x+w]
cv2.imshow("Cropped",cropped_img)

edge = cv2.Canny(cropped_img,20,200)
cv2.imshow("canny",edge)

'''
#Loading the templates in to a list
templates = os.listdir('./templates)
scores = []
for t in templates:
	temp = cv2.imread('templates/'+t,0)
	result = cv2.matchTemplate(edge,t,cv2.TM_CCOEFF_MORMED)
    scores.append(result)
    
max_value = max(scores)
print(score.index(max_value))
'''



#cv2.imwrite("0-template.jpg",edge) # saving the image


#cards = []

print(len(contours))
'''
for contour in contours:
    area = cv2.contourArea(contour)
    x,y,w,h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    if area > 5000 and aspect_ratio > 0.5 and aspect_ratio < 1.5:
        cards.append(contour)
    cv2.drawContours(img_rgb, contour, -1, (0, 255, 0), 2)
'''
        

'''
# Read the template
template = cv2.imread('./images/template.png', 0)
#cv2.imshow("t",template)

# Store width and height of template in w and h
w, h = template.shape[::-1]

# Perform match operations.
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

# Specify a threshold
threshold = 0.3

# Store the coordinates of matched area in a numpy array
loc = np.where(res >= threshold)
print(loc)

# Draw a rectangle around the matched region.
#for pt in zip(*loc[::-1]):#
#	cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
'''

# Show the final image with the matched area.
cv2.imshow('Detected', img_rgb)
cv2.waitKey(0)

