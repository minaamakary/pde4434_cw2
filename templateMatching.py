# Python program to illustrate
# template matching
import cv2
import numpy as np

# Read the main image
img_rgb = cv2.imread('./images/main.jpg')

# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh",thresh1)

# Read the template
template = cv2.imread('./images/template.jpg', 0)
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
for pt in zip(*loc[::-1]):
	cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

# Show the final image with the matched area.
#cv2.imshow('Detected', img_rgb)
cv2.waitKey(0)

