import numpy as np
import cv2

def vision():
	orig = cv2.imread('samplepics/pic5.jpg', cv2.IMREAD_COLOR)
	hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
	
	
	lower_red = np.array([20, 0, 225], dtype = np.uint8) 
	upper_red = np.array([40, 10, 255], dtype = np.uint8)
	
	mask = cv2.inRange(hsv, lower_red, upper_red)
	ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
	
	im2, contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(mask, contours, 3, (0, 255, 0), 3)

	cv2.imshow('original', orig)
	cv2.imshow('hsv', hsv)
	cv2.imshow('mask', mask)
	cv2.imshow('threshold', thresh)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

vision()

