import numpy as np
import cv2
import math
import time

#Returns the coverage area ratio, defined by the ratio of the area of the particle to the area of the bounding box drawn around  the circle
def coverage_area_ratio(par_area, box_area):
	return par_area/box_area
	
#Returns the aspect ratio, defined  by the ratio of the height of the bounding box to the width of the bounding box	
def aspect_ratio(width, height):
	return width/height
	
#Returns a score, defined by 100 - (percent error * 100)
def get_score(score, ideal):
	diff = abs(score - ideal)
	error_pct = ((diff / ideal) * 100)
	return math.floor(100 - error_pct)
	
#Returns a boolean determining a target, a predefined goal is set
def determine_target(cov_score, asp_score):
	if cov_score >= 70 & asp_score >= 70:
		return True
	else:
		return False
	
#The main vision algorithm
def vision():
	#capture a video
	cap = cv2.VideoCapture(0)
	cov_sc = 0
	asp_sc = 0
	
	while True:
		#Do the processing
		#orig = cv2.imread('samplepics/pic2.jpg', cv2.IMREAD_COLOR)
		ret, orig = cap.read()
		hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
		
		#numpy arrays to store bgr values to mask
		lower_red = np.array([50, 0, 225], dtype = np.uint8) 
		upper_red = np.array([70, 10, 255], dtype = np.uint8)
		
		#masking and thresholding the image
		mask = cv2.inRange(hsv, lower_red, upper_red)
		ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
		
		#find the contours of the image to find a boundary
		_, contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		areas = [cv2.contourArea(c) for c in contours]
		if not areas:
			continue
		else:
			max_index = np.argmax(areas)

			cnt = contours[max_index]
			x, y, w, h = cv2.boundingRect(cnt)
			cv2.rectangle(orig, (x,y), (x+w, y+h), (0, 0, 0), 2)
			area = w * h

			cov_sc = get_score(coverage_area_ratio(cv2.contourArea(contours[max_index]), area), (1/3))
			asp_sc = get_score(aspect_ratio(w, h), (5/3))
		
		if determine_target(cov_sc, asp_sc):
			cv2.imwrite('is_target.png', orig)
			print("Coverage score: " + str(cov_sc))
			print("Aspect score: " + str(asp_sc))
			break
			

		cv2.imshow('original', orig)
		#cv2.imshow('hsv', hsv)
		if cv2.waitKey(1) & 0xff == ord('q'):
			break
			
		time.sleep(0.2)
	print("Target detected? " + str(determine_target(cov_sc, asp_sc)))
	cap.release()
	cv2.destroyAllWindows()

vision()

