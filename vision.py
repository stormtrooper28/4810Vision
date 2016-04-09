import numpy as np
import cv2
import math

def coverage_area_ratio(par_area, box_area):
	return par_area/box_area
	
def aspect_ratio(width, height):
	return width/height
	
def get_score(score, ideal):
	diff = abs(score - ideal)
	error_pct = ((diff / ideal) * 100)
	return math.floor(100 - error_pct)
	
def determine_target(cov_score, asp_score):
	average_score = (cov_score + asp_score)/2
	print("Average score: " + str(average_score))
	if average_score >= 70:
		return True
	else:
		return False
	
def vision():
	orig = cv2.imread('samplepics/pic2.jpg', cv2.IMREAD_COLOR)
	hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
	
	
	lower_red = np.array([20, 0, 225], dtype = np.uint8) 
	upper_red = np.array([40, 10, 255], dtype = np.uint8)
	
	mask = cv2.inRange(hsv, lower_red, upper_red)
	ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
	
	
	_, contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	areas = [cv2.contourArea(c) for c in contours]
	max_index = np.argmax(areas)

	cnt = contours[max_index]
	x, y, w, h = cv2.boundingRect(cnt)
	cv2.rectangle(orig, (x,y), (x+w, y+h), (0, 0, 0), 2)
	area = w * h

	cov_sc = get_score(coverage_area_ratio(cv2.contourArea(contours[max_index]), area), (1/3))
	asp_sc = get_score(aspect_ratio(w, h), (5/3))
	
	print("Coverage score: " + str(cov_sc))
	print("Aspect score: " + str(asp_sc))
	print("Target detected? " + str(determine_target(cov_sc, asp_sc)))


	cv2.imshow('original', orig)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

vision()

