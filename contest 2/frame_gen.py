import numpy as np 
import cv2 
import time
import math
cap = cv2.VideoCapture('Test Tom and Jerry.mp4')
frameRate = cap.get(5)
i=0
count = 0
while(cap.isOpened()):
	cap.set(cv2.CAP_PROP_POS_FRAMES, count)
	count += math.floor(frameRate)
	print(count)
	ret, frame = cap.read()
	if (ret != True) or cv2.waitKey(1) & 0xFF == ord('q') or count>=8912:
		break
	cv2.imwrite('test_frames/'+"test"+str(i)+".jpg",frame)	
	i+=1
