import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
for i in range(1, 10):
	img = cv.imread('sample' + str(i) + '.jpg')
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	j = 1
	for (x,y,w,h) in faces:
	    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = img[y:y+h, x:x+w]
	    cv.imwrite('preprocessed/img'+str(j)+'.jpg', roi_color)
	    j += 1
'''cv.imshow('img',roi_color)
cv.waitKey(0)
cv.destroyAllWindows()'''