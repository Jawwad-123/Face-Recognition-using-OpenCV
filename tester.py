import cv2
import os
import numpy as np
import faceRecognition as fr

test_img=cv2.imread('#') # just replace the hash with the path of the test image in your test images directory and try giving full path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

#for (x,y,w,h) in faces_detected:
#	cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)
#
#resized_img=cv2.resize(test_img,(1000,700))
#cv2.imshow("face detection tutorial",resized_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows	

faces,faceID=fr.labels_for_training_data('/FaceRecognition/trainingimages') # try giving full path if gives error
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save('trainingData.yml')
name={0:"shahrukhkhan",1:"Your Name"}

for face in faces_detected:
	(x,y,w,h)=face
	roi_gray=gray_img[y:y+h,x:x+h]
	label,confidence=face_recognizer.predict(roi_gray)
	print("confidence:",confidence)
	print("label:",label)
	fr.draw_rect(test_img,face)
	predicted_name=name[label]
	fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(700,700))
cv2.imshow("face detection tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
