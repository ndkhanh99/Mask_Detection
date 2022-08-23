from signal import signal
from tkinter import Label
from unittest import result
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
from imutils.video import VideoStream
import time
import datetime
import cv2
from cv2 import (VideoCapture)
import os
import socket 
from threading import Thread
import threading
import ssl
import certifi
from playsound import playsound


ssl._create_default_https_context = ssl._create_unverified_context

# Load modal phat hien guong mat trong khung anh 
prototxtPath = r"/Users/macbook/Desktop/face_d-tction_project/face_detector/deploy.prototxt"
weightsPath = r"/Users/macbook/Desktop/face_d-tction_project/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load modal nhan dien deo khau trang da duoc training vao giai thuat
maskNet = load_model("mask_detector_2.model")


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


label = []

suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

s = socket.socket()

s.bind(('127.0.0.1', 8081))

s.listen(0)

def connectServer():
    
    while True:
        global label
        
        results = []
        
        client, addr = s.accept()
        
        while True:
            content =client.recv(32)
            if len(content) == 0:
                break
            else:
                print(content)
                
        client.send(bytes(label,  "utf-8" ))
        
        print('closing connection')
        
        client.close()
        
        return (results)

    

print("[INFO] starting video stream...")
# khoi dong camera 
vs = cv2.VideoCapture(0)
    
while True:
    
	results = []
 
	_, frame = vs.read()
 
	cv2.imshow('frame', frame)
 
	key = cv2.waitKey(1)
	
	face_detector = cv2.CascadeClassifier('new.xml')
	
	print('camera is opening')
 
	video = 1
	
	if (vs.isOpened()):
		
		# _, frame = vs.read()

		# cv2.imshow('frame', frame)

		# key = cv2.waitKey(1)
  
		# if key == ord('q'):
		# 	break
		
		faces = face_detector.detectMultiScale(frame, 1.3, 5)
		
		print(faces)
		
		suffix  = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
		
		path = '/Users/macbook/Desktop/Mask_Detect/images'
		
		filename = ".".join([suffix, 'png'])
		
		if type(faces) == tuple :
			
			print('None of people in camera')
			
			time.sleep(2)
			
		else:
			
			faces = face_detector.detectMultiScale(frame, 1.3, 5)
			
			cv2.imwrite(os.path.join(path, filename ), img=frame)
			
			newimage = cv2.imread("/Users/macbook/Desktop/Mask_Detect/images/" + filename )
			
			frame = imutils.resize(newimage, width=400)
			
			(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
			
			for (box, pred) in zip(locs, preds):
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred
				if mask > withoutMask:
					label = "Masked"
					color = (255, 0, 0)
					print("Da deo khau trang dung cach")
					playsound ("sound/notification.mp3")
					signal = int(input())
					if signal == 1:
						# cv2.imshow("Frame", frame)
						# key = cv2.waitKey(1) & 0xFF
						break
				else:
					label = "No Mask"
					color = (0, 0, 255)
					print('No Mask')
					print('Thay doi vi tri hoac chinh lai khau trang cua ban')
					playsound ("sound/remind.mp3")
					time.sleep(5)
					# cv2.imshow("Frame", frame)
					# key = cv2.waitKey(1) & 0xFF
					break
 
cv2.destroyAllWindows()
vs.stop()

# s = socket.socket()

# s.bind(('127.0.0.1', 8081))

# s.listen(0)

# def connectServer():
    
#     while True:
#         global label
        
#         results = []
        
#         client, addr = s.accept()
        
#         while True:
#             content =client.recv(32)
#             if len(content) == 0:
#                 break
#             else:
#                 print(content)
                
#         client.send(bytes(label,  "utf-8" ))
        
#         print('closing connection')
        
#         client.close()
        
#         return (results)
    
# # def notify():
    
# #     playsound('/Users/macbook/Desktop/face_d-tction_project/notification.mp3')
    
# #     return print('play sound success')
    	

# try:
# 	t = time.time()
# 	t1 = threading.Thread(target=theadProcess)
# 	t2 = threading.Thread(target=connectServer)
# 	t1.start()
# 	t2.start()
# 	t1.join()
# 	t2.join()
# 	print ("done in ", time.time()- t)
 
# except:
# 	print ("error")



# # loop over the frames from the video stream

# 	# show the output frame


# #export signal

# # do a bit of cleanup


	
