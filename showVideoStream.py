from signal import signal
from socket import socket
from tkinter import Label
from unittest import result
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
from imutils.video import VideoStream
import cv2 
import imutils
import datetime
import os
import time
import socket
from playsound import playsound


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

def connectToserver():
    #----- A simple TCP client program in Python using send() function -----

	# Create a client socket

	clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	# Connect to the server

	clientSocket.connect(("127.0.0.1", 1234))

	# Send data to server

	data = "Mask Detected"

	clientSocket.send(data.encode())

	# Receive data from server
	global dataFromServer

	dataFromServer = clientSocket.recv(1024)

	# Print to the console

	print(dataFromServer.decode())

	return dataFromServer

def TransmitData():
    
    global dataFromClient
    
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
    
    serverSocket.bind(("127.0.0.1", 1234))
    
    serverSocket.listen(5);
    
    while True:
        
        (clientConnected, clientAddress) = serverSocket.accept()
        
        print("Accepted a connection request from %s:%s"%(clientAddress[0], clientAddress[1]))
        
        clientConnected.sendto("đã nhận diện đeo khẩu trang".encode('utf-8'), ("127.0.0.1", 8080))
        
        dataFromClient = clientConnected.recv(1024)
        
        if dataFromClient == b"Hoan thanh mot chu trinh" or b"Mask Detected":
            print(dataFromClient.decode())
            print(clientConnected)
            print(clientAddress)
            
            return dataFromClient
        else: 
            clientConnected.close()


    
face_detector = cv2.CascadeClassifier('new.xml')

while True:

    signal = int(input())
    
    if signal == 1:
        
        print("Camera is opening please wait a second..")
        
        cam = cv2.VideoCapture(0)
        
        ret, newframe = cam.read()
        
        cv2.imshow('frame', newframe)
        
        key = cv2.waitKey(1)

        faces = face_detector.detectMultiScale(newframe, 1.3, 5)
        
        print(faces)
        
        suffix  = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        
        path = '/Users/macbook/Desktop/Mask_Detect/test_img'
        
        filename = ".".join([suffix, 'png'])
        
        if len(faces) == 0:
            
            print('no one in sigth of camera')
        else:
            
            cv2.imwrite(os.path.join(path, filename ), img=newframe)
            newimage = cv2.imread("/Users/macbook/Desktop/Mask_Detect/test_img/" + filename )
            if len(newimage) != 0: 
                print('yes') 
                frame = imutils.resize(newimage, width=400)
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
                for (box, pred) in zip(locs, preds):
                    (mask, withoutMask) = pred
                    if mask > withoutMask:
                        label = "Masked"
                        print("Da deo khau trang dung cach")
                        playsound ("sound/notification.mp3")
                        TransmitData()
                        print(dataFromClient)
                        if dataFromClient == b'Hoan thanh mot chu trinh':
                            print(dataFromClient)
                        else:
                            print('khong nhan duoc tin hieu tu server')
                    else:
                        label = "No Mask"
                        print('No Mask')
                        print('Thay doi vi tri hoac chinh lai khau trang cua ban')
                        playsound ("sound/remind.mp3")
                        # time.sleep(5)
            else: 
                print('no') 


cv2.destroyAllWindows()
vs.stop()