from tkinter import *
import tkinter
import winsound
#which allows u to perform the image processing and computer vision
import cv2
# Numpy for array related functions
import numpy as np
#Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils

#the total distance between the two like eyes opening and closing and mouth opening and closing
from scipy.spatial import distance as dist 

main = tkinter.Tk()
main.title("Microsleep Detection of a Driver") 
main.geometry("800x600")
#used to calculate one of the eight different matrix norms or one of the vector norms.
#a norm means a measure of the size of a matrix or vector
def compute(ptA,ptB):
	dist = np.linalg.norm(ptA - ptB)
	return dist

def blinked(a,b,c,d,e,f):
	up = compute(b,d) + compute(c,e)
	down = compute(a,f)
	ratio = up/(2.0*down)

	#Checking if it is blinked
	if(ratio>0.25):
		return 2
	elif(ratio>0.21 and ratio<=0.25):
		return 1
	else:
		return 0

def MOR(drivermouth):
 # compute the euclidean distances between the horizontal 
 point = dist.euclidean(drivermouth[0], drivermouth[6]) 
 # compute the euclidean distances between the vertical 
 point1 = dist.euclidean(drivermouth[2], drivermouth[10])
 point2 = dist.euclidean(drivermouth[4], drivermouth[8]) 
 # taking average
 Ypoint = (point1+point2)/2.0 
 # compute mouth aspect ratio
 mouth_aspect_ratio = Ypoint/point 
 return mouth_aspect_ratio	
	
def startMonitoring():
 pathlabel.config(text="Webcam Connected Successfully") 
#Initializing the camera and taking the instance
 cap = cv2.VideoCapture(0)

#Initializing the face detector and landmark detector
 detector = dlib.get_frontal_face_detector()
 predictor = dlib.shape_predictor('C:\\Users\\srilatha\\Downloads\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat')

#status marking for current state
 sleep = 0
 drowsy = 0
 active = 0
 MOU_AR_THRESH = 0.75
 yawnStatus = False 
 yawns = 0
 status=""
 color=(0,0,0)
 (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
 while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_yawn_status = yawnStatus
    faces = detector(gray)
    face_frame = frame.copy()
    #detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        #face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        #The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36],landmarks[37], 
        	landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
        	landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        mouth = landmarks[mStart:mEnd]
        mouEAR = MOR(mouth)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1) 
        #Now judge what to do for the eye blinks
        yawns=0;
        if mouEAR > MOU_AR_THRESH:
          cv2.putText(frame, "Yawning, DROWSINESS ALERT! ", (10,60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
          yawnStatus = True
          
          output_text = str(yawns + 1)
          cv2.putText(frame, output_text, (10,150),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
          
        else:
          yawnStatus = False

        if(left_blink==0 or right_blink==0):
        	sleep+=1
        	drowsy=0
        	active=0
        	if(sleep>15):
        		status="SLEEPING !!!"
        		color = (255,0,0)
        		winsound.Beep(2000, 1500)
        		

        elif(left_blink==1 or right_blink==1):
        	sleep=0
        	active=0
        	drowsy+=1
        	if(drowsy>3):
        		status="Drowsy !"
        		winsound.Beep(2000, 1500)
        		color = (0,0,255)
                            
        else:
        	drowsy=0
        	sleep=0
        	active+=1
        	if(active>6):
        		status="Active :)"
        		color = (0,255,0)
        	
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)
     
        for n in range(0, 68):
        	(x,y) = landmarks[n]
        	cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
      	break
font = ('times', 16, 'bold')
title = Label(main, text='Microsleep detection of a driver using Machine',anchor=W, justify=LEFT)
title.config(bg='black', fg='white') 
title.config(font=font) 
title.config(height=3, width=120) 
title.place(x=0,y=0)
font1 = ('times', 14, 'bold')
upload = Button(main, text="Start behaviour monitoring using webcam", 
command=startMonitoring)
upload.place(x=200,y=300) 
upload.config(font=font1)
pathlabel = Label(main) 
pathlabel.config(bg='DarkOrange1', fg='white') 
pathlabel.config(font=font1) 
pathlabel.place(x=50,y=250) 
main.config(bg='chocolate1')
main.mainloop()
