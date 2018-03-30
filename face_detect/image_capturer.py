import cv2
import numpy as np
import time

face_detect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ID=raw_input("Please enter the ID(Type 1 for 1st User) : ")
cam=cv2.VideoCapture(0)
num_faces=0
while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        num_faces+=1
        cv2.imwrite("dataset/User."+str(ID)+"."+str(num_faces)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(75)
    cv2.imshow("Face detect",img)
    if(num_faces==200):
        break
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("h"):
            break
cam.release()
cv2.destroyAllWindows()
