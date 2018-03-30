import cv2
import numpy as np
import time

face_detect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
recog=cv2.createLBPHFaceRecognizer();
recog.load("yml_files/trainedData.yml")
id_num=0
font=cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id_num,conf=recog.predict(gray[y:y+h,x:x+w])
        print id_num 
        if(id_num == 1):
            cv2.putText(img,'ID_1',(x+50,y+225),font,1,(255,0,0),2)#Change ID_1 with the name of its respective Person.
        if (id_num == 2):
            cv2.putText(img,'ID_2',(x+50,y+225),font,1,(255,0,0),2)#Change ID_2 with the name of its respective Person.
        if (id_num == 3):
            cv2.putText(img,'ID_3',(x+50,y+225),font,1,(255,0,0),2)#Change ID_3 with the name of its respective Person.

    cv2.imshow("Face detect", img)
    if(cv2.waitKey(1)==ord('h')):
        break
    
cam.release()
cv2.destroyAllWindows()
