##For Running this Python File :

##i)Press win+R
##ii)Type cmd and press Enter
##iii)Now type  cd <Location of the project folder in your PC> (In my case I have typed, cd C:\Users\*****\Documents\My_projects\highway_project )  and Press Enter.
##iv)Now type 	python Serial_Highway_project_CoOrdinates.py -dd eye_detector.dat 	and Press Enter.

from drawnow import drawnow
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse , imutils , dlib , cv2 , serial ,time , math
import winsound as win
import matplotlib.pyplot as plt

obd_data = serial.Serial("COM12" , 9600 ,timeout = 0.1)#Change the COM port of HC05 bluetooth module paired with OBD.
obd_data.flush()

recog = cv2.createLBPHFaceRecognizer();
recog.load("yml_files_main/trainedData.yml")

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def makeFig():

#If the graph to be plotted in sub-plot.(Uncomment Below code and Comment single plot code)
##    axarr[0].plot(time_1 , steer_angle , color = 'r' , linewidth = 1)
##    axarr[0].set_title("Steering Angle")
##    axarr[0].grid(True)
##    axarr[1].plot(time_1 , vehicle_speed , color = '#67C117' , linewidth = 1)
##    axarr[1].set_title("Vehicle Speed")
##    axarr[1].set_xlabel("Time")
##    axarr[1].grid(True)

#If the graph to be plotted in Single plot.
    plt.plot(time_1, steer_angle , color = 'r' , label = 'Steering Angle' , linewidth = 1)
    plt.plot(time_1, vehicle_speed , color = '#67C117' , label = 'Vehicle Speed' , linewidth = 1)
    plt.xlabel('Time')
    plt.grid(True)
    plt.legend(loc = 'upper center' , bbox_to_anchor = (0.5,1.1) , ncol = 4)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def check_steer(prev , pres):
    diff = abs(pres-prev)

    if diff <= 0.01:
        return 1
    else:
        return 0

ap = argparse.ArgumentParser()
ap.add_argument("-dd", "--drowsy_detect", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["drowsy_detect"])

(leS, leE) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reS, reE) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video
fileStream = True
vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

id_num  = prev = pres = i = area = 0
present = previous = mid = [0,0]

new_array = []
array = [None]*25

is_state = []
is_stat=[]

steer_angle = [None]*100
steer_angle[99] = 0
vehicle_speed = [None]*100

#time_ref=1410931895.045
time_ref = time.time()
time_1 = [None]*100

plt.ion() 
fig=plt.figure()

f , axarr = plt.subplots(2, sharex = True)

while True :

    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear = ( eye_aspect_ratio(shape[leS:leE]) + eye_aspect_ratio(shape[reS:reE]) ) / 2.0

        if (ear > 0.3):
            cv2.drawContours(frame, [cv2.convexHull(shape[leS:leE])], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(shape[reS:reE])], -1, (0, 255, 0), 1)
        elif (ear < 0.2):
            is_state.append("drowsy")
            cv2.drawContours(frame, [cv2.convexHull(shape[leS:leE])], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [cv2.convexHull(shape[reS:reE])], -1, (0, 0, 255), 1)

        if len(is_state) == 10:
            if is_state.count("drowsy") > 8:
                cv2.putText(frame, "DROWSY..!!", (150, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            is_state = []

        faces = face_detect.detectMultiScale(gray, 1.3, 5)
        #print id_num

        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            mid = [x+(w/2) , y+(h/2)]
        
            cv2.circle(frame , (int(mid[0]), int(mid[1])) , 2 , (0,0,255) , -1)

            present = mid
            distance = math.sqrt(((present[0]-previous[0])**2)+((present[1]-previous[1])**2))
            print distance

            if distance == 0 :
                is_stat.append(0)
            if distance <= 5 and distance != 0:
                is_stat.append(1)
            if distance >= 5:
                is_stat.append(2)

            id_num, conf = recog.predict(gray[y:y + h, x:x + w])

            #The Instructions for creating ID's is explained in the face_detect\Instructions.txt.
            if (id_num == 1):
                cv2.putText(frame, 'ID_1', (x + 50, y + 225), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)#In place of ID_1, You can put the name of ID 1 Person name.

            elif (id_num == 2):
                cv2.putText(frame, 'ID_2', (x + 50, y + 225), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)#In place of ID_2, You can put the name of ID 2 Person name.

        previous = mid

    if (obd_data.inWaiting() > 0):
        obd_json_data = obd_data.readline()
        obd_json_data = obd_json_data.strip('\n')
        obd_json_data = obd_json_data.strip('\r')

        obd_json_data = obd_json_data.replace('''{"name":''', '')
        obd_json_data = obd_json_data.replace("}",'')
        obd_json_data = obd_json_data.replace('''"value":''','')
        obd_json_data = obd_json_data.replace('''"timestamp":''','')
        obd_json_data = obd_json_data.replace('''"''','')
        obd_json_data = obd_json_data.replace(',','\n')
        #print obd_json_data

        new_array.append(obd_json_data.split('\n'))

        time_diff = abs(time.time()-time_ref)
        time_1.append(time_diff)
        time_1 = time_1[-100:]

        try:
            if new_array[i][0] == 'steering_wheel_angle' :
                prev = steer_angle[99]
                steer_angle.append(float(new_array[i][1]))
                steer_angle = steer_angle[-100:]
                pres = steer_angle[99]
            
            array.append(check_steer(prev,pres))

            if new_array[i][0] == 'vehicle_speed' :
                vehicle_speed.append(float(new_array[i][1]))
                vehicle_speed = vehicle_speed[-100:]

            i = i+1
            if len(new_array) >= 10:
                new_array = []
                i = 0

        except TypeError:
            continue
        
        if len(is_stat) >= 20 and len(array) >= 20:
            if ((is_stat.count(1) + is_stat.count(0) > len(is_stat)*0.8) or (is_stat.count(0) > len(is_stat)*0.6) and (array.count(1) >= len(array)*0.7)):
                print "You are deviated"
                time.sleep(3)
                cv2.putText(frame, "You are DEVIATED!!", (x + 50, y + 225), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (0, 0, 255), 2)
                win.Beep(1000, 500)
                time.sleep(0.25)

            array =[]
            print is_stat
            is_stat = []

        cv2.imshow("Driver_drowsy_system", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("h"):
            break

        drawnow(makeFig)
        plt.pause(0.01)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
