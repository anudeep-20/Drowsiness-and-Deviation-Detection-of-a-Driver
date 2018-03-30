from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse , time , imutils , dlib , cv2 , math
import winsound as win
import matplotlib.pyplot as plt
from drawnow import drawnow

recog = cv2.createLBPHFaceRecognizer();
recog.load("yml_files_main/trainedData.yml")

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

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

is_state = [None]*10
is_stat=[None]*40
present = previous = mid = [0,0]
id_num = i = area = 0


while True :

    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear = ( eye_aspect_ratio(shape[leS:leE]) + eye_aspect_ratio(shape[reS:reE]) ) / 2.0

        if ear == None:
            print "hey"

        if (ear > 0.3):
            cv2.drawContours(frame, [cv2.convexHull(shape[leS:leE])], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(shape[reS:reE])], -1, (0, 255, 0), 1)
        elif (ear < 0.2):
            is_state.append("drowsy")
            cv2.drawContours(frame, [cv2.convexHull(shape[leS:leE])], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [cv2.convexHull(shape[reS:reE])], -1, (0, 0, 255), 1)
         

        if len(is_state) == 30:
            if is_state.count("drowsy") > 25:
                cv2.putText(frame, "DROWSY..!!", (150, 60), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2)
                print "You are feeling DROWSY"

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

            if (id_num == 1):
                cv2.putText(frame, 'Anudeep', (x + 50, y + 225), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

            elif (id_num == 2):
                cv2.putText(frame, 'Kowshik', (x + 50, y + 225), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

        if len(is_stat) >= 40:
            if (is_stat.count(1) + is_stat.count(0) > 25) or (is_stat.count(0) > 12):
                print "You are deviated"
                time.sleep(3)
                cv2.putText(frame, "You are DEVIATED!!", (x + 50, y + 225), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (0, 0, 255), 2)
                win.Beep(1000, 500)
                time.sleep(0.25)

            print is_stat
            is_stat = []

        previous = mid

    cv2.imshow("Driver_drowsy_system", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `h` key is pressed, break from the loop
    if key == ord("h"):
            break

cv2.destroyAllWindows()
vs.stop()
        
