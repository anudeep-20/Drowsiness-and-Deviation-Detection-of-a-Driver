import os
import cv2
import numpy as np
from PIL import Image

recog=cv2.createLBPHFaceRecognizer()
path='dataset'

def get_image_and_IDs(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    face=[]
    IDs=[]
    for imPath in imagePaths:
        faceimg=Image.open(imPath).convert('L')
        Np=np.array(faceimg,'uint8')
        ID=os.path.split(imPath)[-1].split('.')[1]
        ID = int(ID)
        face.append(Np)
        #print ID
        IDs.append(ID)
        cv2.imshow("trainer", Np)
        cv2.waitKey(10)
        
    return IDs,face

ids,face=get_image_and_IDs(path)
recog.train(face,np.array(ids))
recog.save('yml_files/trainedData.yml')
cv2.destroyAllWindows()
