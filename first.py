import cv2
import os
import time
import numpy as np
import pickle
import face_recognition
import argparse
#from mtcnn.mtcnn import MTCNN

from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
#from work import r
#import docopt 
#from sklearn import svm

print("[INFO] loading encodings...")
data = pickle.loads(open('encodings.pickle', "rb").read())
known_face_names=data["names"]

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

####################################################
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

def activity(frame):
    #frame=cv2.imread("io.jpg")
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)#converting to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < thresh:
            return "NOT active"
            #print("person is sleeping")
        else:
            return "Active"
            #print("person is NOT sleeping")
    

####################################################









videoCaptureObject = cv2.VideoCapture(0)
folder_path = (r'C:\Users\Pratik\OneDrive\Desktop\attnan\project')
test = os.listdir(folder_path)
face_classifier =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier  =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') 
result = True
path=(r'C:\Users\Pratik\OneDrive\Desktop\attnan\project\cropimg')
names=[]
active={}
o=0
while(result):
    ret,frame = videoCaptureObject.read()
    cv2.imwrite("NewPicture"+str(o)+".jpg",frame)
    
   
    image =cv2.imread("NewPicture"+str(o)+".jpg")
    #image =cv2.imread("io"+str(h)+".jpg")
    gray =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',gray)
    faces =face_classifier.detectMultiScale(gray,1.5,5)
    #faces = detector.detect_faces(gray,1.5,5)
    if faces is ():
        print('No faces detected!')
        
    # faces_found
    j=0
    for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = image[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (128, 128))
            j += 1
            img=cv2.imwrite(os.path.join(path , "image"+str(j)+".jpg"), lastimg)
            #img =cv2.imread("C:\Users\Pratik\OneDrive\Desktop\attnan\project\cropimg\image"+str(j)+".jpg")
            boxes = face_recognition.face_locations(lastimg,model=args["detection_method"])
            encodings = face_recognition.face_encodings(lastimg, boxes)
            
            
            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"

                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                   
                #print(name)
                if name not in names:
                    names.append(name)
                active[name]=activity(lastimg)
                #print(names)
    #        cv2.imwrite("image"+str(i)+".jpg", lastimg)
            #cv2.imwrite("image"+str(j)+".jpg", lastimg)
            #j=j+1
    
    print(names)
    print(active)
    images="NewPicture"+str(o)+".jpg"
    os.remove(os.path.join(folder_path, images))
    o+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""    for (x,y,w,h) in faces:
        
        # draw_rectangle_around_face     
        cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
    #     cv2.imshow('faces',image)
    #     cv2.waitKey(0)
        
        # cropping_face_only
        roi_color=image[y:y+h,x:x+w]
        roi_gray=gray[y:y+h,x:x+w]
        
        # eyes_detection
        eyes=eye_classifier.detectMultiScale(roi_gray)
        
        # drawing_eyes_one_by_one
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
            cv2.imshow('face&eyes',image)
    #        cv2.waitKey(0)
        
        #cv2.imshow("faces&eyes"+str(i),image)
        #images="NewPicture"+str(i-1)+".jpg"
        #os.remove(os.path.join(folder_path, images))
    #time.sleep(5)
    
    #videoCaptureObject.release()
    #cv2.destroyAllWindows()
#videoCaptureObject.release()

"""

#print(names)
#print(active)
