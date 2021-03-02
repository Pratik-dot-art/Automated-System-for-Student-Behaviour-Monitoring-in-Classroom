import cv2
import os
import time
###################
name=input("Enter your name:")
P_dir=(r'C:\Users\Pratik\OneDrive\Desktop\attnan\project\dataset')
path = os.path.join(P_dir, name)
os.mkdir(path)
##################
videoCaptureObject = cv2.VideoCapture(0)
folder_path = (r'C:\Users\Pratik\OneDrive\Desktop\attnan\project\dataset\\'+name)
test = os.listdir(folder_path)

face_classifier =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier  =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
result = True

for i in range(20):
    ret,frame = videoCaptureObject.read()
    cv2.imwrite(os.path.join(folder_path, "NewPicture"+str(i)+".jpg"),frame)
   
   
    image =cv2.imread(os.path.join(folder_path, "NewPicture"+str(i)+".jpg"))
    #image =cv2.imread("test2.jpg")
    gray =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',gray)
    faces =face_classifier.detectMultiScale(gray,1.5,5)
   
    if faces is ():
        print('No faces detected!')
        continue
    # faces_found
    #j=0
    for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = image[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (256, 256))
            #i += 1
    #        cv2.imwrite("image"+str(i)+".jpg", lastimg)
            cv2.imwrite(os.path.join(folder_path,"image"+str(i)+".jpg"), lastimg)
            #j=j+1
    images="NewPicture"+str(i)+".jpg"
    os.remove(os.path.join(folder_path, images))
    

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
    #         cv2.waitKey(0)
     """  
        #cv2.imshow("faces&eyes"+str(i),image)
    
    #time.sleep(5)
   
    #videoCaptureObject.release()
    #cv2.destroyAllWindows()
videoCaptureObject.release()
