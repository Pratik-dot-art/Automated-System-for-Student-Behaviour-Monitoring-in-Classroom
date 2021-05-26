import cv2
import os
import time
import numpy as np
import pickle
import face_recognition
import argparse
import sys
from imutils import paths
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
from tkinter import *
import time
from datetime import date 
from tkinter import messagebox 
from PIL import ImageTk,Image

sys.setrecursionlimit(10**8)

Ctime=""
Sname=""
names=[]
#"Pratik Durukkar":0,"yogesh":0,"shravni":0,"Siddharth":0,"swaranjali":0
active={}
result=True
o=0
print("[INFO] loading encodings...")
data = pickle.loads(open('encodings.pickle', "rb").read())
known_face_names=data["names"]
argp = argparse.ArgumentParser()
argp.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face model Present: `hog` or `cnn`")
args = vars(argp.parse_args())
##########################################################################################
threshold = 0.25

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

##################################################################
videoCaptureObject = cv2.VideoCapture(0)
folder_path = (r'C:\Users\Pratik\OneDrive\Desktop\attnan\project')
test = os.listdir(folder_path)
face_classifier =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier  =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') 
	#result = True
path=(r'C:\Users\Pratik\OneDrive\Desktop\attnan\project\cropimg')
Tpath=(r'C:\Users\Pratik\OneDrive\Desktop\attnan\project\gui')

root = Tk()
root.geometry("1020x900")
lb1=Label(root,text = 'Input Video :',font = ('Lucida Calligraphy',10))
lb=Label(root,text = 'Enter Student Name :',font = ('Lucida Calligraphy',10),pady = 10)
En=Entry(root,bd = 5,font = ('',10))#Text box for reg

########################EAR METHOD FOR EYE GAZLING##############################
def eye_aspect_ratio(eye):
    l1 = distance.euclidean(eye[1], eye[5])
    l2 = distance.euclidean(eye[2], eye[4])
    l3 = distance.euclidean(eye[0], eye[3])
    ear = (l1 + l2) / (2.0 * l3)
    return ear


def activity(frame):
    #frame=cv2.imread("io.jpg")
    frames = imutils.resize(frame, width=460)
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    subs = detect(gray, 0)
    for subject in subs:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        LEye = shape[lStart:lEnd]
        REye = shape[rStart:rEnd]
        LEAR = eye_aspect_ratio(LEye)
        REAR = eye_aspect_ratio(REye)
        ear = (LEAR + REAR)/2
        if ear < threshold:
            return 0 
        else:
            return 1
    return 1

################################################################################################################################################################################################

def stop():
	global result
	result=False
	global videoCaptureObject
	videoCaptureObject.release()
	canvas.create_rectangle(0,0,720,500,outline="black",fill="black")

################ATTANDANCE METHOD################################
def attd():
	now=date.today()
	act=txt.get('1.0',END)
	count=len(names)
	c=1
	try:
		Attandance=open(str(now)+" "+str(Ctime)+".txt",'a')
		Attandance.write("Total Number of Students Present is : %d \n"%count)
		Attandance.write("\nPresent Student :\n")
		for name in names:
			Attandance.write("\t%d. "%c)
			Attandance.write("%s\n"%name)
			c+1
		#print("Success")
		messagebox.showinfo("showinfo", "Attandance Taken")
		Attandance.write("\nStudent activity's :\n")
		Attandance.write("%s\n"%act)
		Attandance.close()
	except:
		messagebox.showerror("error", "Something went wrong")
		#print("error")
##############################FACE DETECTION AND RECOGNITION##############################
def start():
	txt.delete("1.0","end")
	global videoCaptureObject
	global result
	global Ctime
	Ctime=time.strftime("%H_%M_%S",time.localtime())
	result=True
	if not(videoCaptureObject.isOpened()):
		videoCaptureObject=cv2.VideoCapture(0)
	strt()

def strt():
	global o
	if result:
		ret,frame = videoCaptureObject.read()
		cv2image   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces =face_classifier.detectMultiScale(gray,1.5,5)
		for (x,y,w,h) in faces:
			cv2.rectangle(cv2image,(x,y),(x+w,y+h),(0,0,255),2)

		img   = Image.fromarray(cv2image).resize((720, 500))
		img = ImageTk.PhotoImage(image = img)
		canvas.create_image(0, 0, anchor=NW, image=img)
		if faces == ():
			print("no one")
		j=0

		for (a, b, c, d) in faces:
			r = max(c, d) / 2
			cenx = a + c / 2
			ceny = b + d / 2
			nx = int(cenx - r)
			ny = int(ceny - r)
			nr = int(r * 2)
			faceimg = cv2image[ny:ny+nr, nx:nx+nr]
			lastimg = cv2.resize(faceimg, (128, 128))
			j += 1
			#img=cv2.imwrite(os.path.join(path , "image"+str(j)+".jpg"), lastimg)
			boxes = face_recognition.face_locations(lastimg,model=args["detection_method"])
			encodings = face_recognition.face_encodings(lastimg, boxes)
			for encoding in encodings:
				matches = face_recognition.compare_faces(data["encodings"], encoding)
				name = "notknown"
				if True in matches:
					Match_Id = [i for (i, b) in enumerate(matches) if b]
					counts = {}
					for i in Match_Id:
						name = data["names"][i]
						counts[name] = counts.get(name, 0) + 1
					name = max(counts, key=counts.get)
					if name not in names:
						names.append(name)
						active[name]=0
					#st=activity(lastimg)
					if activity(lastimg):
						active[name]=0
					else:
						active[name]+=1
						#print()
						if active[name]==5:
							#print("%s is sleeping"%name)
							res=str(name)+" is Not Active\n"
							txt.insert(END,res)
							active[name]=0



				"""	if st==1:
						active[name]=0
					if st==0:
						active[name]+=1
						if active[name]==5:
							#print("%s is sleeping"%name)
							res=str(name)+" is sleeping\n"
							txt.insert(END,res)
							active[name]=0"""



		print(active)
		o+=1
		root.after(100,strt)
		root.mainloop()
####################################################################################################
def clo():
	root.destroy()

##########################ENCODEING METHOD FOR DATASET#############################################
def encode():
	
	argp = argparse.ArgumentParser()
	argp.add_argument("-i", "--dataset", default="dataset", 
		help="path of faces and images")
	argp.add_argument("-e", "--encodings", default="encodings.pickle",
		help="path of serialized encoding")
	argp.add_argument("-d", "--detection-method", type=str, default="cnn",
		help="face model Present: `hog` or `cnn`")
	args = vars(argp.parse_args())


	print("[INFO] loading faces...")
	imagePaths = list(paths.list_images(args["dataset"]))


	Savedencode = []
	StudentNames = []

	
	for (i, imagePath) in enumerate(imagePaths):
		
		print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]
		image = cv2.imread(imagePath)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		boxes = face_recognition.face_locations(rgb,
			model=args["detection_method"])

		
		encodings = face_recognition.face_encodings(rgb, boxes)

		
		for encoding in encodings:
			Savedencode.append(encoding)
			StudentNames.append(name)

	
	print("[INFO] serializing encodings of faces...")
	data = {"encodings": Savedencode, "names": StudentNames}
	f = open(args["encodings"], "wb")
	f.write(pickle.dumps(data))
	f.close()
##################################METHOD FOR REGESTATION OF STUDENT##################################################################
def reg():

	root.geometry("1500x900")
	Bu=Button(root,text = ' Register ',bd = 2 ,font = ('',15),padx=5,pady=5,command=reg2)
	Bu2=Button(root,text = 'End Registration ',bd = 2 ,font = ('',15),padx=5,pady=5,command=lambda:[lb.pack_forget(),En.pack_forget(),fr3.pack_forget(),fr2.pack_forget(),fr1.pack_forget(),Bu.pack_forget(),Bu2.pack_forget(),root.geometry("1020x900")])
	fr1=Frame(root,height =10)
	fr1.pack(side=BOTTOM)
	Bu2.pack(side=BOTTOM)
	fr2=Frame(root,height =10)
	fr2.pack(side=BOTTOM)
	Bu.pack(side=BOTTOM)
	fr3=Frame(root,height =10)
	fr3.pack(side=BOTTOM)
	En.pack(side=BOTTOM)
	lb.pack(side=BOTTOM)


def reg2():
	Sname=En.get()
	P_dir=(r'C:\Users\Pratik\OneDrive\Desktop\attnan\project\dataset')
	path = os.path.join(P_dir, Sname)
	if os.path.isdir(path):
		messagebox.showwarning("informatio","Person Already Exists")
		#print("Dir pesrrent")
		return 0
	else:	 
		os.mkdir(path)
	##################
	videoCaptureObject = cv2.VideoCapture(0)
	#folder_path = (r'C:\Users\Pratik\OneDrive\Desktop\attnan\project\dataset\\'+Sname)
	test = os.listdir(path)

	face_classifier =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

	for i in range(20):
	    ret,frame = videoCaptureObject.read()
	    cv2.imwrite(os.path.join(path, "NewPicture"+str(i)+".jpg"),frame)
	    image =cv2.imread(os.path.join(path, "NewPicture"+str(i)+".jpg"))
	    cv2.imshow('gray',image)
	    gray =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	    faces =face_classifier.detectMultiScale(gray,1.5,5)
	   
	    if faces == ():
	        print('faces Not detected!')
	        images="NewPicture"+str(i)+".jpg"
	        os.remove(os.path.join(path, images))
	        continue
	    for (a, y, c, h) in faces:
	            rr = max(c, h) / 2
	            cenx = a + w / 2
	            ceny = y + h / 2
	            nx = int(cenx - rr)
	            ny = int(ceny - rr)
	            nr = int(r * 2)

	            faceimg = image[ny:ny+nr, nx:nx+nr]
	            lastimg = cv2.resize(faceimg, (256, 256))
	            cv2.imwrite(os.path.join(path,"image"+str(i)+".jpg"), lastimg)
	    images="NewPicture"+str(i)+".jpg"
	    os.remove(os.path.join(path, images))
	videoCaptureObject.release()

############################################################################################################################################
lb1.pack(side=TOP)
canvas = Canvas(root ,bg="black",width = 720, height = 500)  
canvas.pack(side=TOP)

txt=Text(root,fg="red",font=("Arial",15),height=5,width=100)
txt.pack(side=TOP)
txt.insert(END,"**Students activity will be here**")

fr=Frame(root,width =100)
fr.pack(side=LEFT)
Sbutton = Button(root,padx=15,pady=10,text = 'Start',command=start)
Sbutton.pack(side=LEFT)

fr=Frame(root,width =40)
fr.pack(side=LEFT)
Abutton2 = Button(root,padx=15,pady=10,text = 'Take Addtandance',command=attd)
Abutton2.pack(side=LEFT)

fr=Frame(root,width =40)
fr.pack(side=LEFT)
STbutton3 = Button(root,padx=15,pady=10,text = 'STOP',command=stop)
STbutton3.pack(side=LEFT)

fr=Frame(root,width =40)
fr.pack(side=LEFT)
SRbutton4 = Button(root,padx=15,pady=10,text = 'Student Registration',command=reg)
SRbutton4.pack(side=LEFT)

fr=Frame(root,width =40)
fr.pack(side=LEFT)
Ebutton5 = Button(root,padx=15,pady=10,text = 'Encode dataset',command=encode)
Ebutton5.pack(side=LEFT)

fr=Frame(root,width =40)
fr.pack(side=LEFT)
Cbutton6 = Button(root,padx=15,pady=10,text = 'Close the system',command=clo)
Cbutton6.pack(side=LEFT)
fr=Frame(root,width =40)
fr.pack(side=LEFT)