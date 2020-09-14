import tensorflow as tf
import tensorflow.keras.backend as K

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.set_session(tf.Session(config=config))

import cv2
import numpy as np
from numpy import dot
from numpy.linalg import norm
from math import ceil, pi
from tensorflow.keras.models import load_model

net = load_model("model/senet50.h5")
# net = cv2.dnn.readNetFromTensorflow("model/senet50.pb")

detector = cv2.dnn.readNetFromCaffe("detector/deploy.prototxt", "detector/res10_300x300_ssd_iter_140000.caffemodel")

landmarks = cv2.dnn.readNetFromCaffe("landmarks/landmark_deploy.prototxt", "landmarks/VanFace.caffemodel")

def detect_face(img, min_conf=0.6):

    detector.setInput(cv2.dnn.blobFromImage(cv2.resize(img, (300, 300))) )
    detections = detector.forward()

    detection = detections[0][0][0]

    (h, w) = img.shape[:2]

    
    box = detection[3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    if startX >=0 and startX < w and endX >= 0 and endX <= w and startY >=0 and startY < h and endY >=0 and endY <=h:
        return img[startY:endY, startX:endX]
    

    return img

def eye_locations(img):

    w = 60
    h = 60

    height, width = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (w, h))

    mean = np.mean(img)
    std = np.std(img)

    img = cv2.dnn.blobFromImage(img, scalefactor=1.0/(0.000001 + std), mean=mean)

    landmarks.setInput(img)

    points = landmarks.forward()[0]

    leftEye_loc = points[36*2: 41*2 + 2];
    rightEye_loc = points[42*2: 47*2 + 2];

    left_center = ( np.mean([ leftEye_loc[2*i] for i in range(len(leftEye_loc)//2) ]), np.mean([ leftEye_loc[2*i+1] for i in range(len(leftEye_loc)//2) ]) )   

    right_center = ( np.mean([ rightEye_loc[2*i] for i in range(len(rightEye_loc)//2) ]), np.mean([ rightEye_loc[2*i+1] for i in range(len(rightEye_loc)//2) ]) )   

    return left_center, right_center


def distance(a, b):
    
    diff = np.asarray(a) - np.asarray(b)
    return np.sqrt(diff[0]**2 + diff[1]**2)


def rotate_image(image, angle):
    
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def correct_face(img):

    img = detect_face(img)

    left_eye, right_eye = eye_locations(img)


    if left_eye[1] > right_eye[1]:
        point_3rd = (right_eye[0], left_eye[1])
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye[0], right_eye[1])
        direction = 1 #rotate inverse direction of clock

    #-----------------------
    #find length of triangle edges

    a = distance(left_eye, point_3rd)
    b = distance(right_eye, point_3rd)
    c = distance(right_eye, left_eye)

    #-----------------------
    #apply cosine rule
    
    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
    
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / pi #radian to degree
        
        #-----------------------
        #rotate base image
        
        if direction == -1:
            angle = 90 - angle
        
        img = rotate_image(img, direction * angle)
        
        #you recover the base image and face detection disappeared. apply again.
        face = detect_face(img)
        
        if face is not None:
            img = face
    
    return img
        

def preprocess(img, opencv=False):
    
    img = cv2.resize(img, (224, 224))
    
    if opencv:

        blob = cv2.dnn.blobFromImage(img, mean=(91.4953, 103.8827, 131.0912), swapRB=True)
        return blob


    else:
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = x.astype("float32")
        
        x = np.expand_dims(x, axis=0)
        x[..., 0] -= 91.4953
        x[..., 1] -= 103.8827
        x[..., 2] -= 131.0912


        return x

def l1_norm(arr):
    return arr / (0.00000001 + np.sum(np.abs(arr)) )


def run(img):

    img = correct_face(img)

    # Run keras
    res = net.predict( preprocess(img) )

    # # Run opencv
    # net.setInput(preprocess(img, opencv=True))
    # res = net.forward()

    res = res.flatten()

    return l1_norm(res)
