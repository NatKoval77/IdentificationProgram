import dlib
import cv2
import numpy as np

def init_dlib():
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    return detector, sp, facerec

def get_dlib_embedding(img_path, detector, sp, facerec):
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(rgb)

    if len(faces) == 0:
        return None

    shape = sp(rgb, faces[0])
    embedding = facerec.compute_face_descriptor(rgb, shape)
    return np.array(embedding) 