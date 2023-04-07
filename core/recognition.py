import cv2
import time
import pickle
import os
from PIL import Image
from numpy import asarray, degrees, true_divide, frombuffer, float64
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from matplotlib import pyplot
from django.db.models import F
from scipy.spatial.distance import cosine
# from . import attendance_writer as at
from core import attendance_writer as at
from core.models import FaceModel


# HaarCascade used as a detector
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# vggface model for face embedding to recognize face
model = VGGFace(model='resnet50', include_top=False,
                input_shape=(224, 224, 3), pooling='avg')
all_face_embeddings = FaceModel.objects.all()

import face_recognition

def detect_facial_landmarks(face):
    # Detect facial landmarks using face_recognition library
    landmarks = face_recognition.face_landmarks(face)
    
    # Return the detected landmarks
    return landmarks

def calculate_motion(landmarks):
    # Calculate motion of facial landmarks by computing the Euclidean distance
    # between consecutive frames for each facial landmark
    motion = []
    for landmark_set in landmarks:
        for landmark in landmark_set.values():
            motion.append(sum([(landmark[i][0] - landmark[i-1][0])**2 + 
                              (landmark[i][1] - landmark[i-1][1])**2 
                              for i in range(1, len(landmark))]))
    
    # Return the calculated motion values
    return motion

def analyze_motion(motion):
    # Analyze motion patterns to determine if face is live or fake
    # You can define your own threshold based on your specific use case
    # and environment
    threshold = 1500  # Example threshold value
    
    # If the sum of motion values is below the threshold, consider the face as live
    is_live = sum(motion) < threshold
    
    # Return True if face is live, False otherwise
    return is_live


def save_embedding(emb, name):
    """
    Function for saving face embeddings to Django database
    """
    face_embedding = FaceModel(name=name, embedding=emb)
    face_embedding.set_face_embedding(emb)
    face_embedding.save()
    return

def dataset():
    """
    Function for retrieving face embeddings from Django database
    """
    all_face_embeddings = FaceModel.objects.all()
    return all_face_embeddings



def extract_face(filename):
    """
    returns the face after extraction and
    returned face is largest among all the extracted faces
    """
    pixels = pyplot.imread(filename)
    faces = detector.detectMultiScale(pixels, 1.1, 5)
    if len(faces) == 0:
        return False
    x1, y1, x2, y2 = get_largest_face(faces)
    face = pixels[y1:y2, x1:x2]
    return face

def save_from_folder(path):
    """
    Save faces details from the folder
    """

    files = [file for file in os.listdir(path) if file.lower().endswith(('.jpeg', '.png', '.jpg'))]
    faces = [extract_face(os.path.join(path, file)) for file in files]
    embeddings = get_embedding(faces)
    for i, emb in enumerate(embeddings):
        name = file.split('.')[0]
        # Check if a face embedding with the same name already exists in the database
        if FaceModel.objects.filter(name=name).exists():
            print(f"Face embedding for {name} already exists in the database.")
            continue
        # Create a new FaceModel instance and save it to the database
        face_embedding = FaceModel(name=name, embedding=emb)
        face_embedding.set_face_embedding(emb)
        face_embedding.save()
        print(f"Face embedding for {name} saved to the database.")


def save_from_file(filename, name):
    """
    saves the face embedding with name in dataset
    parameters:
        ->filename: name of file in which image of person is
        ->name :  name of person
    returns None
    """
    face = extract_face(filename)
    embedding = get_embedding([face])[0]
    save_embedding(embedding, name)


def identify(known_embedding):
    """
    recognize face by comparing it with all faces present in dataset
    parameters:
        -> known_embedding : embedding of face to be recognized
    returns:
        True, str(name) : if person is identified
        False, "unknown" : if face is not identified
    """
    for face_embedding in all_face_embeddings:
        unknown_embedding = face_embedding.get_face_embedding() # Convert bytes to numpy array
        if is_match(unknown_embedding, known_embedding):
            return True, face_embedding.name
    return False, "unknown"


def get_largest_face(faces):
    """
    accept the list of faces and returns the coordinates of the face having largest area
    parameters:
        -> faces : list of faces
    returns:
        x1,y1,x2,y2 : coordinates of largest face
    """
    x, y, w, h = 0, 0, 0, 0
    for face in faces:
        x1, y1, w2, h2 = face
        if w*h < w2*h2:
            w, h = w2, h2
            x, y = x1, y1
    x2, y2 = x + w, y + h
    return x, y, x2, y2


def get_embedding(faces):
    """
    returns embedding of faces using vggface model
    parametes:
        -> faces : list of faces whose embeddings are required
    returns:
        enbedding of faces
    """
    face_array = []
    for face in faces:
        image = Image.fromarray(face)
        image = image.resize((224, 224))
        face_array.append(asarray(image))
    samples = asarray(face_array, 'float32')
    samples = preprocess_input(samples, version=2)
    yhat = model.predict(samples)
    return yhat


def is_match(known_embedding, candidate_embedding, thresh=0.5):
    """
    compare the cosine distance b\w known face encoding and candidate face encoding 
    and tells whether the face are of same person or not
    parameters :  
        -> known_embedding (embedding of face of known person)
        -> candidate_embedding (embedding of face of unknown person)
        -> thersh (thershold for face matching)
    returns:
        True if same person
        False if different person
    """
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        return True
    else:
        return False


# Function used to detect the faces and recoginize them


def recognize():
    attendance = at.Attendance()
    capture = cv2.VideoCapture(0)
    n = 0
    last_10_ids = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img = frame
        if n % 3:
            cv2.imshow("Face Recognition", img)
            continue
            n += 1
        n = 0
        # reducing frame size to speed up face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        faces = detector.detectMultiScale(small_frame, 1.1, 5)

        if len(faces) == 0:
            img = cv2.putText(frame, "no face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("face recognition", img)
            continue
        # selecting largest of all faces only
        x1, y1, x2, y2 = get_largest_face(faces)
        # enclosing face in a rectangle
        img = cv2.rectangle(img, (x1*4, y1*4), (x2*4, y2*4), (0, 255, 0), 2)
        face = small_frame[y1:y2, x1:x2]
        ##liveness detection
        landmarks = detect_facial_landmarks(face)
        if landmarks is not None:
            # Calculate motion of facial landmarks
            motion = calculate_motion(landmarks)
            
            # Analyze motion patterns and determine if face is live or fake
            is_live = analyze_motion(motion)
            
            if is_live:
                emb = get_embedding([face])[0]  # finding embedding of face
                found, id = identify(emb)  # recognizing face
                if len(last_10_ids) <= 10:
                    last_10_ids.append(id)
                    continue
                else:
                    last_10_ids.pop()
                not_sure = max([0 if id==lid  else 1 for lid in last_10_ids])
                if not not_sure:
                    if attendance.write_attendance(id):
                        print(f"Attendance written for {id}")
                    else:
                        print(f"Attendance already exists for {id}")
                else:
                    print("Not sure the person is actually them or I am an idiot.")
            else:
                print("Face liveness detection failed. Possibly a fake or photo/video attack.")
                id = " --- Fake Person"
        else:
            print("Failed to detect facial landmarks. Possibly a fake or photo/video attack.")
            id = " --- Fake Person"
            
            # Draw text on frame with recognized name
        img = cv2.putText(frame, f"Welcome {id}", (x1*4, y1*4), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("face recognition", img)
    capture.release()
    cv2.destroyAllWindows()



def capture():
    capture = cv2.VideoCapture(0)
    q = False
    emb = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            q = False
            break
        img = frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        faces = detector.detectMultiScale(small_frame, 1.1, 5)
        if len(faces) == 0:
            img = cv2.putText(frame, "no face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Press q to quit", img)
            continue
        # selecting largest of all faces only
        x1, y1, x2, y2 = get_largest_face(faces)
        # enclosing face in a rectangle
        img = cv2.rectangle(img, (x1*4, y1*4), (x2*4, y2*4), (0, 255, 0), 2)
        face = small_frame[y1:y2, x1:x2]
        emb = get_embedding([face])[0]  # finding embedding of face
        cv2.imshow("Press c to capture", img)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            q = True
            break
    capture.release()
    cv2.destroyAllWindows()
    return q, emb
