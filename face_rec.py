# import the necessary packages
import time
import cv2 # OpenCV module
import numpy as np #numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import os #os module for reading training data directories and paths

subjects = ["teja","virat_kohli","others"]

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('D:/python codes/FaceRec/face_recognizer/opencv-files/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(img)
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
    return faces, labels

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(img):
    face, rect = detect_face(img)
    if face is not None:
        label, confidence = face_recognizer.predict(face)
        if (confidence<63):
            print ('The confidence is: %s' % confidence)
            label_text = subjects[label]
            print ('Welcome %s' % label_text)
            draw_rectangle(img, rect)
            draw_text(img, label_text, rect[0], rect[1]-5)
            cv2.imshow("Frame", img)
            time.sleep(1)
            return img
        else:
            draw_rectangle(img, rect)
            print ('The confidence is: %s' % confidence)
            label_text = subjects[0]
            print (label_text)
            return img
    else:
        return img
    
print("Training The Device with Saved Faces.")
faces, labels = prepare_training_data("D:/python codes/FaceRec/face_recognizer/training-data")
print("Training Complete")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

cap = cv2.VideoCapture(1)

#Camera starts detection
while True:
    ret,image = cap.read()
    img = predict(image)
    # cv2.imshow("Frame", img)
    # show the frame
    cv2.imshow("Frame", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()