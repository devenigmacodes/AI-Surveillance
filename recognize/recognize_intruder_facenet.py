import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
import os
from datetime import datetime
import json


embedder = FaceNet()
detector = MTCNN()


data = np.load("model/facenet_embeddings.npz", allow_pickle=True)
known_embeddings = data["embeddings"]
known_names = data["names"]

THRESHOLD = 0.9


def log_intruder(face_img):
    os.makedirs("intruders", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"intruders/intruder_{timestamp}.jpg"
    cv2.imwrite(path, face_img)

    entry = {"timestamp": timestamp, "image": path}
    log_file = "intruder_log.json"
    data = []

    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                data = json.load(f)
        except:
            data = []

    data.append(entry)
    with open(log_file, "w") as f:
        json.dump(data, f, indent=2)

    print("🚨 Intruder detected and saved:", path)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, w, h = face["box"]
        face_img = frame[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (160, 160))

        embedding = embedder.embeddings([resized])[0]
        distances = np.linalg.norm(known_embeddings - embedding, axis=1)
        min_dist = np.min(distances)

        if min_dist < THRESHOLD:
            label = "Known"
            color = (0, 255, 0)
        else:
            label = "Intruder"
            color = (0, 0, 255)
            log_intruder(face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Drone Test - Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()