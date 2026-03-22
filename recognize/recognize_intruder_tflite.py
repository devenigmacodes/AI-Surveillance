import cv2
import numpy as np
from mtcnn import MTCNN
import tflite_runtime.interpreter as tflite
from datetime import datetime
import json, os


interpreter = tflite.Interpreter(model_path="model/facenet_model_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

detector = MTCNN()


data = np.load("model/facenet_embeddings.npz", allow_pickle=True)
known_embeddings = data["embeddings"]
THRESHOLD = 0.9  # adjust later if needed

def get_embedding(face_img):
    face = cv2.resize(face_img, (160, 160))
    face = np.expand_dims(face, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], face)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def log_intruder(face_img):
    os.makedirs("intruders", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"intruders/intruder_{ts}.jpg"
    cv2.imwrite(path, face_img)
    entry = {"timestamp": ts, "image": path}
    log_file = "intruder_log.json"
    data = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f: data = json.load(f)
        except: data = []
    data.append(entry)
    with open(log_file, "w") as f: json.dump(data, f, indent=2)
    print("🚨 Intruder logged:", path)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)
    for f in faces:
        x, y, w, h = f["box"]
        face_img = frame[y:y+h, x:x+w]
        embedding = get_embedding(face_img)
        distances = np.linalg.norm(known_embeddings - embedding, axis=1)
        if np.min(distances) < THRESHOLD:
            label = " Known"
            color = (0,255,0)
        else:
            label = "Intruder"
            color = (0,0,255)
            log_intruder(face_img)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    cv2.imshow("Drone Test - Pi Intruder Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()