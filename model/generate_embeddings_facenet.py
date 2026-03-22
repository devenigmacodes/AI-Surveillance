import os
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
from PIL import Image
import cv2
import json

embedder = FaceNet()
detector = MTCNN()

dataset_dir = "dataset"
output_path = "model/facenet_embeddings.npz"

known_embeddings = []
known_names = []

for person_name in os.listdir(dataset_dir):
    person_folder = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(person_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        results = detector.detect_faces(image)
        if len(results) == 0:
            continue

        x, y, w, h = results[0]['box']
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        embedding = embedder.embeddings([face])[0]

        known_embeddings.append(embedding)
        known_names.append(person_name)

np.savez(output_path, embeddings=known_embeddings, names=known_names)
print(f"Saved embeddings for {len(known_names)} faces → {output_path}")
