import cv2
import numpy as np
import os

input_image = "/Users/devanshkumaryaduvanshi/Downloads/person1.jpg"
output_folder = "cropped_signatures"

os.makedirs(output_folder, exist_ok=True)

img = cv2.imread(input_image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Remove left margin containing numbers
h, w = gray.shape
gray = gray[:, int(w*0.15):]

# Threshold
_, binary = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Horizontal projection
projection = np.sum(binary, axis=1)

rows = []
in_row = False
start = 0

for i, val in enumerate(projection):
    if val > 1000 and not in_row:
        start = i
        in_row = True

    elif val < 1000 and in_row:
        end = i
        rows.append((start, end))
        in_row = False

count = 0

for (start, end) in rows:

    row_img = gray[start:end, :]

    h_row, w_row = row_img.shape
    mid = w_row // 2

    left = row_img[:, :mid]
    right = row_img[:, mid:]

    cv2.imwrite(f"{output_folder}/sig_{count}.png", left)
    count += 1

    cv2.imwrite(f"{output_folder}/sig_{count}.png", right)
    count += 1

print("Extracted signatures:", count)