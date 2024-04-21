'''this code is marking present based on the single inputed images.'''

import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os
import pandas as pd

# Directory where the known faces and test images are located
image_directory = "named_images"

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Initialize known_faces dictionary
known_faces = {}

# Load known faces
for image_file in image_files:
    name = os.path.splitext(image_file)[0]  # Name of the person is the filename without the extension
    image_path = os.path.join(image_directory, image_file)
    known_faces[name] = face_recognition.face_encodings(face_recognition.load_image_file(image_path))[0]

known_face_encodings = list(known_faces.values())
known_face_names = list(known_faces.keys())

# Initialize DataFrame to store attendance
attendance = pd.DataFrame(columns=['Name', 'Time', 'Status'])

# Process each image file
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not open or find the image: {image_path}")
        continue

    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Check if the name of the image file (without the extension) is in the list of known faces
        image_name = os.path.splitext(image_file)[0]
        if image_name in known_face_names:
            # Mark the individual as present
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            attendance = pd.concat([attendance, pd.DataFrame([{'Name': name, 'Time': current_time, 'Status': 'Present'}])], ignore_index=True)

# Save the attendance to an Excel file
attendance.to_excel(datetime.now().strftime("%Y-%m-%d") + ".xlsx", index=False)