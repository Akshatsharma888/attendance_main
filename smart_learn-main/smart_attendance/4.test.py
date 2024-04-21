import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import face_recognition
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Directory where the known faces are located
image_directory = Path("named_images")

# Get a list of all image files in the directory
image_files = [f for f in image_directory.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']]

# Initialize known_faces dictionary
known_faces = {}

# Load known faces
for image_file in image_files:
    name = image_file.stem  # Name of the person is the filename without the extension
    print(f"Loading image: {image_file}")
    face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(str(image_file)))
    if face_encodings:
        known_faces[name] = face_encodings[0]
        print(f"Face encoding loaded for: {name}")
    else:
        print(f"No face encoding found for: {name}")

known_face_encodings = list(known_faces.values())
known_face_names = list(known_faces.keys())

# Initialize DataFrame to store attendance
attendance = pd.DataFrame(columns=['Name', 'Time', 'Status'])

# Add all known faces to the attendance DataFrame with status 'Absent'
for name in known_face_names:
    attendance = pd.concat([attendance, pd.DataFrame([{'Name': name, 'Time': None, 'Status': 'Absent'}])], ignore_index=True)

# Path to the group photo
group_photo_path = Path(r"C:\smart_learn-main\smart_attendance\grap.jpg")  # replace 'grap.jpg' with your actual filename and extension

image = cv2.imread(str(group_photo_path))

if image is None:
    print(f"Could not open or find the image: {group_photo_path}")
else:
    print("Group photo loaded successfully.")
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find the locations of all faces in the image
    face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')

    print(f"Found {len(face_locations)} face(s) in the group photo.")

    # Define a function to process each face using parallel processing
    def process_face(face_location):
        top, right, bottom, left = face_location
        face_image = rgb_small_frame[top:bottom, left:right]
        face_encodings = face_recognition.face_encodings(face_image)
        if face_encodings:  
            face_encoding = face_encodings[0]

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Mark the individual as present
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            return (name, current_time)
        else:
            return ("Unknown", None)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_face, face_locations)
    
    # Process the results
    for result, face_location in zip(results, face_locations):
        name, current_time = result
        top, right, bottom, left = face_location
        if name != "Unknown":
            attendance.loc[attendance['Name'] == name, ['Time', 'Status']] = [current_time, 'Present']
        print(f"Face recognized as {name}.")

    # Save the attendance to an Excel file
    attendance_file_name = datetime.now().strftime("%Y-%m-%d") + ".xlsx"
    attendance.to_excel(attendance_file_name, index=False)
    print(f"Attendance saved to {attendance_file_name}.")
