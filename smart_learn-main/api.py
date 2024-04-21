from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import face_recognition
from pathlib import Path

app = Flask(__name__)

# Directory where the known faces are located
image_directory = Path("named_images")

known_faces = {}
known_face_encodings = []
known_face_names = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global known_faces, known_face_encodings, known_face_names
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(image_directory, filename))

    name = os.path.splitext(filename)[0]
    face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(os.path.join(image_directory, filename)))
    if face_encodings:
        known_faces[name] = face_encodings[0]
        known_face_encodings = list(known_faces.values())
        known_face_names = list(known_faces.keys())
        return jsonify({"message": f"Face encoding loaded for: {name}"}), 200
    else:
        return jsonify({"message": f"No face encoding found for: {name}"}), 400

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)

    image = cv2.imread(filename)

    if image is None:
        return jsonify({"error": f"Could not open or find the image: {filename}"}), 400

    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    attendance = pd.DataFrame(columns=['Name', 'Time', 'Status'])

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        attendance = pd.concat([attendance, pd.DataFrame([{'Name': name, 'Time': current_time, 'Status': 'Present'}])], ignore_index=True)

    attendance_file_name = now.strftime("%Y-%m-%d_%H-%M-%S") + ".xlsx"
    attendance.to_excel(attendance_file_name, index=False)

    return send_file(attendance_file_name, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)