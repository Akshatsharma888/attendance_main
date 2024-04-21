import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

akshat_image = face_recognition.load_image_file("smart_attendance/akshat.jpg")
akshat_encoding = face_recognition.face_encodings(akshat_image)[0]

mark_zucker_image = face_recognition.load_image_file("smart_attendance/mark.jpg")
mark_encoding = face_recognition.face_encodings(mark_zucker_image)[0]

mrbean_image = face_recognition.load_image_file("smart_attendance/mr_bean.jpg")
bean_encoding = face_recognition.face_encodings(mrbean_image)[0]

tom_cruise_image = face_recognition.load_image_file("smart_attendance/tom.jpg")
tom_encoding = face_recognition.face_encodings(tom_cruise_image)[0]

known_face_encoding = [akshat_encoding, mark_encoding, bean_encoding, tom_encoding]

known_faces_names = ["akshat", "mark", "mr_bean", "tom"]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + ".csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(
                known_face_encoding, face_encoding
            )
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2

                cv2.putText(
                    frame,
                    name + " Present",
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType,
                )

                if name in students:
                    students.remove(name)
                    print(students)
                    now = datetime.now()  # Moved this line inside the loop
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

        cv2.imshow("attendance system", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break