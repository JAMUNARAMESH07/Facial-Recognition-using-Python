import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)


Rithanya_image = face_recognition.load_image_file("photos/Rithanya.jpg")
Rithanya_encoding = face_recognition.face_encodings(Rithanya_image)[0]

pragadeesh_image = face_recognition.load_image_file("photos/prk.jpg")
pragadeesh_encoding = face_recognition.face_encodings(pragadeesh_image)[0]

Vasuki_image = face_recognition.load_image_file("photos/Vasuki.jpg")
Vasuki_encoding = face_recognition.face_encodings(Vasuki_image)[0]
Karthikaa_image = face_recognition.load_image_file("photos/Karthikaa.jpg")
Karthikaa_encoding = face_recognition.face_encodings(Karthikaa_image)[0]


Jamuna_image = face_recognition.load_image_file("photos/Jamuna.jpg")
Jamuna_encoding = face_recognition.face_encodings(Jamuna_image)[0]


Kani_image = face_recognition.load_image_file("photos/Kani.jpg")
Kani_encoding = face_recognition.face_encodings(Kani_image)[0]


known_face_encoding = [
    Rithanya_encoding,
    pragadeesh_encoding,
    Vasuki_encoding,
    Karthikaa_encoding,
    Jamuna_encoding,
    Kani_encoding
]

known_faces_names = [
    "Rithanya",
    "pragadeesh",
    "Vasuki",
    "Karthikaa",
    "Jamuna",
    "Kani"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
Inwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
    # Convert the numpy array to a `full_object_detection` object.
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encoding_list = []
        for face_location in face_locations:
            face_encoding = face_recognition.face_encodings(rgb_small_frame, [face_location])[0]
            face_encoding_list.append(face_encoding)

        face_encodings = face_encoding_list

        face_names = []
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.6:  # Adjust this threshold as needed
                name = known_faces_names[best_match_index]
                face_names.append(name)
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    Inwriter.writerow([name, current_time])

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


    cv2.imshow("attendance system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
