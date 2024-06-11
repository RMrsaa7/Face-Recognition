import streamlit.components.v1 as components
import streamlit as st

# opencv library
import face_recognition
import numpy as np
import cv2
import os

FRAME_WINDOW = st.image([])

menu = ["HOME", "SCAN"]
choice = st.sidebar.selectbox("Menu", menu)

path = 'absensi'
images = []
classNames = []
# Iterasi melalui setiap sub-folder di dalam folder 'absensi'
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):  # Sesuaikan dengan format file gambar Anda
            filePath = os.path.join(root, file)
            images.append(cv2.imread(filePath))
            classNames.append(os.path.basename(root))  # Menggunakan nama folder sebagai label

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        encode = face_recognition.face_encodings(img)
        if encode:  # Pastikan ada encoding yang ditemukan
            encodeList.append(encode[0])
    return encodeList

col1, col2 = st.columns(2)
cap = cv2.VideoCapture(0)

if choice == 'SCAN':
    with col1:
        st.subheader("SCAN WAJAH")
        run = st.checkbox("Nyalakan kamera")

    if run:
        encodeListUnknown = findEncodings(images)
        print('Encoding complete!')
        while run:
            success, img = cap.read()

            # Proses resize frame dilakukan di awal sebelum proses deteksi wajah
            imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)  # Ubah ukuran frame menjadi setengah dari ukuran asli
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            faceCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListUnknown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListUnknown, encodeFace)
                matchIndex = np.argmin(faceDis)

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2  # Sesuaikan koordinat kembali ke ukuran asli

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "Tidak diketahui", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            FRAME_WINDOW.image(img)
            cv2.waitKey(1)
    else:
        cap.release()

elif choice == 'HOME':
    with col1:
        st.image("absensi/devi/Devi Nuralim.jpeg", width=500)
