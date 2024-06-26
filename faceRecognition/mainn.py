import streamlit.components.v1 as components
import streamlit as st
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

def load_images_and_augment(path):
    images = []
    classNames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                filePath = os.path.join(root, file)
                img = cv2.imread(filePath)
                if img is not None:
                    images.append(img)
                    classNames.append(os.path.basename(root))
                    img_flipped = cv2.flip(img, 1)
                    images.append(img_flipped)
                    classNames.append(os.path.basename(root))
    return images, classNames

images, classNames = load_images_and_augment(path)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

def find_camera_index(max_index=20):
    print("Mencari indeks kamera...")
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Kamera ditemukan pada indeks {index}")
            cap.release()
            return index
        cap.release()
    print("Tidak ada kamera yang terdeteksi.")
    return None

camera_index = find_camera_index()

col1, col2 = st.columns(2)

if choice == 'SCAN':
    with col1:
        st.subheader("SCAN WAJAH")
        run = st.checkbox("Nyalakan kamera")

    if run and camera_index is not None:
        cap = cv2.VideoCapture(camera_index)
        cap.set(3, 320)  # Lebar dikurangi menjadi 320
        cap.set(4, 240)  # Tinggi dikurangi menjadi 240
        encodeListKnown = findEncodings(images)
        print('Encoding complete!')

        while run:
            success, img = cap.read()
            if not success:
                st.write("Gagal membaca frame dari kamera.")
                break

            imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)
            img_gray = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)
            imgS = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

            faceCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)  

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2

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

        cap.release()
    else:
        st.write("Tidak ada kamera eksternal yang ditemukan atau kamera tidak diaktifkan.")

elif choice == 'HOME':
    with col1:
        st.image("absensi/ARS.jpg", width=500)
