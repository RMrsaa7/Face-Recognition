#WEB library
import streamlit.components.v1 as components
from secrets import choice
import streamlit as st

#opencv library
import face_recognition #
import numpy as np
import cv2
import os
import time

 
FRAME_WINDOW = st.image([]) #membuka jendela baru di browser. ketika aplikasi berhasil di run maka otomatis 
#akan membuka jendela baru di browser untuk menampilkan konten/programnya


menu = ["HOME","SCAN"] #variabel menu berfungsi membuat daftar menu yang akan ditampilkan di sidebar aplikasi.
choice = st.sidebar.selectbox("Menu", menu) #tampilan sidebar menu, selectbox digunakan untuk membuat kotak pilihan 
#untuk memilih satu opsi dari daftar yang disediakan pada variabel menu

path = 'absensi' #path menyimpan gambar
images = [] #list gambar (fungsi ini digunakan untuk membaca gambar yang ada di path, lalu dimasukkan kedalam variabel images ini)
classNames = [] #list nama-nama yang terkait dengan setiap gambar
myList = os.listdir(path) #list nama file gambar, untuk membaca setiap file gambar satu per satu dan diproses lebih lanjut.

col1, col2 = st.columns(2) #columns, artinya membuat 2 kolom layout
cap = cv2.VideoCapture(0) #capture video
if choice == 'SCAN': #jika memilih opsi scan, maka blok kode berikutnya akan dieksekusi
    with col1: #memasukkan konten kedalam kolom 1
        st.subheader("SCAN WAJAH") #menampilkan judul kecil. 
        run = st.checkbox("Nyalakan camera") #checkbox ini fungsinya untuk menampilkan checkbox dengan label nyalakan kamera.
        #jika pengguna mencentangnya maka kamera akan diaktifkan.
    if run == True:
        for cl in myList: #menggunakan loop karena kita akan melakukan beberapa eksekusi pada setiap gambar dalam folder absensi
            
            #BAGIAN PREPROCESSIGA PADA GAMBAR
            curlImg = cv2.imread(f'{path}/{cl}') #membaca gambar
            images.append(curlImg) #simpan gambar
            classNames.append(os.path.splitext(cl)[0]) #split nama gambar, nama gambar akan disimpan di variabel classNames
        print(classNames)

        def findEncodings(images): #menemukan encoding (representasi numerik) dari wajah yang ada di gambar
            encodeList = [] #digunakan untuk menyimpan encoding dari setiap wajah
            for img in images:
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #RGB2BGR merupakan format umum yang digunakan dalam pengolahan gambar dgn openCV

                #PROSES EKSTRAKSI PADA GAMBAR
                encode = face_recognition.face_encodings(img)[0] #proses ekstraksi wajah dan dikonversi menjadi numerik yang kemudian akan digunakan untuk mencocokan, mengenali wajah.
                encodeList.append(encode) 
            return encodeList

        def faceList(name):
            with open('absensi.csv', 'r+') as f:
                myDataList = f.readlines()
                nameList = []
                for line in myDataList:
                    entry = line.split(',')
                    nameList.append(entry[0])

        encodeListUnkown = findEncodings(images)
        print('encoding complate!')
        while True:
            success, img = cap.read() #loop yang berjalan terus menerus(real-time) untuk mengambil frame video dari kamera

            #INI BAGIAN PRE PROCESSING
            imgS = cv2.resize(img,(0,0),None,0.25,0.25) #resize untuk mempercepat pemrosesan
            imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB) #BGR2RGB merupakan format umum yang digunakan dalam pengolahan gambar dgn openCV(format yang dibutuhkan facerecognition)
            faceCurFrame = face_recognition.face_locations(imgS) #deteksi lokasi wajah
            #INI BAGIAN EKTRAKSI PADA WAJAH YG DIPINDAI
            encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame) #ekstraksi encoding dari wajah yang terdeteksi

            for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
                matches = face_recognition.compare_faces(encodeListUnkown,encodeFace) #membandingkan encoding wajah yang terdeteksi dengan encoding wajah yang dikenal
                faceDis = face_recognition.face_distance(encodeListUnkown,encodeFace) #menghitung jarak euclidean antara encodig wajah yang terdeteksi dan semua encoding wajah yang dikenal
                #print(faceDis)
                matchesIndex = np.argmin(faceDis) # encoding wajah yang dikenal yang paling mirip/dekat dengan wajah yangterdeteksi
                
                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

                if matches[matchesIndex]:
                    name = classNames[matchesIndex].upper() #jika ada kecocokan, nama diambil dari classNames
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2) #buat frame hijau 
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2) #menambahkan nama yang dikenali
                else:
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
                    cv2.putText(img,"Tidak diketahui",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2) #jika tidak cocok dengan encoding wajah manapun
            FRAME_WINDOW.image(img)
            cv2.waitKey(1)
    else:
        pass

elif choice == 'HOME':
    with col1:
        st.image("absensi/devi nuralim.jpeg",width=500) 