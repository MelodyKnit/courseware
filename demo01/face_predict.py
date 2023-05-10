import cv2 as cv
import numpy as np


recognizer = cv.face.LBPHFaceRecognizer_create()
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer.read("face_model.yml")

# 开启摄像头
cap = cv.VideoCapture(0)
cap.set(3, cv.VideoWriter.fourcc("m", "j", "p", "g"))
names = ["tsp"]
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 获取灰度人脸特征
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 获取人脸特征
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(64, 64)
    )
    for x, y, w, h in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 人脸预测
        roi = gray[y : y + h, x : x + w]
        id_num, confidence = recognizer.predict(roi)
        
        name = names[id_num] if confidence < 50 else "unknown"
        confidence = "{0}%".format(round(100 - confidence))
        cv.putText(
            frame,
            str(name),
            (x + 5, y - 5),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv.putText(
            frame,
            confidence,
            (x + 5, y + h - 5),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    cv.imshow("face", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv.destroyAllWindows()

