import os
import re
from typing import Optional
import cv2 as cv
from PIL import Image
import numpy as np
from pathlib import Path


# 人脸训练模型方法
def get_image_model(file_path: Optional[Path] = None):
    face_sample = []
    ids = []
    # 获取人脸数据
    file_path = file_path or Path(__file__).parent / "faces"
    # 获取所有的图片
    face_detected = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    for img_path in file_path.iterdir():
        img = Image.open(img_path).convert("L")
        # 将图片转成二值化
        img_np = np.array(img, "uint8")
        # LPB提取人脸特征
        faces = face_detected.detectMultiScale(img_np)
        # 文件提取
        fid = int(os.path.split(img_path)[-1].split("-")[0])

        # 假设没有人脸，防止没有人脸图像被训练
        for x, y, w, h in faces:
            ids.append(fid)
            face_sample.append(img_np[y : y + h, x : x + w])
            print("获取图片", img_path, "人脸坐标", x, y, w, h)
    return face_sample, ids


if __name__ == "__main__":
    faces, ids = get_image_model(Path(__file__).parent / "faces")
    # 如果遇到错误AttributeError: module 'cv2' has no attribute 'face'
    # 请升级opencv-python到4.0以上版本
    # pip install --upgrade opencv-contrib-python
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write("face_model.yml")
