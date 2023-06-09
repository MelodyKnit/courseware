"""人脸数据采集"""
import cv2 as cv
from pathlib import Path


# 开启人脸摄像头
cap = cv.VideoCapture(0)
save_path = Path(__file__).parent / "faces"
face_name = input("请输入你的名字: ")
face_id = input("请输入你的ID: ")
# 创建文件夹
save_path.mkdir(parents=True, exist_ok=True)
num = 0
while cap.isOpened():
    ret, show = cap.read()
    if ret is True:
        # 获取到摄像头，并且灰度图像
        show = cv.cvtColor(show, cv.COLOR_BGR2GRAY)
    else:
        break
    cv.imshow("capture test again", show)

    k = cv.waitKey(1) & 0xFF
    # if k == ord("s"):
    num += 1
    print(f"save capture {num}")
    cv.imencode(".jpg", show)[1].tofile(
        save_path / f"{face_id}-{face_name}-{num}.jpg"
    )
    if k == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
