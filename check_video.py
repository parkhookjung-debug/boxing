import cv2
import os

path = "bivol match.mp4"
print("파일 존재:", os.path.exists(path))
print("파일 크기:", os.path.getsize(path) / (1024*1024), "MB")

cap = cv2.VideoCapture(path)
print("열림:", cap.isOpened())
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("총 프레임:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("해상도:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("재생시간:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)), "초")
cap.release()
