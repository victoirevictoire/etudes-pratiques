from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
results = model("carte.png", imgsz=1024, conf=0.3)
results[0].show()  # affiche l'image avec les boîtes



# bash:
# yolo detect predict model=best.pt source=carte.png imgsz=1024 conf=0.3
