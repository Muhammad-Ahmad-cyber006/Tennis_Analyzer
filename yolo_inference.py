from ultralytics import YOLO
model = YOLO('yolov8x')#pretrained yolo model load
# save=True -> output video runs and folder mei predictions save ho jaayegi agar track use kro tu track mei
result = model.predict('input_videos/input_video.mp4', save=True)  