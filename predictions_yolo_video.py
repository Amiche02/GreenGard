import supervision as sv
import numpy as np
from ultralytics import YOLO
import cv2

VIDEO_PATH = "data/Growing_tomatoes.mp4"

model = YOLO('Plant_disease_model/best.pt')

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)


def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, imgsz=1280)[0]

    if results.boxes.xyxy.shape[0] > 0:
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu(),
            confidence=results.boxes.conf.cpu(),
            class_id=results.boxes.cls.cpu().int(),
        )

        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

        labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    else:
        # No detections, return the original frame
        frame = frame

    return frame

# Open a window to display the processed video
cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Processed Video", 800, 600)

# Process the video and display the frames
cap = cv2.VideoCapture(VIDEO_PATH)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame, None)
    cv2.imshow("Processed Video", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()