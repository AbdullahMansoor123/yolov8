import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
target_idx = 0  # person
# Open the video file
video_path = "104134.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if not success:
        break
    if success:
        # frame = cv2.flip(frame,1)
        # Run YOLOv8 inference on the frame
        results = model(frame, classes=target_idx)
        # print(results[0].boxes.data)
        for result in results[0]:
            bbox = result.boxes.data[0][:4]
            class_conf = result.boxes.data[0][4]
            class_idx = result.boxes.data[0][5]
            # print()
            print(bbox)
            for box in result.boxes.cpu().numpy():
                r = box.xyxy[0].astype(int)
                cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)
        # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
