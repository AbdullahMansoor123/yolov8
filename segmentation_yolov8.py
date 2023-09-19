import numpy as np
from ultralytics import YOLO
import cv2

vcap = cv2.VideoCapture(0)
while True:
    ok, frame = vcap.read()
    if not ok:
        break
    cv2.flip(frame, 1)
    # print(frame.shape)
    model = YOLO('yolov8n-seg.pt')
    results = model(frame, classes=0)
    for result in results[0]:
        mask = result.masks.masks

        # mask1 = mask[0]
        polygon = mask.xy[0]
        # print(polygon)q
        # cv2.polylines(frame,polygon,True,(0, 255, 0),-1)
        print(mask)

    # annotated_frame = results[0].plot()
    #     cv2.imshow('output', mask)
    if cv2.waitKey(1) == ord('q'):
        break

vcap.release()
cv2.destroyAllWindows()
