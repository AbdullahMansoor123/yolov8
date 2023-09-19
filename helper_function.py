import cv2
import numpy as np
from datetime import datetime


def person_detector(result, frame, previous_frame):
    if len(result) > 0:
        previous_frame = frame
    else:
        result = previous_frame
    # bbox = result[:4]
    conf = result[4]
    # row: coordinates of the detected person
    x1, y1, x2, y2 = int(result[0]), int(result[1]), int(result[2]), int(result[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    label_text_box(frame, f'{conf:.2f}', x1, y1, tbox_color=(190, 190, 190))
    return frame


def blur_roi(result, frame, blur_h, factor=3):
    x1, y1, x2, y2 = int(result[0]), int(result[1]), int(result[2]), int(result[3])
    blur_height = int((y2 - y1) * blur_h)  # take the height of the box
    face_roi = frame[y1:blur_height, x1:x2]
    try:
        blur_face = blur(face_roi, factor=factor)
        # replace detection face with blur one
        frame[y1:blur_height, x1:x2] = blur_face
    except:
        pass
    return frame


def label_text_box(frame, text, left, top, tbox_color=None, fontFace=1, fontScale=0.7, fontThickness=1):
    textSize = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    text_w = textSize[0][0]
    text_h = textSize[0][1]
    y_adjust = 10

    cv2.rectangle(frame, (left, top - text_h - y_adjust), (left + text_w + y_adjust, top), tbox_color, -1)
    cv2.putText(frame, text, (left + 5, top - 5), fontFace, fontScale, (0, 0, 0), fontThickness, cv2.LINE_AA)
    return frame


def cal_total_time(video_paths):
    total_frames = 0

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames += frame_count
        cap.release()
    return float(f'{float((total_frames / 25) / 60):.2f}')


def frame_maker(timestamp, fps):
    x = datetime.strptime(timestamp, '%M:%S.%f')
    seconds = x.minute * 60 + x.second + x.microsecond / 1000000
    return seconds * fps


def display_metrics_tabel(frame, size, data):
    # table background color
    frame_width, frame_height = size
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (480, 250), (230, 230, 230), -1)
    alpha = 0.4  # Transparency factor.
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    # Txt = k*F*A
    y = round(0.015 * frame_height * (frame_width / frame_height))
    x = round(0.0029 * frame_width * (frame_width / frame_height))
    fontFace = 2
    fontScale = 0.8
    cv2.putText(frame, "Station    Operators    Productivity", (x, y), fontFace, fontScale, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def net_model(config_file, model_file):
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    return net


def blur(face, factor=3):
    h, w = face.shape[:2]

    if factor < 1:
        factor = 1
    if factor > 5:
        factor = 5

    w_k = int(w / factor)
    h_k = int(h / factor)

    if w_k % 2 == 0: w_k += 1
    if h_k % 2 == 0: h_k += 1
    blurred = cv2.GaussianBlur(face, (int(w_k), int(h_k)), 0, 0)
    return blurred


def face_blur_rect(frame, net, factor, scale, size, mean, detection_threshold):
    img = frame.copy()
    img_out = frame.copy()

    blob = cv2.dnn.blobFromImage(img, scalefactor=scale, size=size, mean=mean)
    net.setInput(blob)
    detection = net.forward()
    w = img.shape[1]
    h = img.shape[0]

    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]

        if confidence > detection_threshold:
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box

            # extract face roi
            face_roi = img[int(y1):int(y2), int(x1):int(x2)]
            blur_face = blur(face_roi, factor=factor)
            # replace detection face wit    h blur one
            img_out[int(y1):int(y2), int(x1):int(x2)] = blur_face
            return img_out
        else:
            return frame


def face_blur_ellipse(image, net, factor=3, detect_threshold=0.90, write_mask=False):
    img = image.copy()
    img_blur = img.copy()

    elliptical_mask = np.zeros(img.shape, dtype=img.dtype)

    # Prepare image and perform inference.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123])
    net.setInput(blob)
    detections = net.forward()

    h, w = img.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detect_threshold:
            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box

            # The face is defined by the bounding rectangle from the detection.
            face = img[int(y1):int(y2), int(x1):int(x2), :]

            # Blur the rectangular area defined by the bounding box.
            face = blur(face, factor=factor)

            # Copy the `blurred_face` to the blurred image.
            img_blur[int(y1):int(y2), int(x1):int(x2), :] = face

            # Specify the elliptical parameters directly from the bounding box coordinates.
            e_center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
            e_size = (x2 - x1, y2 - y1)
            e_angle = 0.0

            # Create an elliptical mask.
            elliptical_mask = cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle),
                                          (255, 255, 255), -1, cv2.LINE_AA)
            # Apply the elliptical mask
            np.putmask(img, elliptical_mask, img_blur)

    if write_mask:
        cv2.imwrite('elliptical_mask.jpg', elliptical_mask)

    return img
