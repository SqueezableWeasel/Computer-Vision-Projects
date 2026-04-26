import cv2
import numpy as np
import imageio
from ultralytics import YOLO

video_path = "ferret.mp4"
output_gif = "output_kalman_with_YoloV8.gif"

cap = cv2.VideoCapture(video_path)

# YOLOv8 classification model
# Replace with your own classifier if desired
model = YOLO("yolov8n-cls.pt")

back_sub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

kalman = cv2.KalmanFilter(4, 2)

# State: x, y, dx, dy
kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)

# Measurement: x, y
kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
kalman.errorCovPost = np.eye(4, dtype=np.float32)

initialized = False
gif_frames = []

# GIF resize width (smaller = smaller file)
gif_width = 640

# Keep every Nth frame for smaller GIF
frame_skip = 2
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1

    fg_mask = back_sub.apply(frame)

    fg_mask = cv2.medianBlur(fg_mask, 5)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    prediction = kalman.predict()

    if initialized:
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 255), -1)

    if contours:
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > 800:
            x, y, w, h = cv2.boundingRect(c)

            motion_crop = frame[y:y+h, x:x+w]

            label_text = "Unknown"

            if motion_crop.size > 0:
                results = model.predict(
                    source=motion_crop,
                    verbose=False
                )

                probs = results[0].probs
                class_id = int(probs.top1)
                conf = float(probs.top1conf)

                class_name = results[0].names[class_id]
                label_text = f"{class_name}: {conf:.2f}"

            center_x = x + w // 2
            center_y = y + h // 2

            measurement = np.array([
                [np.float32(center_x)],
                [np.float32(center_y)]
            ])

            if not initialized:
                kalman.statePost = np.array([
                    [center_x],
                    [center_y],
                    [0],
                    [0]
                ], np.float32)
                initialized = True

            corrected = kalman.correct(measurement)
            corr_x, corr_y = int(corrected[0]), int(corrected[1])

            # Draw box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Draw label
            cv2.putText(
                frame,
                label_text,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Draw centers
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (corr_x, corr_y), 6, (0, 0, 255), -1)

    # Save every Nth frame to GIF
    if frame_index % frame_skip == 0:
        h, w = frame.shape[:2]
        scale = gif_width / w
        new_h = int(h * scale)

        resized = cv2.resize(frame, (gif_width, new_h))

        # Convert BGR -> RGB for GIF
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        gif_frames.append(rgb)

cap.release()

# Write GIF
imageio.mimsave(
    output_gif,
    gif_frames,
    duration=0.08,   # seconds per frame
    loop=0
)

print("Saved:", output_gif)
