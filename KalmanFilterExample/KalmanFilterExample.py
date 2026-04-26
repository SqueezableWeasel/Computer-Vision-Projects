import cv2
import numpy as np
import imageio

video_path = "ferret.mp4"
output_gif = "output_kalman.gif"

cap = cv2.VideoCapture(video_path)

back_sub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

kalman = cv2.KalmanFilter(4, 2)

# State = [x, y, dx, dy]
kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)

kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
kalman.errorCovPost = np.eye(4, dtype=np.float32)

initialized = False
frames = []

frame_skip = 2      # use every 2nd frame for smaller GIF
resize_scale = 0.5  # shrink output size

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

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

            corr_x = int(corrected[0])
            corr_y = int(corrected[1])

            # Bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Raw center
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            # Kalman center
            cv2.circle(frame, (corr_x, corr_y), 6, (0, 0, 255), -1)

    # Resize for smaller GIF
    frame = cv2.resize(
        frame,
        None,
        fx=resize_scale,
        fy=resize_scale
    )

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frames.append(rgb)

cap.release()

# Save GIF
imageio.mimsave(output_gif, frames, fps=10)

print("Saved:", output_gif)
