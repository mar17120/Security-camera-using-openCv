import cv2
import winsound
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Camera not opened")
    exit()

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=True
)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("motion_record.avi", fourcc, 20.0, (frame_width, frame_height))

last_alert = 0
cooldown = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = bg_subtractor.apply(frame)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 3000:
            continue

        motion_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    current_time = time.time()
    if motion_detected and (current_time - last_alert > cooldown):
        winsound.Beep(800, 150)
        last_alert = current_time

    if motion_detected:
        out.write(frame)

    cv2.imshow("Security Feed", frame)
    cv2.imshow("Motion Mask", mask)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()