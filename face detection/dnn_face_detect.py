import cv2
from playsound import playsound
import time

# Load the DNN model
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Open the webcam
cap = cv2.VideoCapture(0)

# Set a flag to play sound only once per detection
last_detection_time = 0
cooldown = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the image for DNN
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_detected = False

    # Process detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  # Threshold
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_detected = True

    # Play sound once per face detection
    if face_detected and (time.time() - last_detection_time > cooldown):
        playsound("ding.mp3")  # Make sure you have this file in your folder
        last_detection_time = time.time()

    cv2.imshow("DNN Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
