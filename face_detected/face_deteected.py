import cv2
import os
import urllib.request

cascade_file = "haarcascade_frontalface_default.xml"

if not os.path.exists(cascade_file):
    print("Downloading cascade file...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_file)
    print("Downloaded successfully!")

face_cascade = cv2.CascadeClassifier(cascade_file)

if face_cascade.empty():
    print("Error: Cascade classifier failed to load.")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Press 'q' or 'ESC' to quit")

img_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    output_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    count = len(faces)

    cv2.putText(output_frame, f"Faces Detected: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", output_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:  # Press 'q' or 'ESC' to quit
        break
    elif key == ord('s'):  # Press 's' to save the captured frame
        filename = f"detected_face_{img_counter}.png"
        cv2.imwrite(filename, output_frame)
        print(f"Saved frame: {filename}")
        img_counter += 1

    if cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("Video stream closed and resources released.")