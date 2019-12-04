import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 1280)

cone = cv2.imread("cone.png", -1)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        print("Face at x{} y{} w{} h{}".format(x, y, w, h))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, "CONE", (x, int(y+h/2)), cv2.FONT_HERSHEY_DUPLEX, (h/80), (0, 100, 250), 2)  # Blue, green, red

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
