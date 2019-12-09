import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 3840)

cone = cv2.imread("cone.png", cv2.IMREAD_COLOR)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        print("Face at x{} y{} w{} h{}".format(x, y, w, h))
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        mode = 1

        if mode == 0:

            # Add the additional faces
            try:  # Avoid going out of screen bounds
                frame[y:y+h, x-h:x] = frame[y:y+h, x:x+h]
            except:
                cv2.putText(frame, "OUT OF BOUNDS LEFT!".format(x, y, h), (x, y-20), cv2.FONT_HERSHEY_DUPLEX, (h / 370),
                            (0, 0, 255), 2)  # Blue, green, red
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            try:  # Avoid going out of screen bounds
                frame[y:y+h, x+h:x+2*h] = frame[y:y+h, x:x+h]
            except:
                cv2.putText(frame, "OUT OF BOUNDS RIGHT!".format(x, y, h), (x, y+h+30), cv2.FONT_HERSHEY_DUPLEX, (h / 370),
                            (0, 0, 255), 2)  # Blue, green, red
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        elif mode == 1:

            cone_small = cv2.resize(cone, (w, h))
            frame[y:y+h, x:x+h] = cone_small[0:w, 0:h]  # Put the cone image over the faces
            cv2.putText(frame, "CONE", (x, y), cv2.FONT_HERSHEY_DUPLEX, (h / 85), (0, 100, 250), 2)  # Blue, green, red
            cv2.putText(frame, "(x{}, y{}, h{})".format(x, y, h), (x, y + h - 10), cv2.FONT_HERSHEY_DUPLEX, (h / 320), (255, 255, 255), 2)  # Blue, green, red

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
