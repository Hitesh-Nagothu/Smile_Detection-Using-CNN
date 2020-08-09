from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", help="path to where teh face cascade resides")
ap.add_argument("-m", "--model", required = True, help="path to pretrained model")
ap.add_argument("-v", "--video", help="path to video file")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # loading the HAAR cascade face detector and the Lenet model
model = load_model(args["model"])

if not args.get("video", False):    camera = cv2.VideoCapture(0)

else:
    camera = cv2.VideoCapture(args["video"])  # camera is a pointer to either the webcam or the video provided

while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are reading a video and not grabbed any frame then we have reached the end of video
    if args.get("video") and not grabbed:
        break

    # Now we resize the frame, convert it to grayscale and then clone the original fram so we can draw on it later in
    # the program

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    # detect the faces in the input frame, then clone the frame to draw on it

    # detectMultiScale method handles detecting the bounding box (x,y)-coordinates of faces in the frame:

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the dace from the grayscale image, resize it to a fixed 28x28 pixels and then prepare
        # the ROI for classification via the CNN

        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        (notSmiling, smiling) = model.predict(roi)[0]

        label = "Smiling" if smiling > notSmiling else "Not Smiling"  # compares the probabilities returend by the
        # model predict

        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        cv2.imshow("Face", frameClone)

        # exit loop if q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
