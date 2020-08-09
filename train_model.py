from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

from pyimagesearch.nn.conv import LeNet
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataser of faces")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []

for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    # load the image. pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)  # resize it into a fixed input size of 28x28
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

# scale the rawl pixels intensities to the range [0,1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# converting labels to one hot encoding
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 2)

# Since there are uneven number of samples for each class, our model might end up training hard on a particular class. To avoid this we can handle this issue by
#computing the class weights

#accounting for the skew in the labeled data

classTotals= labels.sum(axis=0)                 #returns [9474, 3690] array for not-smiling and smiling respectively
classWeight= classTotals.max()/ classTotals     #retuns [1, 2.56]

#now given the weights, we amplify the per-instance loss by a larger weight when seeing "smiling" examples

(trainX, testX, trainY, testY)= train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

#model initialilization

print("[INFO] compiling model...")
model= LeNet.build( width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer= "adam", metrics= ["accuracy"])

#training the network
print("[INFO] training network..")
H = model.fit(trainX, trainY, validation_data= (testX, testY), class_weight= classWeight, batch_size=64, epochs =15, verbose=1)


#evaluate the network
print("[INFO} evaluating network")
predictions= model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

#saving model
print("[INFO] serializing network...")
model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,15), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0,15), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch#")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()





