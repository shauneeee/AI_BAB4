# Example 4.14
# Webcam Image Classification using CNN Models
# Added DenseNet121

import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions

from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3,
    densenet
)


# ==========================================
# Initialize Model
# ==========================================

# VGG16
# model = vgg16.VGG16(weights='imagenet')
# preprocess = vgg16.preprocess_input

# ResNet50
model = resnet50.ResNet50(weights='imagenet')
preprocess = resnet50.preprocess_input

# MobileNet
# model = mobilenet.MobileNet(weights='imagenet')
# preprocess = mobilenet.preprocess_input

# InceptionV3
# model = inception_v3.InceptionV3(weights='imagenet')
# preprocess = inception_v3.preprocess_input

# DenseNet121
# model = densenet.DenseNet121(weights='imagenet')
# preprocess = densenet.preprocess_input


# Display model summary
print(model.summary())


# ==========================================
# Open Webcam
# ==========================================

camera = cv2.VideoCapture(0)

image_size = 224

# InceptionV3 uses 299 x 299
# image_size = 299


# ==========================================
# Webcam Classification
# ==========================================

while camera.isOpened():

    # Read frame from webcam
    ok, cam_frame = camera.read()

    if not ok:
        print("Cannot read camera frame.")
        break

    # Resize frame
    frame = cv2.resize(
        cam_frame,
        (image_size, image_size)
    )

    # Convert image to array
    numpy_image = img_to_array(frame)

    # Add batch dimension
    image_batch = np.expand_dims(
        numpy_image,
        axis=0
    )

    # Preprocess image
    processed_image = preprocess(
        image_batch.copy()
    )

    # Predict
    predictions = model.predict(
        processed_image,
        verbose=0
    )

    # Decode predictions
    label = decode_predictions(
        predictions,
        top=1
    )

    # Get predicted class
    class_name = label[0][0][1]
    probability = label[0][0][2]

    # Display prediction
    text = "{}, {:.1f}%".format(
        class_name,
        probability * 100
    )

    cv2.putText(
        cam_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Display webcam
    cv2.imshow(
        "Video Image",
        cam_frame
    )

    # Press ESC to quit
    key = cv2.waitKey(30)

    if key == 27:
        break


# ==========================================
# Release Camera
# ==========================================

camera.release()
cv2.destroyAllWindows()