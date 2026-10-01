# Example 4.14 - Modified
# Webcam Image Classification using ResNet50 and DenseNet121

import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions

from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import densenet


# ==========================================
# Image Size
# ==========================================

image_size = 224


# ==========================================
# Load Models
# ==========================================

print("Loading ResNet50...")
model_resnet = resnet50.ResNet50(weights='imagenet')

print("Loading DenseNet121...")
model_densenet = densenet.DenseNet121(weights='imagenet')


# ==========================================
# Open Webcam
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera cannot be opened.")
    exit()


# ==========================================
# Webcam Classification
# ==========================================

while camera.isOpened():

    # Read frame from webcam
    ok, cam_frame = camera.read()

    if not ok:
        print("Cannot read camera frame.")
        break


    # ======================================
    # Resize Image
    # ======================================

    frame = cv2.resize(
        cam_frame,
        (image_size, image_size)
    )


    # ======================================
    # Convert Image to Array
    # ======================================

    numpy_image = img_to_array(frame)

    image_batch = np.expand_dims(
        numpy_image,
        axis=0
    )


    # ======================================
    # ResNet50 Prediction
    # ======================================

    processed_resnet = resnet50.preprocess_input(
        image_batch.copy()
    )

    predictions_resnet = model_resnet.predict(
        processed_resnet,
        verbose=0
    )

    label_resnet = decode_predictions(
        predictions_resnet,
        top=1
    )

    class_resnet = label_resnet[0][0][1]
    probability_resnet = label_resnet[0][0][2]


    # ======================================
    # DenseNet121 Prediction
    # ======================================

    processed_densenet = densenet.preprocess_input(
        image_batch.copy()
    )

    predictions_densenet = model_densenet.predict(
        processed_densenet,
        verbose=0
    )

    label_densenet = decode_predictions(
        predictions_densenet,
        top=1
    )

    class_densenet = label_densenet[0][0][1]
    probability_densenet = label_densenet[0][0][2]


    # ======================================
    # Display ResNet50 Prediction
    # ======================================

    text_resnet = "ResNet50: {} ({:.1f}%)".format(
        class_resnet,
        probability_resnet * 100
    )

    cv2.putText(
        cam_frame,
        text_resnet,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


    # ======================================
    # Display DenseNet121 Prediction
    # ======================================

    text_densenet = "DenseNet121: {} ({:.1f}%)".format(
        class_densenet,
        probability_densenet * 100
    )

    cv2.putText(
        cam_frame,
        text_densenet,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ======================================
    # Display Webcam
    # ======================================

    cv2.imshow(
        "ResNet50 vs DenseNet121",
        cam_frame
    )


    # ======================================
    # Press ESC to Quit
    # ======================================

    key = cv2.waitKey(30)

    if key == 27:
        break


# ==========================================
# Release Camera
# ==========================================

camera.release()
cv2.destroyAllWindows()