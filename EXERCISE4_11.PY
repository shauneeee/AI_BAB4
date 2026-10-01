# Example 4.13
# Compare VGG16 and VGG19 using Webcam

import cv2
import numpy as np
import time

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications.imagenet_utils import decode_predictions


# Image size
image_size = 224


# ==========================================
# Load VGG16 and VGG19
# ==========================================

print("Loading VGG16...")
model_vgg16 = vgg16.VGG16(weights='imagenet')

print("Loading VGG19...")
model_vgg19 = vgg19.VGG19(weights='imagenet')


# ==========================================
# Open Webcam
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera cannot be opened.")
    exit()


while camera.isOpened():

    # Read webcam frame
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


    # ======================================
    # VGG16 Prediction
    # ======================================

    processed_vgg16 = vgg16.preprocess_input(
        image_batch.copy()
    )

    start_vgg16 = time.time()

    predictions_vgg16 = model_vgg16.predict(
        processed_vgg16,
        verbose=0
    )

    time_vgg16 = time.time() - start_vgg16

    label_vgg16 = decode_predictions(
        predictions_vgg16,
        top=1
    )

    class_vgg16 = label_vgg16[0][0][1]
    probability_vgg16 = label_vgg16[0][0][2]


    # ======================================
    # VGG19 Prediction
    # ======================================

    processed_vgg19 = vgg19.preprocess_input(
        image_batch.copy()
    )

    start_vgg19 = time.time()

    predictions_vgg19 = model_vgg19.predict(
        processed_vgg19,
        verbose=0
    )

    time_vgg19 = time.time() - start_vgg19

    label_vgg19 = decode_predictions(
        predictions_vgg19,
        top=1
    )

    class_vgg19 = label_vgg19[0][0][1]
    probability_vgg19 = label_vgg19[0][0][2]


    # ======================================
    # Display VGG16 Result
    # ======================================

    text_vgg16 = "VGG16: {} ({:.1f}%)".format(
        class_vgg16,
        probability_vgg16 * 100
    )

    cv2.putText(
        cam_frame,
        text_vgg16,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


    # ======================================
    # Display VGG19 Result
    # ======================================

    text_vgg19 = "VGG19: {} ({:.1f}%)".format(
        class_vgg19,
        probability_vgg19 * 100
    )

    cv2.putText(
        cam_frame,
        text_vgg19,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ======================================
    # Display Prediction Time
    # ======================================

    time_text = "Time - VGG16: {:.3f}s | VGG19: {:.3f}s".format(
        time_vgg16,
        time_vgg19
    )

    cv2.putText(
        cam_frame,
        time_text,
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )


    # Show webcam
    cv2.imshow(
        "VGG16 vs VGG19",
        cam_frame
    )


    # Press ESC to quit
    key = cv2.waitKey(1)

    if key == 27:
        break


# ==========================================
# Release Camera
# ==========================================

camera.release()
cv2.destroyAllWindows()