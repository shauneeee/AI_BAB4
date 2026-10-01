# Example 4.15
# Image Classification using if-else
# Select Model for Classification

import cv2
import numpy as np

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import DenseNet201

from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet201

from tensorflow.keras.applications.imagenet_utils import decode_predictions


# ============================================================
# 1. Select Model
# ============================================================

choice = input(
    "Choose model (1=VGG16, 2=ResNet50, 3=MobileNetV2, 4=DenseNet201): "
)


# ============================================================
# 2. Select Model using IF-ELSE
# ============================================================

if choice == "1":

    print("Using VGG16...")

    model = VGG16(
        weights="imagenet"
    )

    preprocess_input = preprocess_vgg16

    model_name = "VGG16"


elif choice == "2":

    print("Using ResNet50...")

    model = ResNet50(
        weights="imagenet"
    )

    preprocess_input = preprocess_resnet50

    model_name = "ResNet50"


elif choice == "3":

    print("Using MobileNetV2...")

    model = MobileNetV2(
        weights="imagenet"
    )

    preprocess_input = preprocess_mobilenetv2

    model_name = "MobileNetV2"


elif choice == "4":

    print("Using DenseNet201...")

    model = DenseNet201(
        weights="imagenet"
    )

    preprocess_input = preprocess_densenet201

    model_name = "DenseNet201"


else:

    print("Invalid choice.")
    exit()


# ============================================================
# 3. Display Model Summary
# ============================================================

print("\nSelected model:", model_name)
print("\nModel Summary:")

model.summary()


# ============================================================
# 4. Open Webcam
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Camera cannot be opened.")
    exit()


# ============================================================
# 5. Image Classification
# ============================================================

while camera.isOpened():

    ret, cam_frame = camera.read()

    if not ret:

        print("Cannot read camera frame.")
        break


    # Resize image
    frame = cv2.resize(
        cam_frame,
        (224, 224)
    )


    # Convert image to NumPy array
    image = np.asarray(frame)


    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )


    # Preprocess image
    image = preprocess_input(image)


    # Predict
    predictions = model.predict(
        image,
        verbose=0
    )


    # Decode prediction
    label = decode_predictions(
        predictions,
        top=1
    )


    class_name = label[0][0][1]

    probability = label[0][0][2]


    # Display result
    text = "{}: {} {:.1f}%".format(
        model_name,
        class_name,
        probability * 100
    )


    cv2.putText(
        cam_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # Show webcam
    cv2.imshow(
        "Classification",
        cam_frame
    )


    # Press ESC to exit
    key = cv2.waitKey(1)

    if key == 27:
        break


# ============================================================
# 6. Release Camera
# ============================================================

camera.release()
cv2.destroyAllWindows()