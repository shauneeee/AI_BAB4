# Example 4.15
# Compare Different CNN Models using Webcam

import cv2
import numpy as np
import time

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
# 1. Load Models
# ============================================================

print("Loading VGG16...")
model_vgg16 = VGG16(
    weights='imagenet'
)

print("Loading ResNet50...")
model_resnet50 = ResNet50(
    weights='imagenet'
)

print("Loading MobileNetV2...")
model_mobilenetv2 = MobileNetV2(
    weights='imagenet'
)

print("Loading DenseNet201...")
model_densenet201 = DenseNet201(
    weights='imagenet'
)


# ============================================================
# 2. Model List
# ============================================================

models = [
    ("VGG16", model_vgg16, preprocess_vgg16),
    ("ResNet50", model_resnet50, preprocess_resnet50),
    ("MobileNetV2", model_mobilenetv2, preprocess_mobilenetv2),
    ("DenseNet201", model_densenet201, preprocess_densenet201)
]


# ============================================================
# 3. Model Summary
# ============================================================

for name, model, preprocess in models:

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.summary()


# ============================================================
# 4. Open Webcam
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera cannot be opened.")
    exit()


# ============================================================
# 5. Webcam Classification
# ============================================================

while camera.isOpened():

    ret, cam_frame = camera.read()

    if not ret:
        print("Cannot read camera frame.")
        break

    results = []

    # --------------------------------------------------------
    # Predict using every model
    # --------------------------------------------------------

    for name, model, preprocess in models:

        # Resize webcam frame
        frame = cv2.resize(
            cam_frame,
            (224, 224)
        )

        # Convert to NumPy array
        image = np.asarray(frame)

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )

        # Preprocess image
        image = preprocess(image)

        # Measure prediction time
        start_time = time.time()

        predictions = model.predict(
            image,
            verbose=0
        )

        end_time = time.time()

        prediction_time = end_time - start_time

        # Decode prediction
        label = decode_predictions(
            predictions,
            top=1
        )

        class_name = label[0][0][1]
        probability = label[0][0][2]

        results.append(
            (
                name,
                class_name,
                probability,
                prediction_time
            )
        )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    y_position = 30

    for name, class_name, probability, prediction_time in results:

        text = "{}: {} {:.1f}% | {:.3f}s".format(
            name,
            class_name,
            probability * 100,
            prediction_time
        )

        cv2.putText(
            cam_frame,
            text,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        y_position += 30


    # Show webcam
    cv2.imshow(
        "CNN Model Comparison",
        cam_frame
    )


    # ESC to exit
    key = cv2.waitKey(1)

    if key == 27:
        break


# ============================================================
# 6. Release Camera
# ============================================================

camera.release()
cv2.destroyAllWindows()