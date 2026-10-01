# Example 4.11
# Compare VGG16 and VGG19

import time
import numpy as np

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input as preprocess_vgg16
)

from tensorflow.keras.applications.vgg19 import (
    VGG19,
    preprocess_input as preprocess_vgg19
)

from tensorflow.keras.applications.imagenet_utils import decode_predictions


# ==========================================
# Load Image
# ==========================================

img_path = 'Elephant.jpg'

img = image.load_img(
    img_path,
    target_size=(224, 224)
)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


# ==========================================
# VGG16
# ==========================================

print("\n====================================")
print("VGG16")
print("====================================")

model_vgg16 = VGG16(weights='imagenet')

x_vgg16 = preprocess_vgg16(x.copy())

start_time = time.time()

predictions_vgg16 = model_vgg16.predict(
    x_vgg16,
    verbose=0
)

time_vgg16 = time.time() - start_time

results_vgg16 = decode_predictions(
    predictions_vgg16,
    top=5
)

print("\nTop 5 Predictions VGG16:")

for result in results_vgg16[0]:
    print(
        result[1],
        "-",
        result[2] * 100,
        "%"
    )

print("\nPrediction Time:",
      time_vgg16,
      "seconds")


# ==========================================
# VGG19
# ==========================================

print("\n====================================")
print("VGG19")
print("====================================")

model_vgg19 = VGG19(weights='imagenet')

x_vgg19 = preprocess_vgg19(x.copy())

start_time = time.time()

predictions_vgg19 = model_vgg19.predict(
    x_vgg19,
    verbose=0
)

time_vgg19 = time.time() - start_time

results_vgg19 = decode_predictions(
    predictions_vgg19,
    top=5
)

print("\nTop 5 Predictions VGG19:")

for result in results_vgg19[0]:
    print(
        result[1],
        "-",
        result[2] * 100,
        "%"
    )

print("\nPrediction Time:",
      time_vgg19,
      "seconds")


# ==========================================
# COMPARISON
# ==========================================

print("\n====================================")
print("PERFORMANCE COMPARISON")
print("====================================")

print("\nVGG16")
print("Prediction Time:",
      time_vgg16,
      "seconds")

print("\nVGG19")
print("Prediction Time:",
      time_vgg19,
      "seconds")