# Example 4.9 - Comparison of CNN Performance

import matplotlib.pyplot as plt
import time

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical


# ==========================================
# Load MNIST Dataset
# ==========================================

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape
X_train = X_train.reshape(
    X_train.shape[0], 28, 28, 1
).astype('float32')

X_test = X_test.reshape(
    X_test.shape[0], 28, 28, 1
).astype('float32')

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]


# ==========================================
# Function to Create CNN Model
# ==========================================

def create_model(filters, kernel_size):

    model = Sequential()

    model.add(
        Conv2D(
            filters,
            kernel_size,
            input_shape=(28, 28, 1),
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(pool_size=(2, 2))
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Flatten()
    )

    model.add(
        Dense(128, activation='relu')
    )

    model.add(
        Dense(num_classes, activation='softmax')
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


# ==========================================
# Experiment 1
# Original: 32 filters, kernel (5,5)
# ==========================================

print("\n===== EXPERIMENT 1 =====")
print("32 filters, kernel (5,5)")

model1 = create_model(32, (5, 5))

start_time = time.time()

history1 = model1.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
    verbose=1
)

time1 = time.time() - start_time

score1 = model1.evaluate(
    X_test,
    y_test,
    verbose=0
)

accuracy1 = score1[1]
error1 = 1 - accuracy1

print("Accuracy :", accuracy1 * 100, "%")
print("Error    :", error1 * 100, "%")
print("Time     :", time1, "seconds")


# ==========================================
# Experiment 2
# 64 filters, kernel (5,5)
# ==========================================

print("\n===== EXPERIMENT 2 =====")
print("64 filters, kernel (5,5)")

model2 = create_model(64, (5, 5))

start_time = time.time()

history2 = model2.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
    verbose=1
)

time2 = time.time() - start_time

score2 = model2.evaluate(
    X_test,
    y_test,
    verbose=0
)

accuracy2 = score2[1]
error2 = 1 - accuracy2

print("Accuracy :", accuracy2 * 100, "%")
print("Error    :", error2 * 100, "%")
print("Time     :", time2, "seconds")


# ==========================================
# Experiment 3
# 32 filters, kernel (3,3)
# ==========================================

print("\n===== EXPERIMENT 3 =====")
print("32 filters, kernel (3,3)")

model3 = create_model(32, (3, 3))

start_time = time.time()

history3 = model3.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
    verbose=1
)

time3 = time.time() - start_time

score3 = model3.evaluate(
    X_test,
    y_test,
    verbose=0
)

accuracy3 = score3[1]
error3 = 1 - accuracy3

print("Accuracy :", accuracy3 * 100, "%")
print("Error    :", error3 * 100, "%")
print("Time     :", time3, "seconds")


# ==========================================
# Comparison
# ==========================================

print("\n====================================")
print("PERFORMANCE COMPARISON")
print("====================================")

print("\n32 Filters + (5,5)")
print("Accuracy :", accuracy1 * 100, "%")
print("Error    :", error1 * 100, "%")
print("Time     :", time1, "seconds")

print("\n64 Filters + (5,5)")
print("Accuracy :", accuracy2 * 100, "%")
print("Error    :", error2 * 100, "%")
print("Time     :", time2, "seconds")

print("\n32 Filters + (3,3)")
print("Accuracy :", accuracy3 * 100, "%")
print("Error    :", error3 * 100, "%")
print("Time     :", time3, "seconds")