# Example 4.9
# Comparison: Original CNN vs CNN with Additional Conv2D Layer

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
# MODEL 1 - ORIGINAL EXAMPLE 4.9
# ==========================================

print("\n====================================")
print("MODEL 1 - ORIGINAL")
print("====================================")

model1 = Sequential()

model1.add(
    Conv2D(
        32,
        (5, 5),
        input_shape=(28, 28, 1),
        activation='relu'
    )
)

model1.add(
    MaxPooling2D(pool_size=(2, 2))
)

model1.add(
    Dropout(0.2)
)

model1.add(
    Flatten()
)

model1.add(
    Dense(128, activation='relu')
)

model1.add(
    Dense(num_classes, activation='softmax')
)

model1.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model1.summary()

# Training
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

# Evaluation
score1 = model1.evaluate(
    X_test,
    y_test,
    verbose=0
)

accuracy1 = score1[1]
error1 = 1 - accuracy1


# ==========================================
# MODEL 2 - MODIFIED
# ==========================================

print("\n====================================")
print("MODEL 2 - WITH ADDITIONAL LAYERS")
print("====================================")

model2 = Sequential()

# First Conv block
model2.add(
    Conv2D(
        32,
        (5, 5),
        input_shape=(28, 28, 1),
        activation='relu'
    )
)

model2.add(
    MaxPooling2D(pool_size=(2, 2))
)

model2.add(
    Dropout(0.2)
)

# Additional Conv block
model2.add(
    Conv2D(
        32,
        (5, 5),
        activation='relu'
    )
)

model2.add(
    MaxPooling2D(pool_size=(2, 2))
)

model2.add(
    Dropout(0.2)
)

# Flatten
model2.add(
    Flatten()
)

model2.add(
    Dense(128, activation='relu')
)

model2.add(
    Dense(num_classes, activation='softmax')
)

model2.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model2.summary()

# Training
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

# Evaluation
score2 = model2.evaluate(
    X_test,
    y_test,
    verbose=0
)

accuracy2 = score2[1]
error2 = 1 - accuracy2


# ==========================================
# PERFORMANCE COMPARISON
# ==========================================

print("\n====================================")
print("PERFORMANCE COMPARISON")
print("====================================")

print("\nMODEL 1 - ORIGINAL")
print("Accuracy :", accuracy1 * 100, "%")
print("Error    :", error1 * 100, "%")
print("Time     :", time1, "seconds")

print("\nMODEL 2 - ADDITIONAL CONV2D")
print("Accuracy :", accuracy2 * 100, "%")
print("Error    :", error2 * 100, "%")
print("Time     :", time2, "seconds")