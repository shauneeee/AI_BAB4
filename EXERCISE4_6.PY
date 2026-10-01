# Example 4.7 - Modified
# Add another Conv2D, MaxPooling2D, and Dropout layer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Create model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(
    32,
    (5, 5),
    input_shape=(28, 28, 1),
    activation='relu'
))

# 1st Max Pooling Layer
model.add(MaxPooling2D(
    pool_size=(2, 2)
))

# 1st Dropout Layer
model.add(Dropout(0.2))

# 2nd Convolutional Layer
model.add(Conv2D(
    32,
    (5, 5),
    activation='relu'
))

# 2nd Max Pooling Layer
model.add(MaxPooling2D(
    pool_size=(2, 2)
))

# 2nd Dropout Layer
model.add(Dropout(0.2))

# Flatten
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(
    128,
    activation='relu'
))

# Output Layer
model.add(Dense(
    2,
    activation='softmax'
))

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Display model summary
print(model.summary())