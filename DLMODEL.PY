# Example 4.7

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Create model
model = Sequential()

# Convolutional Layer
model.add(Conv2D(
    32,
    (5, 5),
    input_shape=(28, 28, 1),
    activation='relu'
))

# Max Pooling Layer
model.add(MaxPooling2D(
    pool_size=(2, 2)
))

# Dropout Layer
model.add(Dropout(0.2))

# Flatten Layer
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