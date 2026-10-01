# Example 4.2 Multiple Layer Perceptron
# 3 Inputs - Logical OR

from sklearn.neural_network import MLPClassifier

# Input: x1, x2, x3
X = [
    [0., 0., 0.],
    [0., 0., 1.],
    [0., 1., 0.],
    [0., 1., 1.],
    [1., 0., 0.],
    [1., 0., 1.],
    [1., 1., 0.],
    [1., 1., 1.]
]

# Output OR 3 input
y = [0, 1, 1, 1, 1, 1, 1, 1]

clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    random_state=1
)

# Training
clf.fit(X, y)

# Testing
print("Prediction:")
print(clf.predict([
    [0., 0., 0.],
    [1., 1., 1.],
    [1., 0., 0.],
    [0., 0., 1.]
]))

# Melihat bentuk weight setiap layer
print("Shape of coefficients:")
print([coef.shape for coef in clf.coefs_])