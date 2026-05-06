import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
import os

# 1. Load or create a dataset for this project
print("Fetching MNIST dataset...")
# Using a subset for speed in this script example
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
X = X / 255.0  # Normalize

# Using a smaller subset to train faster for demonstration
X_train, X_test, y_train, y_test = train_test_split(X[:10000], y[:10000], test_size=0.2, random_state=42)

# 2. Train a scikit-learn model
print("Training MLP Classifier (scikit-learn)...")
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=20, alpha=1e-4,
                      solver='sgd', verbose=10, random_state=1,
                      learning_rate_init=.1)

model.fit(X_train, y_train)

# 3. Save the trained model as model.pkl using joblib
print("Saving model to model.pkl...")
joblib.dump(model, 'model.pkl')

# 4. Prints accuracy at the end
score = model.score(X_test, y_test)
print(f"Training Complete. Accuracy: {score:.4f}")
