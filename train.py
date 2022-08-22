#%%
"""
Simple training script to train a KNN model for the iris dataset
"""

# Load dataset
from sklearn.datasets import load_iris

# Train model
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import accuracy_score

# I/O
import joblib

#%%

# Load datset
X, y = load_iris(return_X_y=True)

# Setup model
knn = KNeighborsClassifier()

# Train model
knn.fit(X, y)

# Test it works
y_pred = knn.predict(X)

print(accuracy_score(y, y_pred).round(2))

#%%
# Write model to disk
joblib.dump(knn, "model.pkl")
