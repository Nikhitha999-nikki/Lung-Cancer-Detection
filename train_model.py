import pickle, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset (30 features, target: 0=malignant, 1=benign)
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)
target_names = list(data.target_names)  # ['malignant','benign']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model = scaler + logistic regression
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])
pipe.fit(X_train, y_train)

# Report accuracy
acc = accuracy_score(y_test, pipe.predict(X_test))
print(f"✅ Test accuracy: {acc*100:.2f}%")

# Pick one known example from each class to help you test the UI
benign_idx = int(np.where(y == 1)[0][0])
malignant_idx = int(np.where(y == 0)[0][0])
examples = {
    "benign":  X[benign_idx].tolist(),
    "malignant": X[malignant_idx].tolist()
}

# Save everything together so Flask knows feature order & labels
artifact = {
    "model": pipe,
    "feature_names": feature_names,
    "target_names": target_names,
    "examples": examples
}
with open("model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("💾 Saved model.pkl with model + feature names + examples.")
