import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

MODEL_PATH = "model.joblib"


print("\n")
print("STEP 1: Load and explore the dataset")

iris = load_iris()
X = iris.data
y = iris.target
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
print(f"Number of samples:  {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names:      {list(iris.feature_names)}")
print(f"Target classes:     {iris.target_names.tolist()}")



print("\n")
print("STEP 2: Train, test split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

print("\n")
print("STEP 3: Train the classifier")

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Training finished")
print(f"Learned coefficients shape: {model.coef_.shape} (3 classes x 4 features)")

print("\n")
print("STEP 4: Predictions on the test set")

y_pred = model.predict(X_test)

print("Actual vs predicted 10:")
for i in range(10):
    print(iris.target_names[y_test[i]], "->", iris.target_names[y_pred[i]])


print("\n")
print("STEP 5: Evaluation")


accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy:.0%})")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix (rows = actual, columns = predicted):")
label_width = len("Actual / Predicted")
column_width = max(12, max(len(name) for name in iris.target_names))
header = f"{'Actual / Predicted':<{label_width}}" + "".join(
    f"{name:>{column_width}}" for name in iris.target_names
)
print(header)
for name, row in zip(iris.target_names, cm):
    print(f"{name:<{label_width}}" + "".join(f"{v:>{column_width}}" for v in row))

correct = np.trace(cm)
incorrect = cm.sum() - correct

print(f"\nCorrect predictions:   {correct}")
print(f"Incorrect predictions: {incorrect}")
print("\n")

confusions = [
    f"{cm[i, j]} x {iris.target_names[i]} predicted as {iris.target_names[j]}"
    for i in range(len(cm))
    for j in range(len(cm))
    if i != j and cm[i, j] > 0
]

if confusions:
    print("Confused classes:")
    for c in confusions:
        print("  -", c)
else:
    print("No classes were confused with each other.")
print("\n")

print("STEP 6: Save the model")

joblib.dump(model, MODEL_PATH)

if os.path.exists(MODEL_PATH):
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"Model saved to '{MODEL_PATH}'")
else:
    raise FileNotFoundError(f"'{MODEL_PATH}' was not created!")
print("\n")

print("STEP 7: Load and reuse the model")

loaded_model = joblib.load(MODEL_PATH)
loaded_pred = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, loaded_pred)

print(f"Original accuracy: {accuracy:.4f}")
print(f"Loaded accuracy:   {loaded_accuracy:.4f}")


identical = np.array_equal(y_pred, loaded_pred)
print(f"Are prediction identical ?   {identical}")