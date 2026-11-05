
from pathlib import Path
import joblib
from sklearn.datasets import load_iris


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model.joblib"

model = joblib.load(MODEL_PATH)
species_names = load_iris().target_names


def predict(features):
    # features = [sepal length, sepal width, petal length, petal width]
    # predict() expects a list of samples, so one flower goes in double brackets
    label = model.predict([features])[0]
    return species_names[label]


if __name__ == "__main__":
    print(predict([5.1, 3.5, 1.4, 0.2]))
