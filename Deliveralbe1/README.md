# SUML

## Deliverable 1 - Iris classifier

`Deliveralbe1/D1.py` loads the Iris dataset, trains a logistic regression
classifier, evaluates it, saves it to `model.joblib`, and reloads it to check
that the predictions are identical.

### Requirements

- Python 3.10 or newer
- Packages listed in `requirements.txt` (numpy, scikit-learn, joblib)

### Run

```bash
git clone <repo-url>
cd SUML

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
python Deliveralbe1/D1.py
```

The script needs no external data (the dataset ships with scikit-learn).
It writes `model.joblib` to the directory you run it from.
