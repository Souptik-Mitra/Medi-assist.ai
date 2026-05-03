import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_medical_symptoms_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "symptom_model.pkl")

FEATURE_COLS = [
    "age",
    "gender",
    "fever",
    "cough",
    "fatigue",
    "headache",
    "muscle_pain",
    "nausea",
    "vomiting",
    "diarrhea",
    "skin_rash",
    "loss_smell",
    "loss_taste",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "temperature_c",
    "oxygen_saturation",
    "wbc_count",
    "hemoglobin",
    "platelet_count",
    "crp_level",
    "glucose_level",
]


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURE_COLS + ["diagnosis"] if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    X = df[FEATURE_COLS]
    y = df["diagnosis"]

    numeric_features = [c for c in FEATURE_COLS if c != "gender"]
    categorical_features = ["gender"]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    clf = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Training RandomForest...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    medians = X.median(numeric_only=True).to_dict()
    medians["age"] = float(X["age"].median())
    mode_gender = str(X["gender"].mode().iloc[0])

    bundle = {
        "pipeline": clf,
        "feature_cols": FEATURE_COLS,
        "medians": medians,
        "mode_gender": mode_gender,
        "target_name": "diagnosis",
    }
    joblib.dump(bundle, MODEL_PATH)
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
