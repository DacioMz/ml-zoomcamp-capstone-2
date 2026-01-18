import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"
    ]

    df = pd.read_csv(
        url,
        header=None,
        names=columns,
        na_values="?",
        skipinitialspace=True
    )

    df = df.dropna()
    df["income"] = (df["income"] == ">50K").astype(int)

    return df


def train_model(df):
    X = df.drop("income", axis=1)
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    random_state=42
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f"ROC AUC on test set: {auc:.4f}")

    return model


def save_model(model, filename="model.bin"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def main():
    df = load_data()
    model = train_model(df)
    save_model(model)


if __name__ == "__main__":
    main()
