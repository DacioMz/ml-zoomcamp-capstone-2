import pickle
import pandas as pd

from flask import Flask, request, jsonify

MODEL_PATH = "model.bin"

app = Flask(__name__)


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    df = pd.DataFrame([customer])
    proba = model.predict_proba(df)[0, 1]

    result = {
        "income_probability": float(proba),
        "income_gt_50k": bool(proba >= 0.5)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
