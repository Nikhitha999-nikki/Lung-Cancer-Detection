from flask import Flask, render_template, request
import pickle, numpy as np

app = Flask(__name__)

# Load the full artifact (model + feature names + examples)
with open("model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
feature_names = artifact["feature_names"]  # exact order required by the model
target_names = artifact["target_names"]    # ['malignant','benign']
examples = artifact["examples"]            # dict with sample inputs

@app.route("/")
def home():
    # Prefill with a benign example so you can test instantly
    default_values = examples["benign"]
    return render_template(
        "index.html",
        feature_names=feature_names,
        default_values=default_values
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect features strictly in the same order used for training
        values = []
        for name in feature_names:
            v = float(request.form[name])
            values.append(v)
        X = np.array(values, dtype=float).reshape(1, -1)

        # Predict class and probabilities
        pred = int(model.predict(X)[0])  # 0=malignant, 1=benign
        proba = model.predict_proba(X)[0]  # [P(malignant), P(benign)]

        # Human-friendly interpretation
        pred_name = target_names[pred]           # 'malignant' or 'benign'
        if pred_name == "malignant":
            message = "Breast Cancer Detected"
        else:
            message = "No Breast Cancer Detected"

        return render_template(
            "result.html",
            message=message,
            raw_output=pred,
            pred_name=pred_name.capitalize(),
            p_malignant=f"{proba[0]*100:.1f}%",
            p_benign=f"{proba[1]*100:.1f}%"
        )
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
