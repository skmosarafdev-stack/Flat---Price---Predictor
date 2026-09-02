from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Model load
model_path = Path(__file__).parent / "model" / "flat_price_model.joblib"
model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    values = {}
    if request.method == "POST":
        try:
            values = {
                "Area_sqft": float(request.form["area"]),
                "Facing": request.form["facing"],
                "Floor": int(request.form["floor"]),
                "Parking_sqft": float(request.form["parking"]),
                "Bedrooms": int(request.form["bedrooms"])
            }
            df = pd.DataFrame([values])
            prediction = float(model.predict(df)[0])
        except Exception as e:
            error = f"Please enter valid details: {e}"
    return render_template("index.html", prediction=prediction, error=error, values=values)

@app.route("/batch", methods=["GET", "POST"])
def batch_predict():
    results = None
    error = None
    if request.method == "POST":
        try:
            file = request.files["file"]
            df = pd.read_csv(file)

            required = ["Area_sqft", "Facing", "Floor", "Parking_sqft", "Bedrooms"]
            for col in required:
                if col not in df.columns:
                    raise ValueError(f"CSV must have column: {col}")

            preds = model.predict(df)
            df["Predicted_Price_Lakhs"] = preds
            results = df.to_dict(orient="records")
        except Exception as e:
            error = str(e)
    return render_template("batch.html", results=results, error=error)

if __name__ == "__main__":
    app.run(debug=True)