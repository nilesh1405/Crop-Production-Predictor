from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ---------------- Load model ----------------
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
pipeline = joblib.load(model_path)

# ---------------- Utility function ----------------
def clean_numeric(x):
    return pd.to_numeric(str(x).replace(",", ""), errors="coerce")

def predict_production(state, district, season, crop, area):
    df_input = pd.DataFrame([{
        "State_Name": state,
        "District_Name": district,
        "Season": season,
        "Crop": crop,
        "Area": area
    }])
    df_input["Area"] = clean_numeric(df_input["Area"])
    return float(pipeline.predict(df_input)[0])

# ---------------- Routes ----------------
@app.route('/')
def home():
    return render_template("index.html")  # Use correct folder: templates/index.html

@app.route('/predict', methods=['POST'])
def predict():
    state = request.form.get("State_Name")
    district = request.form.get("District_Name")
    season = request.form.get("Season")
    crop = request.form.get("Crop")
    area = request.form.get("Area")

    try:
        area = float(area)
    except:
        return render_template("index.html", prediction_text="Invalid Area value!")

    prediction = predict_production(state, district, season, crop, area)
    return render_template("index.html", prediction_text=f"Predicted Production: {prediction:.2f}")

# ---------------- Run app ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
