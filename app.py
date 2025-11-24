from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
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

# ---------------- Web Routes ----------------
@app.route('/')
def home():
    return render_template("index.html")

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

    try:
        prediction = predict_production(state, district, season, crop, area)
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

    return render_template("index.html", prediction_text=f"Predicted Production: {prediction:.2f}")

# ---------------- API Route ----------------
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    state = data.get("State_Name")
    district = data.get("District_Name")
    season = data.get("Season")
    crop = data.get("Crop")
    area = data.get("Area")

    try:
        area = float(area)
    except:
        return jsonify({"error": "Invalid Area value!"}), 400

    try:
        prediction = predict_production(state, district, season, crop, area)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"Predicted_Production": round(prediction, 2)})

# ---------------- Run App ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
