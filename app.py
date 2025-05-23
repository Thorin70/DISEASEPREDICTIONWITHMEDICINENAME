from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model & label encoder
model = joblib.load("disease_prediction_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset to extract symptom names and disease-medicine mapping
df = pd.read_csv("Training_with_medicine.csv")

# Extract all symptoms from the dataset (excluding last two columns: "prognosis" & "medicine")
symptoms = list(df.columns[:-2])

# Define minimum symptoms required for prediction
MIN_SYMPTOMS_REQUIRED = 5  # Must select exactly 5 symptoms

@app.route("/")
def home():
    return render_template("index.html", symptoms=symptoms)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Receive selected symptoms as JSON
        selected_symptoms = {symptom: 1 if data.get(symptom) == "1" else 0 for symptom in symptoms}

        # Count the number of selected symptoms
        selected_count = sum(selected_symptoms.values())

        # Ensure exactly 5 symptoms are selected
        if selected_count != MIN_SYMPTOMS_REQUIRED:
            return jsonify({"error": f"Please select exactly {MIN_SYMPTOMS_REQUIRED} symptoms for accurate prediction."})

        # Convert to DataFrame
        input_data = pd.DataFrame([selected_symptoms])

        # Make a prediction
        prediction = model.predict(input_data)
        disease = label_encoder.inverse_transform(prediction)[0]

        # Retrieve the medicine for the predicted disease
        medicine_list = df[df['prognosis'] == disease]['medicine'].values

        if len(medicine_list) > 0:
            medicine = medicine_list[0]  # Get the first medicine (modify if multiple exist)
        else:
            medicine = "No medicine found in database."

        return jsonify({"disease": disease, "medicine": medicine})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
