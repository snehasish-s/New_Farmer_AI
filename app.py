from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import io
import requests  # <-- Added for weather API

app = Flask(__name__)

# Load models
crop_health_model = tf.keras.models.load_model("models/crop_health_model.keras")
yield_model = tf.keras.models.load_model("models/yield_model.keras")

# Class labels for crop health
class_labels = {
    0: "Healthy",
    1: "Nutrient Deficient",
    2: "Leaf Blight",
    3: "Rust",
    4: "Other Disease"
}


    

# ==============================
# Routes
# ==============================
@app.route("/")
def home():
      
    return render_template("index.html")

@app.route("/crophealth")
def crophealth_page():
    return render_template("crophealth.html")

@app.route("/yieldprediction")
def yieldprediction_page():
    return render_template("yieldprediction.html")

@app.route("/irrigation")
def irrigation_page():
    return render_template("irrigation.html") 

# ==============================
# Crop Health Prediction
# ==============================
@app.route("/predict_model1", methods=["POST"])
def predict_model1():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    try:
        img = tf.keras.utils.load_img(io.BytesIO(file.read()), target_size=(128, 128))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = crop_health_model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100

        result = {
            "label": class_labels[class_idx],
            "status": "Healthy" if class_idx == 0 else "Unhealthy",
            "accuracy": f"{confidence:.2f}%"
        }

        print("DEBUG RESULT:", result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# Yield Prediction (Form input)
# ==============================
@app.route("/predict_yield", methods=["POST"])
def predict_yield():
    try:
        rainfall = float(request.form.get("rainfall", 0))
        temperature = float(request.form.get("temperature", 0))
        humidity = float(request.form.get("humidity", 0))

        features = np.array([[rainfall, temperature, humidity]])
        prediction = yield_model.predict(features)
        yield_value = float(prediction[0][0])

        result = {"predicted_yield": f"{yield_value:.2f} kg/ha"}
        print("DEBUG RESULT:", result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict_model2", methods=["POST"])
def predict_model2():
    try:
        data = request.get_json()
        rainfall = float(data["rainfall"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])

        input_data = np.array([[rainfall, temperature, humidity]])
        prediction = yield_model.predict(input_data)
        predicted_yield = float(prediction[0][0])

        return jsonify({
            "yield": f"{predicted_yield:.2f} kg/ha"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# Run Flask
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
