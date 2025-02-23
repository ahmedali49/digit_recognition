from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "digit_recognition_model.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")

def preprocess_image(image):
    """Preprocess the image to match the input format of the model."""
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)
    image = cv2.bitwise_not(image)  # Invert colors (Black -> White)
    image = image.astype('float32') / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for model
    return image

@app.route("/", methods=["GET"])
def index():
    """Render the main webpage."""
    return render_template("index.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    """Predict the digit from an uploaded image."""
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_label = np.argmax(prediction)
        return jsonify({"prediction": int(predicted_label)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_canvas", methods=["POST"])
def predict_canvas():
    """Predict the digit from a canvas drawing."""
    try:
        data = request.json.get("image", "")
        if not data:
            return jsonify({"error": "No image data received"}), 400

        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(io.BytesIO(image_data))

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_label = np.argmax(prediction)

        return jsonify({"prediction": int(predicted_label)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
