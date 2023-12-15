import os
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import tensorflow as tf
from keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['ALLOWED_EXTENSION'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['LABELS_FILE'] = 'labels.txt'
app.config['MODEL_FILE'] = 'new_anime_classification.h5'

def allowed_file(filename):
    return '.' in filename and \
        filename.split('.', 1)[1] in app.config['ALLOWED_EXTENSION']

# Load the trained model
model = load_model(app.config['MODEL_FILE'], compile=False)
with open(app.config['LABELS_FILE'], 'r') as file:
    labels = file.read().splitlines()

def predict_anime_classification(image):
    # Pre-processing input image
    img = Image.open(image).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0) # (128, 128, 3)
    normalized_img_array = (img_array.astype(np.float32) / 255.0)


    # Predicting the image
    predictions = model.predict(normalized_img_array)
    index = np.argmax(predictions)
    class_name = labels[index]
    confidence_score = predictions[0][index]

    return class_name[3:], confidence_score

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Welcome to Anime Classification API",
        },
        "data": None
    }), 200

@app.route("/prediction", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            # Save input image
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            class_name, confidence_score = predict_anime_classification(image_path)
            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success Predicting",
                },
                "data": {
                    "anime_classification": class_name,
                    "confidence": float(confidence_score)
                },
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Invalid file format. Please upload a PNG, JPG, or JPEG image",
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method Not Allowed",
            },
            "data": None,
        }), 405

if __name__ == '__main__':
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))