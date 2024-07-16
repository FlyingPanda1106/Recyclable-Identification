from flask import Flask, render_template, request
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

# Path to the trained model
model_path = ''
model = load_model(model_path)

# Class names corresponding to the dataset
class_names = [
    'Cardboard', 'Compost', 'Glass', 'Metal',
    'Paper', 'Plastic', 'Trash'
]

# Recycling information dictionary (as provided earlier)

# Function to preprocess an image
def preprocess_image(img_path):
    try:
        img = cv.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image at path: {img_path}")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (244, 244))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error loading or processing image: {e}")
        return None

# Function to predict the item type and provide recycling information
def predict_item(img_path):
    img = preprocess_image(img_path)
    if img is not None:
        prediction = model.predict(img)
        index = np.argmax(prediction)
        predicted_label = class_names[index]
        confidence = prediction[0][index] * 100
        info = recycling_info[predicted_label]
        return predicted_label, confidence, info
    else:
        return None, None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            predicted_label, confidence, info = predict_item(file_path)
            os.remove(file_path)
            if predicted_label:
                return render_template('index.html', label=predicted_label, confidence=confidence, info=info, image=file.filename)
            else:
                return render_template('index.html', error="Image preprocessing failed.")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
