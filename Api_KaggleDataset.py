# Install necessary libraries
!pip install flask tensorflow pillow


# Import Libraries
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os


# Load Trained Model
model = load_model('flower_model.h5')
class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # Update as per dataset classes


# Initialize Flask App
app = Flask(__name__)

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Preprocess Image
    img = load_img(file, target_size=(150, 150))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction[0])]
    
    return jsonify({'prediction': predicted_class})


# Run Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

