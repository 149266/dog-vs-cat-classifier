from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# -- Last inn modellen --
model = tf.keras.models.load_model('model.h5')

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((64, 64))  
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# -- Dette er ruten som viser nettsiden --
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# -- Dette er API-endepunktet som håndterer bildeopplastingen --
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    image_bytes = file.read()
    img = preprocess_image(image_bytes)
    prediction = model.predict(img)
    label = 'Dog' if prediction[0] > 0.5 else 'Cat'

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
