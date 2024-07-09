from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import io
from PIL import Image
import json
from lib_save import read_save
improc = read_save()

app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the Image Processing API!"


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Extract parameters from the form data
    parameters = request.form.get('parameters')
    if parameters:
        parameters = json.loads(parameters)  # Convert JSON string to dict

    if file:
        # Read the image file
        img = Image.open(file.stream)
        img = np.array(img)
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Your custom processing logic
        processed_img, circle, line = improc.read_params(parameters, img, show= False)

        # Convert processed image to a format that can be sent back
        _, buffer = cv2.imencode('.jpg', processed_img["final"])
        img_io = io.BytesIO(buffer)

        return send_file(img_io, mimetype='image/jpeg')


# def custom_processing(img, parameters):
#     # Example processing: convert to grayscale if specified in parameters
#     if parameters and parameters.get('convert_to') == 'grayscale':
#         return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Add your custom processing logic here
#     # For example, you could apply a blur if specified in parameters
#     if parameters and parameters.get('blur'):
#         ksize = parameters['blur']
#         return cv2.GaussianBlur(img, (ksize, ksize), 0)
#
#     return img  # Return the original image if no processing is specified


if __name__ == '__main__':
    app.run(debug=True)

#curl -X POST http://127.0.0.1:5000/upload -F "file=@/path/to/your/image.jpg" -F 'parameters={"convert_to":"grayscale", "blur": 5}' -o processed_image.jpg
