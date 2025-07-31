from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/fish_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names
class_names = ['Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 
               'Fourfinger Threadfin', 'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 
               'Gourami', 'Grass Carp', 'Green Spotted Puffer', 'Indian Carp', 'Indo-Pacific Tarpon',
               'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish', 'Mosquito Fish',
               'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
               'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia']

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array.astype(np.float32)  # Make sure dtype matches model input

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            img_array = prepare_image(file_path)

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Get prediction and confidence
            prediction_scores = output_data[0]
            max_score = np.max(prediction_scores)
            predicted_index = np.argmax(prediction_scores)
            confidence = round(float(max_score) * 100, 2)  # Convert to percentage

            if max_score >= 0.5:
                predicted_class = class_names[predicted_index]
            else:
                predicted_class = "No fish detected"
                confidence = None

            return render_template("index.html", prediction=predicted_class, confidence=confidence, img_path=file_path)

    return render_template("index.html", prediction=None, confidence=None)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
