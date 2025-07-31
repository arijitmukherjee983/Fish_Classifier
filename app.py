from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model/FishModelClassifier_V6.h5")

# Class names (update as per your model)
class_names = ['Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 
               'Fourfinger Threadfin', 'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 
               'Gourami', 'Grass Carp', 'Green Spotted Puffer', 'Indian Carp', 'Indo-Pacific Tarpon',
                 'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish', 'Mosquito Fish',
                   'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
                   'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia']

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # update size if needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            img_array = prepare_image(file_path)
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]

            return render_template("index.html", prediction=predicted_class, img_path=file_path)

    return render_template("index.html", prediction=None)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

