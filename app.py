from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the trained GAN generator
model = load_model("GAN_Denoising_Generator.keras")


@app.route("/", methods=["GET", "POST"])
def home():

    denoised_image = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            # Open uploaded image
            image = Image.open(file).convert("RGB")

            # Resize to model input size
            image = image.resize((128, 128))

            # Convert image to NumPy array
            image_array = np.array(image).astype("float32")

            # Normalize from [0, 255] to [-1, 1]
            image_array = image_array / 127.5 - 1

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            # Generate denoised image
            result = model.predict(image_array, verbose=0)

            # Convert output from [-1, 1] to [0, 1]
            result = (result[0] + 1) / 2

            # Keep pixel values between 0 and 1
            result = np.clip(result, 0, 1)

            # Create static folder if it does not exist
            os.makedirs("static", exist_ok=True)

            # Save denoised image
            output_path = "static/denoised.png"

            Image.fromarray(
                (result * 255).astype("uint8")
            ).save(output_path)

            denoised_image = output_path

    return render_template(
        "index.html",
        denoised_image=denoised_image
    )


if __name__ == "__main__":
    app.run(debug=True)
