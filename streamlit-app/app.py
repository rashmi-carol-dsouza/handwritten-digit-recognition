import streamlit as st
import tensorflow as tf
from keras.models import load_model
# make a prediction for a new image.
import keras
import numpy as np
from numpy import argmax
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from PIL import Image, ImageOps
import requests
import os


st.title("Handwritten Digit Classification")

# Define the URL where the model is hosted

google_drive_url = "https://drive.google.com/file/d/16XFgFfhUS7hN5xgidqxHNwm3hKYIbt5k/view?usp=drive_link"

# Function to download the model
@st.cache(allow_output_mutation=True)
def load_model_st(google_drive_url):
    # Ensure the directory exists before writing the file
    model_path = "models/base-model-2.h5"
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Download the model from Google Drive
    response = requests.get(google_drive_url)
    with open(model_path, "wb") as f:
        f.write(response.content)

    # Load the model from the local file
    model = load_model(model_path)
    return model

# Check if the model exists, and if not, download it
if not os.path.exists("models/base-model-2.h5"):
    st.info("Downloading the model file...")
    with st.spinner('Model is being loaded..'):
        model = load_model_st(google_drive_url)
else:
    with st.spinner('Model is being loaded..'):
      model = load_model("models/base-model-2.h5")


# load and prepare the image
def load_image(file):
    # load the image
    img = load_img(file, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


def import_and_predict(image,model):
    # load the image
    img = load_image(file)
    # predict the class
    predicted_probs = model.predict(img)
    # Get the predicted digit
    predicted_digit = np.argmax(predicted_probs)
    # Get the confidence score for the predicted digit
    confidence_score = np.max(predicted_probs)
    
    return predicted_digit, confidence_score


file = st.file_uploader("Please upload an image of a digit you would like to classify", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)



if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, width=100)
    prediction, confidence = import_and_predict(image, model)
    percentage = int(confidence*100)
    string = F"The digit in this image is {prediction} with a {percentage}% confidence."
    st.success(string)





 

 

