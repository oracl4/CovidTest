import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

def import_and_predict(image_data, model):
    size = (150,150)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC))/255.    
    img_reshape = img_resize[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction

model = tf.keras.models.load_model('Resnet_Covid19Model.h5')

st.write("""
         # Covid-19 X-Ray Predictions
         """
         )

st.write("This is a simple image classification web app to predict Covid-19 based on X-Ray Images")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)

    st.image(image, use_column_width=True)
    
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("Result : Covid-19")
    elif np.argmax(prediction) == 1:
        st.write("Result : Normal")
    else:
        st.write("Result : Viral Pneumonia")
    
    st.text("Probability (0: Covid-19, 1: Normal, 2: Viral Pneumonia)")
    st.write(prediction)