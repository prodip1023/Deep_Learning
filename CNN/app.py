import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


MODEL_PATH = '/Users/hackthebox/Downloads/Deep_Learning/CNN/vanila_cnn_model.h5'

model = load_model(MODEL_PATH)

class_name = ['anu','mou','prodip','sudh']

st.set_page_config(page_title="Image Classification",layout='centered')
st.sidebar.title("Upload your image")
st.markdown("This appilication will try to give your a classification of your image its build based on vanila CNN architecture")
upload_file = st.sidebar.file_uploader("Choose your image",type=["jpeg","jpg","png"])
if upload_file is not None:
    img = Image.open(upload_file).convert("RGB")
    st.image(img,caption="Your image")
    image_resize = img.resize((128,128))
    img_array = image.img_to_array(image_resize)/255.0
    img_batch = np.expand_dims(img_array,axis=0)
    prediction = model.predict(img_batch)
    predicted_class = class_name[np.argmax(prediction)]

    st.success(f"This Image is predicted to be :{predicted_class}")
    st.subheader("Below is your confidence score for all the class")
    print(prediction)
    for index,score in enumerate(prediction[0]):
        st.write(f"{class_name[index]} : {score*100:.2f}%")


