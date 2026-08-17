import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from pathlib import Path
from PIL import Image

MODEL_FILE_ID = "1zDDkguVcGBjuqbzYSfqpoJZOIoD2sl3F"
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "train_plant_model.keras"


@st.cache_resource(show_spinner="Loading disease detection model...")
def load_disease_model():
    """Load the bundled model, downloading it once when Streamlit Cloud lacks it."""
    if not MODEL_PATH.is_file():
        gdown.download(id=MODEL_FILE_ID, output=str(MODEL_PATH), quiet=True)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError("The model download failed. Check that the Google Drive file is shared with 'Anyone with the link'.")

    return tf.keras.models.load_model(MODEL_PATH)

def model_prediction(test_image):
    model = load_disease_model()
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])
#app_mode = st.sidebar.selectbox("Select Page",["Home"," ","Disease Recognition"])

img = Image.open(APP_DIR / "diseases.png")

# display image using streamlit
# width is used to set the width of an image
st.image(img)

#Main Page
if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)
    
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Early_Blight', 'Healthy', 'Late_Blight']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

