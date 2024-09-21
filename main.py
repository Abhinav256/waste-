import streamlit as st
import tensorflow as tf
import numpy as np
import base64

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        base64_image = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://upload.wikimedia.org/wikipedia/commons/4/49/A_black_image.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Tensorflow Model Prediction for Waste Classification
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_garbage_classification_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Dictionary to hold waste type descriptions or actions (if needed)
waste_descriptions = {
    'cardboard': 'This is cardboard, which can be recycled.',
    'glass': 'This is glass, which can be recycled in specific bins.',
    'metal': 'This is metal, which can be recycled.',
    'paper': 'This is paper, which can be recycled.',
    'plastic': 'This is plastic, which can often be recycled.',
    'trash': 'This is trash and should be disposed of properly.'
}

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Waste Classification"])

if app_mode == "Waste Classification":
    st.header("Waste Classification")
    set_background("home_page.jpg")
    
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
        
        if st.button("Predict"):
            st.balloons()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            # Waste class names corresponding to the trained model
            class_name = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
            waste_type = class_name[result_index]
            
            st.success(f"Model predicts it's {waste_type}")
            st.write("Description:")
            st.info(waste_descriptions[waste_type])
