import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load the saved model
model = load_model('my_cnn_model.h5')

# Define the target size for the images
target_size = (204, 136)

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    # Convert the image to RGB
    image = image.convert('RGB')
    # Resize the image
    img = np.array(image)
    resized_img = cv2.resize(img, target_size)
    resized_img = resized_img / 255.0
    return np.expand_dims(resized_img, axis=0)

# Streamlit app
st.title("Plant Pathology Classification 🌿")
st.write("This is a foliar diseases in apple trees identification model. Upload leaf image and get diagnostics!")
st.write("You can also try uploading these sample images below.")

# Load the images and captions
images = ["scab.jpg", "healthy.png", "rust.png"]
captions = ["Scab", "Healthy", "Rust"]

# Create columns and display images
cols = st.columns(3)
for col, image, caption in zip(cols, images, captions):
    col.image(image, caption=caption)

# Upload an image
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)[0]

    # Define the class names
    class_names = ['healthy', 'multiple_diseases', 'rust', 'scab']

    # Interpret the predictions
    if predictions[0] > 0.5:
        message = "The plant looks healthy! 💪"
    elif predictions[1] > 0.5:
        message = "The plant has both rust and scab diseases! 💀"
    elif predictions[2] > 0.5:
        message = "The plant has rust disease! 🤒"
    elif predictions[3] > 0.5:
        message = "The plant has scab disease! 🤒"
    else:
        message = "Unable to determine the condition of the plant! 🤔"

    # Display the message
    st.header(message)

    # Display the raw predictions (optional)
    st.write("Raw Predictions:")
    for i, class_name in enumerate(class_names):
        st.write(f"***{class_name}*** possibility is {round(predictions[i] * 100):.0f}%")

    st.write('BONUS: This is the ROC accuracy graph of the used model.')
    st.image('ROC.png')