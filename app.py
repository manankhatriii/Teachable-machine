import streamlit as st
import tensorflow as tf
import tf_keras
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import os
import time

st.markdown("""
    <style>
    body {
        font-family: "Times New Roman";
        background-color: #F5F5F5;
    }
    .main-title {
        text-align: center;
        color: #4A4A4A;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 40px;
    }
    .stTextInput>div>input, .stSelectbox>div>div>div>div>div>input {
        border-radius: 8px;
        background-color: #E8E8E8;
        border: none;
        padding: 10px;
        font-size: 18px;
        color: #333;
    }
    .stButton>button {
        background-color: #7393B3;
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 16px;
        border: none;
    }
    .stButton {
    text-align: center;
    color: white;
    }
    .stButton>button {
        display: inline-block;
        color: white !important; 
    }
    .stButton>button:hover {
        background-color: #6A9C89;
        color: white;
    }
    .stButton>button:active {
        background-color: #4F725E;
        color: white;
    }
    .summary {
        margin-top: 40px;
        font-size: 20px;
        color: #555;
        text-align: center;
    }
    .column-title {
        font-size: 22px;
        font-weight: 600;
        color: #3C3CFF;
        margin-bottom: 10px;
    }
    .uploaded-images {
        font-size: 18px;
        color: #4A4A4A;
    }
    .error {
        color: red;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>MNM's Teachable Machine</h1>", unsafe_allow_html=True)

MIN_IMAGES_PER_CLASS = 25
IMAGE_SIZE = (224, 224)
BASE_MODEL = hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", 
    trainable=False
)

num_classes = st.selectbox("Select number of classes:", range(2, 11), index=0)

class_names = []
image_data = {}

def load_preprocess_image(image):
    img = Image.open(image)
    img= img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img = np.array(img) / 255.0
    return img

cols = st.columns(2)
for i in range(1, num_classes + 1):
    col = cols[(i-1) % 2]

    with col:
        class_name = st.text_input(f"Enter name for Class {i}", f"Class {i}", key=f'class_name_{i}')
        class_names.append(class_name)
        images = st.file_uploader(f"Class {i}:", accept_multiple_files=True, type=["jpg", "png", "jpeg"], key=f"upload_{i}")
        if images:
            image_data[class_name] = images

condition_check = True
st.write("<div class='summary'>Classes summary:</div>", unsafe_allow_html=True)
for class_name, images in image_data.items():
    st.write(f"<div class='uploaded-images'>{class_name}: {len(images)} images uploaded.</div>", unsafe_allow_html=True)
    if len(images) < MIN_IMAGES_PER_CLASS:
        condition_check = False
        st.markdown(f"<span class='error'>* {class_name} must have at least {MIN_IMAGES_PER_CLASS} images.</span>", unsafe_allow_html=True)

if not condition_check:
    st.stop()

if st.button("Start Training"):
    train_images, train_labels = [], []
    val_images, val_labels = [], []
    test_images, test_labels = [], []

    class_map = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name, images in image_data.items():
        images_list = list(images)
        np.random.shuffle(images_list)
        
        train_split = int(0.6 * len(images_list))
        val_split = int(0.8 * len(images_list)) 

        train_set = images_list[:train_split]
        val_set = images_list[train_split:val_split]
        test_set = images_list[val_split:]

        train_images += [load_preprocess_image(img) for img in train_set]
        val_images += [load_preprocess_image(img) for img in val_set]
        test_images += [load_preprocess_image(img) for img in test_set]

        train_labels += [class_map[class_name]] * len(train_set)
        val_labels += [class_map[class_name]] * len(val_set)
        test_labels += [class_map[class_name]] * len(test_set)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    with tf.device("/GPU:0"):
        model = tf_keras.Sequential([
            BASE_MODEL,
            tf_keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    progress_bar = st.progress(0)
    
    class ProgressCallback(tf_keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / self.params['epochs']
            progress_bar.progress(progress)
            
    early_stopping = tf_keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    with tf.device("/GPU:0"):
        history = model.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            epochs=5,
            batch_size=32,
            callbacks=[ProgressCallback(), early_stopping]
        )

    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    st.write("Training complete.")
    st.write(f"Final Test accuracy: {test_accuracy * 100:.2f}%")
    
    st.session_state['trained_model']= model
    st.session_state['class_map']= class_map
    
if 'trained_model' in st.session_state:
    model = st.session_state['trained_model']
    model_save_path = "trained_model.h5"
    model.save(model_save_path)
    
    with open(model_save_path, 'rb') as f:
        st.download_button(
            label="Download Trained Model",
            data=f,
            file_name=model_save_path,
            mime='application/x-hdf'
        )    

st.markdown("<h2 class='main-title'>Predict Image Class</h2>", unsafe_allow_html=True)
uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png", "jpeg"])

if uploaded_image and 'trained_model' in st.session_state:
    model= st.session_state['trained_model']
    class_map= st.session_state['class_map']
    
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    image = load_preprocess_image(uploaded_image)
    image = np.expand_dims(image, axis=0)

    with tf.device("/GPU:0"):
        prediction = model.predict(image)
    
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Predicted Class: {predicted_class}")

else:
    if not uploaded_image:
        st.write("Please upload an image to classify.")
    
    if 'trained_model' not in st.session_state:
        st.write("Model is not trained yet. Please train the model before making predictions.")