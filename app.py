import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os

# Load and train model once (caches to avoid retraining every time)
@st.cache_resource
def load_and_train_model():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train_flat = X_train.reshape(len(X_train), 28 * 28)
    X_test_flat = X_test.reshape(len(X_test), 28 * 28)

    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_flat, y_train, epochs=5, verbose=0)
    
    return model, X_train_flat, y_train

# Load model and dataset
model, X_train_flat, y_train = load_and_train_model()

st.title("Shustah's Eye")
st.write("Draw a digit below and let the AI predict it. If wrong, correct it to improve the model!")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",  
    stroke_width=20,                
    stroke_color="#FFFFFF",         
    background_color="#000000",     
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    canvas_image = Image.fromarray(canvas_result.image_data.astype('uint8'))
    img_gray = canvas_image.convert('L')
    img_resized = img_gray.resize((28, 28), Image.LANCZOS)
    
    img_array = np.array(img_resized) / 255.0
    img_flat = img_array.reshape(1, 28 * 28)

    # Predict the digit
    prediction = model.predict(img_flat)
    predicted_label = np.argmax(prediction)
    
    st.subheader("Prediction")
    st.write(f"**Predicted Digit:** {predicted_label}")
    st.image(img_resized, caption="Processed Input (28x28)", width=140)

    # Ask user if the prediction is correct
    correct_label = st.selectbox("Correct the prediction if needed:", list(range(10)), index=int(predicted_label))

    if st.button("Submit Feedback & Retrain"):
        if correct_label != predicted_label:
            st.write(f"Updating model: {predicted_label} → {correct_label}")

            # Add corrected data to the training set
            X_train_flat = np.append(X_train_flat, img_flat, axis=0)
            y_train = np.append(y_train, correct_label)

            # Retrain model
            model.fit(X_train_flat, y_train, epochs=1, verbose=0)
            st.success("Model updated with new training data! 🚀")

