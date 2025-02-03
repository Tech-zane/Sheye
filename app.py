import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

@st.cache_resource
def load_and_train_model():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784) / 255.0
    
    model = keras.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15, validation_split=0.2, verbose=0)
    return model, X_train, y_train

model, X_train, y_train = load_and_train_model()

# Rest of your UI code with color inversion and retraining improvements...

# Rest of your Streamlit UI code...

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
            # Add 10 augmented variations of the corrected image
            for _ in range(10):
                # Add slight random rotations/translations
                augmented_img = img_array + np.random.normal(0, 0.1, (28,28))
                X_train = np.append(X_train, augmented_img.reshape(1,784), axis=0)
                y_train = np.append(y_train, correct_label)
            
            # Retrain with lower learning rate
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=10, verbose=0)  # More epochs
            st.success("Model updated with new training data! 🚀")

