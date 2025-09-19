import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import random
import time
import base64
import io

# --- Conceptual Introduction to Neural Networks ---
# A neural network is a series of interconnected layers of "neurons."
# Each neuron takes in data, processes it, and passes it to the next layer.
#
# Our simple app would use a Convolutional Neural Network (CNN) with:
# - Input Layer: The 28x28 pixel image of the number.
# - Convolutional Layers: These layers contain "neurons" that scan the image
#   to identify features like edges, curves, and corners. The more neurons,
#   the more detailed features it can learn.
# - Dense (or Fully Connected) Layers: These layers take the learned features
#   and make a final prediction. A layer with more neurons can handle more
#   complex decision-making.
# - Output Layer: A final set of neurons that gives us the prediction (0-9).

def mock_predict(image_array):
    """
    This function simulates a model prediction.
    In a real app, you would have a more complex model with multiple layers.
    For example, a model trained on the MNIST dataset with thousands of neurons.
    """
    st.write("Processing...")
    time.sleep(1) # Simulate a small processing delay
    
    # For demonstration, we'll return a random number.
    # You would replace this with your actual model's prediction.
    return random.randint(0, 9)

# File path for conceptual feedback storage
FEEDBACK_FILE = "https://raw.githubusercontent.com/Excitedicecream/CSV-Files/refs/heads/main/Number.csv"

def save_feedback(image_data, prediction, correct_number):
    """
    Saves the user's feedback to a CSV file.
    Note: This is a conceptual function. In a real web application, this
    would require a persistent file system and proper access permissions.
    """
    # Convert image data to base64 string to store in the CSV
    buffered = io.BytesIO()
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Create a new DataFrame for the new feedback entry
    feedback_entry = pd.DataFrame([{
        "image_data_b64": img_str,
        "predicted_number": prediction,
        "correct_number": correct_number,
        "timestamp": pd.Timestamp.now()
    }])

    # Try to load existing data and append the new entry
    try:
        existing_df = pd.read_csv(FEEDBACK_FILE)
        updated_df = pd.concat([existing_df, feedback_entry], ignore_index=True)
    except FileNotFoundError:
        updated_df = feedback_entry
    
    # Save the updated DataFrame back to the CSV file
    updated_df.to_csv(FEEDBACK_FILE, index=False)

def main():
    st.set_page_config(
        page_title="Deep Learning Number Recognizer",
        page_icon="🤖"
    )

    st.title("Deep Learning Number Recognizer")
    st.markdown(
        "Draw a single digit in the canvas below and click 'Recognize' "
        "to see the app's prediction. "
        "This app uses a conceptual model with a mock prediction function."
    )

    # Use session state to store values across reruns
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "image_data" not in st.session_state:
        st.session_state.image_data = None
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False

    # Create a canvas component
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )

    st.markdown("---")

    if st.button("Recognize Number"):
        if canvas_result.image_data is not None:
            # Store the data in session state for later use
            st.session_state.image_data = canvas_result.image_data
            
            # Convert the RGBA numpy array to a PIL Image
            img = Image.fromarray(st.session_state.image_data.astype('uint8'), 'RGBA')
            img_resized = img.convert('L').resize((28, 28))

            # Convert the image to a numpy array for model input
            img_array = np.array(img_resized)
            img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0

            # The prediction step
            st.session_state.prediction = mock_predict(img_array)

            st.session_state.show_feedback = True
        else:
            st.error("Please draw a number on the canvas first!")
    
    # Show prediction and feedback buttons only after a prediction is made
    if st.session_state.show_feedback and st.session_state.prediction is not None:
        st.success(f"I predict this is the number: **{st.session_state.prediction}**")

        st.markdown("---")
        st.markdown("Was the prediction correct?")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Correct"):
                # Conceptually save correct feedback
                save_feedback(st.session_state.image_data, st.session_state.prediction, st.session_state.prediction)
                st.success("Great! Your feedback has been noted. The model 'improves' with this data.")
                st.session_state.show_feedback = False
        with col2:
            if st.button("❌ Wrong"):
                st.session_state.wrong_feedback_given = True
    
    if st.session_state.get("wrong_feedback_given"):
        correct_number = st.number_input(
            "What was the correct number?",
            min_value=0,
            max_value=9,
            step=1,
            key="correct_num_input"
        )
        if st.button("Submit Correct Answer"):
            # Conceptually save wrong feedback
            save_feedback(st.session_state.image_data, st.session_state.prediction, correct_number)
            st.info(f"Thank you for the feedback! The correct number was {correct_number}. This data has been conceptually saved.")
            st.session_state.show_feedback = False
            st.session_state.wrong_feedback_given = False

if __name__ == "__main__":
    main()
